# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.amp as amp
from datetime import datetime
from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# ============================
# 配置
# ============================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
SPLIT_DIR = r"splits_musid"
SAVE_ROOT = r"Hashmani's Dataset/FusionCache_1024x576_interface"

# 权重路径
RGHNET_CKPT = r"rghnet_best_c2.pth"
DCE_WEIGHTS = r"Epoch99.pth"

# 统一尺寸
UNET_IN_W = 1024
UNET_IN_H = 576

# sinogram 统一尺寸
RESIZE_H = 2240
RESIZE_W = 180

# 参数
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

# seg-mask -> Radon 通道生成方式
# - "interface": 只提取 sky/non-sky 主交界线(按列取 sky 下边界)并可选 RANSAC 捋直，再做 Radon（推荐，与你论文设想一致）
# - "canny":     直接对二值 mask 做 Canny 得到所有边缘，再做 Radon（旧基线）
SEG_EDGE_MODE = "interface"

# ========= 门控回退（建议开启） =========
# 目标：当 UNet 的 sky/non-sky 掩码非常破碎或边界明显不可信时，
# 不强行用 interface-line Radon（容易把第4通道带偏），而是回退到更“粗糙但稳”的方案。
ENABLE_SEG_GATING = True

# 回退方式：
# - "canny": 回退到 old baseline（对 mask 做 Canny 得到所有边缘，再 Radon）
# - "zeros": 直接把第4通道置零（让 CNN 只依赖传统3通道）
SEG_FALLBACK_MODE = "canny"

# 门控阈值（在 1024x576 尺度上调参）
GATE_SKY_RATIO_MIN = 0.05      # sky 像素占比过低/过高通常说明 mask 崩了
GATE_SKY_RATIO_MAX = 0.95
GATE_COL_SKY_MIN_RATIO = 0.60  # 至少 60% 的列需要存在 sky，否则交界线不稳定

# 用“交界线点集拟合到直线”的残差 RMSE 衡量边界是否过于波浪/破碎（单位：像素）
GATE_BOUNDARY_RMSE_MAX = 8.0

# 检查“边界下方(海面区域)仍被预测为 sky”的泄漏比例
GATE_LEAK_MARGIN_PX = 5        # 从边界往下留一点 margin，避免线宽/平滑导致误判
GATE_LEAK_RATIO_MAX = 0.02     # 海面区域若 >2% 仍是 sky，说明 mask 很破碎

# 计算泄漏比例时采样列数（越大越准，越小越快；W=1024 时 256 很合适）
GATE_LEAK_SAMPLE_COLS = 256

# interface-line 细节
INTERFACE_THICKNESS = 2          # 画线粗细（建议 1~3）
INTERFACE_SMOOTH_K = 15          # y(x) 平滑窗口（奇数更好，<=1 表示不平滑）
INTERFACE_USE_RANSAC = True      # True: RANSAC 捋直为直线；False: 直接用 polyline
RANSAC_ITERS = 200
RANSAC_THRESH_PX = 3.0           # inlier 距离阈值（在 1024x576 尺度上）
RANSAC_MIN_INLIER_RATIO = 0.35   # 内点比例阈值，低于则退回 polyline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ================= seg-mask -> Radon 辅助 =================
def _extract_sky_bottom_boundary(mask_pp: np.ndarray):
    """
    mask_pp: (H,W) uint8/bool, sky=1, non-sky=0
    return: xs (N,), ys (N,) float32  (只保留有 sky 的列)
    """
    sky = (mask_pp > 0).astype(np.uint8)
    H, W = sky.shape
    col_has = sky.sum(axis=0) > 0
    if col_has.sum() < 2:
        return None, None, W, H

    rev = sky[::-1, :]  # 从底往上
    idx_from_bottom = rev.argmax(axis=0)  # 每列第一个 1 的位置
    ys = (H - 1 - idx_from_bottom).astype(np.float32)
    ys[~col_has] = np.nan
    xs = np.arange(W, dtype=np.float32)

    # 缺失列插值
    valid = np.isfinite(ys)
    ys = np.interp(xs, xs[valid], ys[valid]).astype(np.float32)

    return xs, ys, W, H


def _smooth_1d(y: np.ndarray, k: int):
    if k is None or k <= 1:
        return y
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(y, kernel, mode="same").astype(np.float32)


def _ransac_fit_line_normal(xs: np.ndarray, ys: np.ndarray, iters: int = 200, thresh: float = 3.0, seed: int = 2024):
    """
    用 RANSAC 拟合 ax + by + c = 0 （归一化），返回 (a,b,c,inlier_mask,inlier_ratio)
    """
    rng = np.random.default_rng(seed)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    n = pts.shape[0]
    if n < 2:
        return None

    best_inliers = None
    best_cnt = -1
    best_abc = None

    for _ in range(int(iters)):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        # 两点确定直线： (y1-y2)x + (x2-x1)y + (x1*y2-x2*y1)=0
        a = (y1 - y2)
        b = (x2 - x1)
        c = (x1 * y2 - x2 * y1)
        norm = np.hypot(a, b)
        if norm < 1e-6:
            continue
        a, b, c = a / norm, b / norm, c / norm

        d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inliers = d <= thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
            best_abc = (a, b, c)

    if best_inliers is None or best_cnt < 2:
        return None

    inlier_ratio = best_cnt / float(n)

    # 用 inliers 做一次最小二乘（总最小二乘 / PCA）精修法向量
    P = pts[best_inliers]
    mu = P.mean(axis=0)
    X = P - mu
    # 2x2 协方差
    cov = (X.T @ X) / max(P.shape[0], 1)
    # 最小特征值对应法向量
    w, v = np.linalg.eigh(cov)
    normal = v[:, 0]  # (nx, ny)
    a, b = normal[0], normal[1]
    norm = np.hypot(a, b)
    if norm < 1e-6:
        a, b, c = best_abc
    else:
        a, b = a / norm, b / norm
        c = -(a * mu[0] + b * mu[1])

    return (float(a), float(b), float(c), best_inliers, float(inlier_ratio))


def _line_intersections_with_image(a: float, b: float, c: float, W: int, H: int):
    """
    返回与图像边界的两个交点（尽量）。图像坐标：x in [0,W-1], y in [0,H-1]
    """
    pts = []

    # 与 x=0, x=W-1
    for x in [0.0, float(W - 1)]:
        if abs(b) > 1e-6:
            y = -(a * x + c) / b
            if 0 <= y <= H - 1:
                pts.append((x, y))
    # 与 y=0, y=H-1
    for y in [0.0, float(H - 1)]:
        if abs(a) > 1e-6:
            x = -(b * y + c) / a
            if 0 <= x <= W - 1:
                pts.append((x, y))

    # 去重
    uniq = []
    for p in pts:
        if all((abs(p[0]-q[0]) > 1e-3 or abs(p[1]-q[1]) > 1e-3) for q in uniq):
            uniq.append(p)

    if len(uniq) >= 2:
        return uniq[0], uniq[1]
    elif len(uniq) == 1:
        return uniq[0], uniq[0]
    else:
        # 退化情况：返回水平中线
        return (0.0, H/2.0), (float(W-1), H/2.0)


def build_interface_line_image(mask_pp: np.ndarray,
                               thickness: int = 2,
                               smooth_k: int = 15,
                               use_ransac: bool = True,
                               ransac_iters: int = 200,
                               ransac_thresh: float = 3.0,
                               ransac_min_inlier_ratio: float = 0.35):
    """
    生成“白底黑线”的 sky/non-sky 主交界线图（实际上输出为 uint8 图：线=255，背景=0）
    - 先按列取 sky 的最底部边界 y(x)
    - 可选：RANSAC 把边界捋直成直线（更适合 horizon）
    """
    xs, ys, W, H = _extract_sky_bottom_boundary(mask_pp)
    line_img = np.zeros((H, W), dtype=np.uint8)
    if xs is None:
        return line_img

    ys = _smooth_1d(ys, smooth_k)

    if use_ransac:
        res = _ransac_fit_line_normal(xs, ys, iters=ransac_iters, thresh=ransac_thresh)
        if res is not None:
            a, b, c, inliers, ratio = res
            if ratio >= ransac_min_inlier_ratio:
                p1, p2 = _line_intersections_with_image(a, b, c, W, H)
                cv2.line(line_img,
                         (int(round(p1[0])), int(round(p1[1]))),
                         (int(round(p2[0])), int(round(p2[1]))),
                         255, int(thickness))
                return line_img
        # 如果 RANSAC 失败/质量太差，则退回 polyline（仍然只用交界线信息，不回退到 Canny）
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(line_img, [pts], isClosed=False, color=255, thickness=int(thickness))
    return line_img


def assess_interface_mask_quality(mask_pp: np.ndarray,
                                 smooth_k: int = 15,
                                 sky_ratio_min: float = 0.05,
                                 sky_ratio_max: float = 0.95,
                                 col_sky_min_ratio: float = 0.60,
                                 boundary_rmse_max: float = 8.0,
                                 leak_margin_px: int = 5,
                                 leak_ratio_max: float = 0.02,
                                 leak_sample_cols: int = 256):
    """\
    用非常轻量的规则判断：当前 sky/non-sky mask 是否足够“可信”，适合用 interface-line Radon。

    返回：
      ok: bool
      info: dict（可用于 debug 打印）
    """
    info = {}

    sky = (mask_pp > 0)
    H, W = sky.shape

    sky_ratio = float(sky.mean())
    info["sky_ratio"] = sky_ratio
    if sky_ratio < float(sky_ratio_min) or sky_ratio > float(sky_ratio_max):
        info["reason"] = "sky_ratio_out_of_range"
        return False, info

    col_has = (sky.sum(axis=0) > 0)
    col_ratio = float(col_has.mean())
    info["col_sky_ratio"] = col_ratio
    if col_ratio < float(col_sky_min_ratio):
        info["reason"] = "too_many_empty_columns"
        return False, info

    # 提取并平滑边界 y(x)
    rev = sky[::-1, :]
    idx_from_bottom = rev.argmax(axis=0)
    ys = (H - 1 - idx_from_bottom).astype(np.float32)
    ys[~col_has] = np.nan
    xs = np.arange(W, dtype=np.float32)

    valid = np.isfinite(ys)
    if int(valid.sum()) < 2:
        info["reason"] = "not_enough_boundary_points"
        return False, info
    ys = np.interp(xs, xs[valid], ys[valid]).astype(np.float32)
    ys = _smooth_1d(ys, smooth_k)

    # 1) 用“拟合直线的残差 RMSE”衡量边界是否过于波浪/破碎
    try:
        k1, k0 = np.polyfit(xs, ys, deg=1)
        pred = k1 * xs + k0
        rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    except Exception:
        rmse = 1e9
    info["boundary_rmse"] = rmse
    if rmse > float(boundary_rmse_max):
        info["reason"] = "boundary_too_wavy"
        return False, info

    # 2) 检查边界下方是否存在大量 sky 泄漏（海面碎 sky 小岛 / 大块误分）
    sample_n = int(min(max(8, leak_sample_cols), W))
    cols = np.linspace(0, W - 1, sample_n).round().astype(np.int32)
    cols = np.unique(cols)

    leak = 0
    denom = 0
    m = int(max(0, leak_margin_px))
    for x in cols:
        y0 = int(round(float(ys[x]) + m))
        y0 = 0 if y0 < 0 else (H if y0 > H else y0)
        if y0 >= H:
            continue
        denom += (H - y0)
        leak += int(sky[y0:, int(x)].sum())
    leak_ratio = float(leak / denom) if denom > 0 else 0.0
    info["leak_ratio"] = leak_ratio
    if leak_ratio > float(leak_ratio_max):
        info["reason"] = "too_much_sky_leak_below_boundary"
        return False, info

    info["reason"] = "ok"
    return True, info
# =========================================================

def _read_image_any_ext(img_dir: str, img_name: str):
    candidates = [img_name, img_name + ".jpg", img_name + ".JPG", img_name + ".png"]
    for f in candidates:
        p = os.path.join(img_dir, f)
        if os.path.exists(p): return cv2.imread(p)
    return None

def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6:
        sino_norm = (sino - mi) / (ma - mi)
    else:
        sino_norm = np.zeros_like(sino, dtype=np.float32)
    h_curr = sino_norm.shape[0]
    container = np.zeros((target_h, target_w), dtype=np.float32)
    start_h = (target_h - h_curr) // 2
    if h_curr <= target_h:
        container[start_h:start_h + h_curr, :] = sino_norm
    else:
        crop_start = (h_curr - target_h) // 2
        container[:, :] = sino_norm[crop_start:crop_start + target_h, :]
    return container

def calculate_radon_label(x1, y1, x2, y2, img_w, img_h, resize_h, resize_w):
    # 计算中心偏移
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1
    
    # 角度计算
    line_angle = np.arctan2(dy, dx)
    theta_rad = line_angle - np.pi / 2
    while theta_rad < 0: theta_rad += np.pi
    while theta_rad >= np.pi: theta_rad -= np.pi
    
    # Rho 计算
    # 公式：x * cos + y * sin = rho
    # 使用中点计算
    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)

    # 映射到 Pixel Index
    # Radon 变换后的 sinogram 高度对应图像对角线长度
    # 但我们统一 Pad 到了 RESIZE_H (2240)
    # 所以要基于 RESIZE_H 进行归一化
    
    # 图像的物理对角线长度
    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    
    # rho = 0 对应中心
    rho_pixel_pos = rho + original_diag / 2.0
    
    # 加上 padding 的偏移量
    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top
    
    label_rho = final_rho_idx / (resize_h - 1)
    label_theta = np.rad2deg(theta_rad) / 180.0
    
    return float(np.clip(label_rho, 0, 1)), float(np.clip(label_theta, 0, 1))

def post_process_mask_top_connected(mask_np):
    # 简化的连通域处理
    valid = (mask_np != 255)
    sky = ((mask_np == 1) & valid).astype(np.uint8)
    # 闭运算连接断裂
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE, MORPH_CLOSE))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    # 连通域分析
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_TOP] <= TOP_TOUCH_TOL:
            keep[labels == i] = 1
    out = mask_np.copy()
    out[(mask_np == 1) & (keep == 0)] = 0
    return out

def _load_split_indices(split_dir):
    tr = np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64).tolist()
    va = np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64).tolist()
    te = np.load(os.path.join(split_dir, "test_indices.npy")).astype(np.int64).tolist()
    return {"train": tr, "val": va, "test": te}

def build_cache_for_split(df, indices, out_dir, seg_model, detector, theta_scan):
    ensure_dir(out_dir)
    split_name = os.path.basename(out_dir)
    print(f"Processing {split_name}: {len(indices)}")

    # === Gate 日志：用于你后续统计/调参 GATE_BOUNDARY_RMSE_MAX 与 GATE_LEAK_RATIO_MAX ===
    gate_log_fp = None
    gate_log_path = None
    if str(SEG_EDGE_MODE).lower().strip() == "interface" and bool(ENABLE_SEG_GATING):
        ensure_dir("eval_outputs")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gate_log_path = os.path.join("eval_outputs", f"gate_fallback_{split_name}_{ts}.txt")
        gate_log_fp = open(gate_log_path, "w", encoding="utf-8")
        gate_log_fp.write("# gate log (one line per sample)\\n")
        gate_log_fp.write(f"# split={split_name}  fallback_mode={SEG_FALLBACK_MODE}\\n")
        gate_log_fp.write(
            "idx\\timg_name\\tgate_ok\\tuse_interface\\treason\\tsky_ratio\\tcol_sky_ratio\\tboundary_rmse\\tleak_ratio\\tfallback_mode\\n"
        )

    def _fmt(x, nd=6):
        if x is None:
            return ""
        try:
            if isinstance(x, (float, int, np.floating, np.integer)):
                if np.isfinite(float(x)):
                    return f"{float(x):.{nd}f}"
                return ""
        except Exception:
            pass
        return str(x)


    # 统计门控回退情况（便于你写论文对比/做消融）
    n_interface = 0
    n_canny = 0
    n_fallback = 0
    
    for idx in tqdm(indices, ncols=80):
        row = df.iloc[idx]
        img_name = str(row.iloc[0])
        try:
            # 原始坐标 (1920x1080)
            x1_org, y1_org = float(row.iloc[1]), float(row.iloc[2])
            x2_org, y2_org = float(row.iloc[3]), float(row.iloc[4])
        except: continue

        bgr = _read_image_any_ext(IMG_DIR, img_name)
        if bgr is None: continue
        
        h_orig, w_orig = bgr.shape[:2]
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Resize 到 UNet 尺寸 (1024x576)
        rgb_unet = cv2.resize(rgb0, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(rgb_unet.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2. UNet 推理 (C2)
        with torch.no_grad():
            with amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                restored_t, seg_logits, _ = seg_model(inp, None, True, True)
        
        restored_np = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
        mask_np = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # 3. 后处理
        mask_pp = post_process_mask_top_connected(mask_np)

        # 4. 特征提取 (基于 1024x576 的图像)
        try:
            _, _, _, trad_sinos = detector.detect(restored_bgr)
        except: trad_sinos = []
        
        processed_stack = []
        for s in trad_sinos[:3]:
            processed_stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

        # --- seg-mask -> Radon (第4通道) ---
        seg_mode = str(SEG_EDGE_MODE).lower().strip()

        if seg_mode == "canny":
            # 旧基线：mask 全边缘
            seg_img = cv2.Canny((mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
            if EDGE_DILATE > 0:
                k = np.ones((3, 3), np.uint8)
                seg_img = cv2.dilate(seg_img, k, iterations=EDGE_DILATE)
            n_canny += 1

        else:
            # 新方案：只取 sky/non-sky 主交界线（可 RANSAC 捋直）
            use_interface = True
            gate_info = {"reason": "gating_disabled"}
            if ENABLE_SEG_GATING:
                ok, gate_info = assess_interface_mask_quality(
                    mask_pp,
                    smooth_k=INTERFACE_SMOOTH_K,
                    sky_ratio_min=GATE_SKY_RATIO_MIN,
                    sky_ratio_max=GATE_SKY_RATIO_MAX,
                    col_sky_min_ratio=GATE_COL_SKY_MIN_RATIO,
                    boundary_rmse_max=GATE_BOUNDARY_RMSE_MAX,
                    leak_margin_px=GATE_LEAK_MARGIN_PX,
                    leak_ratio_max=GATE_LEAK_RATIO_MAX,
                    leak_sample_cols=GATE_LEAK_SAMPLE_COLS,
                )
                if not ok:
                    use_interface = False


            # 记录 gate 指标（无论最终是否 fallback）
            if gate_log_fp is not None:
                gate_ok_int = 1 if bool(ok) else 0
                reason = gate_info.get("reason", "")
                sky_ratio = gate_info.get("sky_ratio", None)
                col_ratio = gate_info.get("col_sky_ratio", None)
                rmse = gate_info.get("boundary_rmse", None)
                leak = gate_info.get("leak_ratio", None)
                fb_mode = "" if bool(use_interface) else str(SEG_FALLBACK_MODE)
                gate_log_fp.write(
                    f"{idx}\t{img_name}\t{gate_ok_int}\t{int(bool(use_interface))}\t{reason}\t"
                    f"{_fmt(sky_ratio)}\t{_fmt(col_ratio)}\t{_fmt(rmse, nd=4)}\t{_fmt(leak)}\t{fb_mode}\n"
                )
            if use_interface:
                seg_img = build_interface_line_image(
                    mask_pp,
                    thickness=INTERFACE_THICKNESS,
                    smooth_k=INTERFACE_SMOOTH_K,
                    use_ransac=INTERFACE_USE_RANSAC,
                    ransac_iters=RANSAC_ITERS,
                    ransac_thresh=RANSAC_THRESH_PX,
                    ransac_min_inlier_ratio=RANSAC_MIN_INLIER_RATIO,
                )
                if EDGE_DILATE > 0:
                    k = np.ones((3, 3), np.uint8)
                    seg_img = cv2.dilate(seg_img, k, iterations=EDGE_DILATE)
                n_interface += 1
            else:
                # 门控失败：回退
                fb = str(SEG_FALLBACK_MODE).lower().strip()
                if fb == "zeros":
                    seg_img = np.zeros((UNET_IN_H, UNET_IN_W), dtype=np.uint8)
                else:
                    seg_img = cv2.Canny((mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
                    if EDGE_DILATE > 0:
                        k = np.ones((3, 3), np.uint8)
                        seg_img = cv2.dilate(seg_img, k, iterations=EDGE_DILATE)
                n_fallback += 1

        seg_sino = detector._radon_gpu(seg_img, theta_scan)
        processed_stack.append(process_sinogram(seg_sino, RESIZE_H, RESIZE_W))

        combined_input = np.stack(processed_stack, axis=0).astype(np.float32)

        # 5. [核心修正] 计算 Label 时，必须将坐标缩放到 1024x576
        scale_x = UNET_IN_W / w_orig
        scale_y = UNET_IN_H / h_orig
        
        x1_s, y1_s = x1_org * scale_x, y1_org * scale_y
        x2_s, y2_s = x2_org * scale_x, y2_org * scale_y
        
        # 传入 UNet 的尺寸，而不是原图尺寸
        l_rho, l_theta = calculate_radon_label(x1_s, y1_s, x2_s, y2_s, UNET_IN_W, UNET_IN_H, RESIZE_H, RESIZE_W)
        label = np.array([l_rho, l_theta], dtype=np.float32)

        np.save(os.path.join(out_dir, f"{idx}.npy"), {"input": combined_input, "label": label})

    
    if gate_log_fp is not None:
        gate_log_fp.close()
        print(f"[GateLog] saved -> {gate_log_path}")

# split 完成后打印统计
    if str(SEG_EDGE_MODE).lower().strip() == "interface":
        total = max(1, (n_interface + n_fallback))
        fb_rate = 100.0 * n_fallback / float(total)
        print(f"[Seg-Interface] ok={n_interface} fallback={n_fallback} (fallback_rate={fb_rate:.2f}%)")

def main():
    ensure_dir(SAVE_ROOT)
    splits = _load_split_indices(SPLIT_DIR)
    
    print(f"[Load] Model: {RGHNET_CKPT}")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(RGHNET_CKPT, map_location=DEVICE), strict=False)
    model.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)
    df = pd.read_csv(CSV_PATH, header=None)

    for split in ["train", "val", "test"]:
        build_cache_for_split(df, splits[split], os.path.join(SAVE_ROOT, split), model, detector, theta_scan)

if __name__ == "__main__":
    main()