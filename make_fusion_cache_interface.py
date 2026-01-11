# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.amp as amp
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
    print(f"Processing {os.path.basename(out_dir)}: {len(indices)}")
    
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
        if str(SEG_EDGE_MODE).lower() == "canny":
            seg_img = cv2.Canny((mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
            if EDGE_DILATE > 0:
                k = np.ones((3, 3), np.uint8)
                seg_img = cv2.dilate(seg_img, k, iterations=EDGE_DILATE)
        else:
            seg_img = build_interface_line_image(
                mask_pp,
                thickness=INTERFACE_THICKNESS,
                smooth_k=INTERFACE_SMOOTH_K,
                use_ransac=INTERFACE_USE_RANSAC,
                ransac_iters=RANSAC_ITERS,
                ransac_thresh=RANSAC_THRESH_PX,
                ransac_min_inlier_ratio=RANSAC_MIN_INLIER_RATIO,
            )
            # 可选：再膨胀一点点，增强 Radon 峰值
            if EDGE_DILATE > 0:
                k = np.ones((3, 3), np.uint8)
                seg_img = cv2.dilate(seg_img, k, iterations=EDGE_DILATE)

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