import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

from unet_model import RestorationGuidedHorizonNet
from cnn_model import HorizonResNet
from gradient_radon import TextureSuppressedMuSCoWERT
from dataset_loader import synthesize_rain_fog

# ================= 配置 =================
IMG_DIR = r"Hashmani's Dataset/clear"
OUT_DIR = r"demo_results"

UNET_CKPT = "rghnet_best_c2.pth"       
CNN_CKPT = "splits_musid/best_fusion_cnn_1024x576_interface.pth" 
DCE_WEIGHTS = "Epoch99.pth"

ENABLE_DEGRADATION = True
UNET_W, UNET_H = 1024, 576
RESIZE_H, RESIZE_W = 2240, 180

# seg-mask -> Radon 方式（需与训练 cache 一致）
SEG_EDGE_MODE = "interface"      # "interface" / "canny"
INTERFACE_THICKNESS = 2
INTERFACE_SMOOTH_K = 15
INTERFACE_USE_RANSAC = True
RANSAC_ITERS = 200
RANSAC_THRESH_PX = 3.0
RANSAC_MIN_INLIER_RATIO = 0.35
NUM_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
# =======================================

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

def process_sinogram(sino, th, tw):
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6: sino_norm = (sino - mi) / (ma - mi)
    else: sino_norm = np.zeros_like(sino, dtype=np.float32)
    h = sino_norm.shape[0]
    c = np.zeros((th, tw), dtype=np.float32)
    st = (th - h) // 2
    if h <= th: c[st:st+h, :] = sino_norm
    else: 
        cs = (h - th) // 2
        c[:, :] = sino_norm[cs:cs+th, :]
    return c

def post_process_mask(mask_np):
    # 简单的连通域处理，保持一致性
    valid = (mask_np != 255)
    sky = ((mask_np == 1) & valid).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_TOP] <= 0: keep[labels == i] = 1
    out = mask_np.copy()
    out[(mask_np == 1) & (keep == 0)] = 0
    return out

def get_line_ends(rho_norm, theta_norm, w, h):
    # 1. 还原到 resize_h 尺度
    diag = np.sqrt(w**2 + h**2)
    pad_top = (RESIZE_H - diag) / 2.0
    rho_real = rho_norm * (RESIZE_H - 1) - pad_top - (diag / 2.0)
    
    theta_rad = np.deg2rad(theta_norm * 180.0)
    
    # 2. 计算端点
    cx, cy = w / 2.0, h / 2.0
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x0 = cos_t * rho_real
    y0 = sin_t * rho_real
    
    scale = max(w, h) * 2
    pt1 = (int(cx + x0 - scale * sin_t), int(cy + y0 + scale * cos_t))
    pt2 = (int(cx + x0 + scale * sin_t), int(cy + y0 - scale * cos_t))
    return pt1, pt2

def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    random.seed(SEED); torch.manual_seed(SEED)
    
    print("Loading models...")
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE), strict=False)
    unet.eval()

    cnn = HorizonResNet(in_channels=4).to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location=DEVICE))
    cnn.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
    samples = random.sample(imgs, min(NUM_SAMPLES, len(imgs)))

    print(f"Testing {len(samples)} images...")

    for i, path in enumerate(samples):
        bgr = cv2.imread(path)
        if bgr is None: continue
        h_orig, w_orig = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Resize to UNet size (1024x576)
        rgb_unet = cv2.resize(rgb, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_LINEAR)
        
        if ENABLE_DEGRADATION:
            rgb_in_np = synthesize_rain_fog(rgb_unet, p_clean=0.0)
        else:
            rgb_in_np = rgb_unet.astype(np.float32) / 255.0

        t_in = torch.from_numpy(rgb_in_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2. UNet
        with torch.no_grad():
            res_t, seg_t, _ = unet(t_in, None, True, True)
        
        res_np = (res_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask_np = seg_t.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 3. Post process
        res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)
        mask_pp = post_process_mask(mask_np)

        # 4. Feature Extraction (on 1024x576)
        try: _, _, _, trads = detector.detect(res_bgr)
        except: trads = []
        
        stack = []
        for s in trads[:3]: stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(stack) < 3: stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))
        
        # --- seg-mask -> Radon (第4通道) ---
        if str(SEG_EDGE_MODE).lower() == "canny":
            seg_img = cv2.Canny((mask_pp * 255).astype(np.uint8), 50, 150)
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
        seg_sino = detector._radon_gpu(seg_img, theta_scan)
        stack.append(process_sinogram(seg_sino, RESIZE_H, RESIZE_W))

        cnn_in = torch.from_numpy(np.stack(stack)).float().unsqueeze(0).to(DEVICE)

        # 5. CNN Predict
        with torch.no_grad():
            preds = cnn(cnn_in).cpu().numpy()[0]
        
        # 6. Visualize
        # 注意：preds 是基于 1024x576 的，我们需要把它画在原图 1920x1080 上
        # 所以先在 1024x576 上算出端点，再按比例缩放端点坐标
        
        pt1_s, pt2_s = get_line_ends(preds[0], preds[1], UNET_IN_W, UNET_IN_H)
        
        scale_x = w_orig / UNET_IN_W
        scale_y = h_orig / UNET_IN_H
        
        pt1_orig = (int(pt1_s[0] * scale_x), int(pt1_s[1] * scale_y))
        pt2_orig = (int(pt2_s[0] * scale_x), int(pt2_s[1] * scale_y))

        vis_final = bgr.copy()
        cv2.line(vis_final, pt1_orig, pt2_orig, (0, 0, 255), 3)

        # Save
        vis_in_show = cv2.resize(cv2.cvtColor(rgb_in_np, cv2.COLOR_RGB2BGR), (w_orig, h_orig))
        res_show = cv2.resize(res_bgr, (w_orig, h_orig))
        vis_in_show = (vis_in_show * 255).astype(np.uint8)

        comb = np.hstack([vis_in_show, res_show, vis_final])
        
        # Resize for display convenience (too wide)
        h_disp = 400
        r = h_disp / h_orig
        comb_disp = cv2.resize(comb, (int(comb.shape[1]*r), h_disp))
        
        prefix = "degraded" if ENABLE_DEGRADATION else "clean"
        cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_{i}.png"), comb_disp)
        print(f"Saved {i}")

    print("Done")

if __name__ == "__main__":
    main()