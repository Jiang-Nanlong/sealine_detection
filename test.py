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
CNN_CKPT = "splits_musid/best_fusion_cnn_1024x576.pth"
DCE_WEIGHTS = "Epoch99.pth"

ENABLE_DEGRADATION = True
UNET_W, UNET_H = 1024, 576
UNET_IN_W, UNET_IN_H = UNET_W, UNET_H  # backward-compatible alias
RESIZE_H, RESIZE_W = 2240, 180

NUM_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 39

# ---- Postprocess configs (RANSAC refine using sky/non-sky boundary points) ----
CONF_RANSAC_TRIGGER = 0.35  # only run RANSAC when CNN confidence is low
LEAK_RATIO_MAX = 0.08  # max allowed non-sky pixels in top band (mask quality gate)
TOP_BAND_RATIO = 0.15  # top band height ratio for leak check
BOUNDARY_STEP = 2  # sample boundary points every N columns
MIN_BOUNDARY_PTS = 80
RANSAC_ITERS = 300
RANSAC_DIST_THRESH = 3.0  # in UNet-scale pixels
RANSAC_MIN_INLIER_RATIO = 0.25


# -----------------------------------------------------------------------------
# =======================================

def process_sinogram(sino, th, tw):
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6:
        sino_norm = (sino - mi) / (ma - mi)
    else:
        sino_norm = np.zeros_like(sino, dtype=np.float32)
    h = sino_norm.shape[0]
    c = np.zeros((th, tw), dtype=np.float32)
    st = (th - h) // 2
    if h <= th:
        c[st:st + h, :] = sino_norm
    else:
        cs = (h - th) // 2
        c[:, :] = sino_norm[cs:cs + th, :]
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
    diag = np.sqrt(w ** 2 + h ** 2)
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


def _line_from_2pts(p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    n = np.hypot(a, b) + 1e-12
    return a / n, b / n, c / n


def _point_line_dist(pts, a, b, c):
    # pts: [N,2]
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)


def _fit_line_tls(pts):
    # Total least squares fit: returns normalized (a,b,c) for ax+by+c=0
    # pts: [N,2]
    mu = pts.mean(axis=0)
    X = pts - mu[None, :]
    # smallest eigenvector of covariance is the normal
    cov = X.T @ X
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]  # normal
    a, b = float(n[0]), float(n[1])
    c = -(a * float(mu[0]) + b * float(mu[1]))
    nn = np.hypot(a, b) + 1e-12
    return a / nn, b / nn, c / nn


def _line_to_endpoints(a, b, c, w, h):
    # Intersect with image rectangle; return two points for drawing
    pts = []
    # x = 0, x = w-1
    for x in (0.0, float(w - 1)):
        if abs(b) > 1e-8:
            y = -(a * x + c) / b
            if 0.0 <= y <= float(h - 1):
                pts.append((int(round(x)), int(round(y))))
    # y = 0, y = h-1
    for y in (0.0, float(h - 1)):
        if abs(a) > 1e-8:
            x = -(b * y + c) / a
            if 0.0 <= x <= float(w - 1):
                pts.append((int(round(x)), int(round(y))))
    if len(pts) >= 2:
        # de-dup
        uniq = []
        for p in pts:
            if p not in uniq:
                uniq.append(p)
        if len(uniq) >= 2:
            return uniq[0], uniq[1]
        return pts[0], pts[1]
    # fallback: a point at center + a direction vector
    cx, cy = w // 2, h // 2
    dx, dy = -b, a
    p1 = (int(cx - 2 * w * dx), int(cy - 2 * w * dy))
    p2 = (int(cx + 2 * w * dx), int(cy + 2 * w * dy))
    return p1, p2


def compute_leak_ratio(mask_pp, top_band_ratio=TOP_BAND_RATIO):
    h, w = mask_pp.shape[:2]
    top_h = max(1, int(round(h * float(top_band_ratio))))
    top = mask_pp[:top_h, :]
    # non-sky(0) pixels in top band
    return float(np.mean(top == 0))


def extract_boundary_points(mask_pp, step=BOUNDARY_STEP):
    """Extract sky/non-sky boundary points from a 0/1 mask.
    Returns Nx2 float array in (x,y) coordinates (UNet scale).
    """
    h, w = mask_pp.shape[:2]
    pts = []
    for x in range(0, w, int(step)):
        col = mask_pp[:, x]
        idx = np.where(col == 0)[0]  # first non-sky from top
        if idx.size == 0:
            continue
        y0 = int(idx[0])
        if y0 <= 0:
            continue
        pts.append((float(x), float(y0 - 1)))
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def ransac_refine_line(boundary_pts, w, h, n_iters=RANSAC_ITERS, dist_thresh=RANSAC_DIST_THRESH,
                       min_inlier_ratio=RANSAC_MIN_INLIER_RATIO, seed=SEED):
    """RANSAC robust line fit for boundary points. Returns (pt1, pt2, rmse, inlier_ratio) or None."""
    if boundary_pts is None or len(boundary_pts) < MIN_BOUNDARY_PTS:
        return None

    rng = np.random.default_rng(int(seed))
    pts = np.asarray(boundary_pts, dtype=np.float32)
    n = pts.shape[0]
    best_inliers = None
    best_cnt = 0

    for _ in range(int(n_iters)):
        i1, i2 = rng.choice(n, size=2, replace=False)
        a, b, c = _line_from_2pts(pts[i1], pts[i2])
        d = _point_line_dist(pts, a, b, c)
        inliers = d < float(dist_thresh)
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers

    if best_inliers is None:
        return None

    inlier_ratio = float(best_cnt) / float(n)
    if inlier_ratio < float(min_inlier_ratio):
        return None

    in_pts = pts[best_inliers]
    a, b, c = _fit_line_tls(in_pts)
    d = _point_line_dist(in_pts, a, b, c)
    rmse = float(np.sqrt(np.mean(d * d))) if d.size > 0 else 1e9
    pt1, pt2 = _line_to_endpoints(a, b, c, w, h)
    return pt1, pt2, rmse, inlier_ratio


def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    random.seed(SEED);
    torch.manual_seed(SEED)

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
        try:
            _, _, _, trads = detector.detect(res_bgr)
        except:
            trads = []

        stack = []
        for s in trads[:3]: stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(stack) < 3: stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

        edges = cv2.Canny((mask_pp * 255).astype(np.uint8), 50, 150)
        seg_sino = detector._radon_gpu(edges, theta_scan)
        stack.append(process_sinogram(seg_sino, RESIZE_H, RESIZE_W))

        cnn_in = torch.from_numpy(np.stack(stack)).float().unsqueeze(0).to(DEVICE)

        # 5. CNN Predict
        with torch.no_grad():
            preds, conf = cnn(cnn_in, return_conf=True)
            preds = preds.cpu().numpy()[0]
            conf = float(conf.cpu().numpy()[0])  # 6. Visualize
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
        comb_disp = cv2.resize(comb, (int(comb.shape[1] * r), h_disp))

        prefix = "degraded" if ENABLE_DEGRADATION else "clean"
        cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_{i}.png"), comb_disp)
        print(f"Saved {i}")

    print("Done")


if __name__ == "__main__":
    main()