# -*- coding: utf-8 -*-
"""
evaluate_full_pipeline.py

Full-pipeline evaluation on MU-SID test split:
    FusionCache(.npy) -> Fusion-CNN -> (optional) UNet seg-mask -> boundary RANSAC refine -> metrics + outliers

✅ PyCharm friendly: edit configs below, then run.
✅ Outputs:
    - eval_full_outputs/full_eval_test.csv
    - eval_full_outputs/outliers/*.png
    - eval_full_outputs/outliers.txt
    - eval_full_outputs/gate_stats.csv   (for tuning GATE_* thresholds)

Notes
-----
1) This script assumes your FusionCache npy stores:
      - x: (4, 2240, 180) float32
      - y: (2,) float32, [rho_norm, theta_norm] with theta_norm in [0,1] => theta_deg=theta_norm*180
      - img_name: original image filename (e.g., "DSC_1234_5.jpg")
   If img_name is missing, we fallback to "<npy_stem>.jpg".

2) UNet (RestorationGuidedHorizonNet) returns (restored_img, seg_logits, target_dce).
   We use seg_logits to form a binary mask:
      sea=0, sky=1

3) RANSAC boundary refinement is only applied when:
      cnn_conf < CONF_REFINE_THRESH
   and boundary fit passes gates:
      leak_ratio <= GATE_LEAK_RATIO_MAX
      rmse_px    <= GATE_BOUNDARY_RMSE_MAX
      inlier_ratio >= RANSAC_MIN_INLIER_RATIO

4) Metrics:
      - rho abs error (UNet space px)
      - theta error (deg, wrap-aware, period=180)
      - mean point->line distance (UNet space px)
      - edgeY error (UNet space px)  : mean(|y_left_diff|, |y_right_diff|)
   Also reports original-scale approx (1920x1080) rho/lineDist/edgeY by scaling factor.
"""

import os
import math
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

try:
    import cv2
except Exception as e:
    raise RuntimeError("This script needs opencv-python (cv2). Please install it in your env.") from e


# =========================
# ======= CONFIG ==========
# =========================

# ---- Paths ----
CNN_WEIGHTS_PATH  = r"splits_musid/best_fusion_cnn_1024x576.pth"

# Set this to your trained UNet weights. Example from train_unet.py:
#   rghnet_best_a.pth / rghnet_best_b.pth / rghnet_best_c.pth
UNET_WEIGHTS_PATH = r"rghnet_best_c2.pth"

# FusionCache folder (test split)
CACHE_DIR          = r"Hashmani's Dataset/FusionCache_1024x576\test"

# Original images root (must contain img_name files from cache, used for UNet + visualization)
ORIGINAL_IMG_ROOT  = r"Hashmani's Dataset\MU-SID"

# Outputs
OUT_DIR            = r"eval_full_outputs"


# ---- Runtime ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0          # PyCharm/Windows safe
BATCH_SIZE = 1           # full pipeline uses per-sample UNet + draw


# ---- Geometry (must match your cache / training) ----
@dataclass
class DenormConfig:
    # UNet/resized image space used by the horizon labels (same as your restore+seg input size)
    unet_w: int = 1024
    unet_h: int = 576

    # rho axis length used when building sinograms (your cache uses 2240 x 180)
    resize_h: int = 2240

    angle_range_deg: float = 180.0

    # for original-scale "approx" reporting
    orig_w: int = 1920
    orig_h: int = 1080

CFG = DenormConfig()


# ---- RANSAC / gating ----
CONF_REFINE_THRESH = 0.35          # refine only when CNN confidence is low
RANSAC_ITERS = 250
RANSAC_RESIDUAL_PX = 3.0           # inlier threshold on vertical residual (px) in UNet space
RANSAC_MIN_INLIER_RATIO = 0.35     # minimal inlier ratio to accept a fit

GATE_BOUNDARY_RMSE_MAX = 5.0       # will log rmse to help you tune
GATE_LEAK_RATIO_MAX = 0.30         # fraction of columns without a valid boundary point (0~1)

# boundary extraction robustness
BOUNDARY_MIN_POINTS = 200          # too few points => skip RANSAC
BOUNDARY_SMOOTH_MEDIAN_K = 7       # odd number; applied to y(x) before RANSAC


# ---- Outlier saving ----
SAVE_OUTLIERS = True
TOPK_OUTLIERS = 50
THRESH_RHO_ORIG_PX = 20.0
THRESH_EDGEY_ORIG_PX = 20.0
THRESH_THETA_DEG = 2.0

DRAW_THICKNESS = 2
FONT_SCALE = 0.6


# ---- Reproducibility ----
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# ===== Math helpers ======
# =========================

def wrap_angle_deg(err_deg: float, period: float = 180.0) -> float:
    """Smallest absolute error on periodic domain."""
    e = abs(err_deg) % period
    return min(e, period - e)

def angular_diff_deg(a: np.ndarray, b: np.ndarray, period: float = 180.0) -> np.ndarray:
    """Vectorized wrap-aware absolute difference."""
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)

def denorm_rho_theta(rho_norm: np.ndarray, theta_norm: np.ndarray, cfg: DenormConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized labels/preds -> (rho_real_px, theta_deg) in UNet space,
    using the same diagonal-padding mapping used when generating radon sinograms.
    """
    w, h = cfg.unet_w, cfg.unet_h
    diag = math.sqrt(w * w + h * h)
    pad_top = (cfg.resize_h - diag) / 2.0

    final_rho_idx = rho_norm * (cfg.resize_h - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)
    theta_deg = (theta_norm * cfg.angle_range_deg) % cfg.angle_range_deg
    return rho_real, theta_deg

def norm_rho_theta(rho_real: float, theta_deg: float, cfg: DenormConfig) -> Tuple[float, float]:
    """Inverse of denorm_rho_theta: (rho_real_px, theta_deg) -> normalized in [0,1]."""
    w, h = cfg.unet_w, cfg.unet_h
    diag = math.sqrt(w * w + h * h)
    pad_top = (cfg.resize_h - diag) / 2.0

    final_rho_idx = rho_real + pad_top + (diag / 2.0)
    rho_norm = float(final_rho_idx / (cfg.resize_h - 1.0))
    theta_norm = float((theta_deg % cfg.angle_range_deg) / cfg.angle_range_deg)
    return rho_norm, theta_norm

def summarize(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return f"{name}: (empty)"
    return (
        f"{name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}, max={arr.max():.4f}"
    )


# =========================
# ===== Line geometry =====
# =========================

def line_intersections_in_image(rho: float, theta_deg: float, w: int, h: int) -> List[Tuple[float, float]]:
    """
    Intersections between the infinite line and the image border rectangle.
    Line is defined in centered coords:
        (x-cx)*cos + (y-cy)*sin = rho
    Returns a list of (x, y) in top-left coords.
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    pts = []

    # x = 0
    x = 0.0
    if abs(sin_t) > 1e-8:
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if 0.0 <= y <= (h - 1.0):
            pts.append((x, y))

    # x = w-1
    x = float(w - 1)
    if abs(sin_t) > 1e-8:
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if 0.0 <= y <= (h - 1.0):
            pts.append((x, y))

    # y = 0
    y = 0.0
    if abs(cos_t) > 1e-8:
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if 0.0 <= x <= (w - 1.0):
            pts.append((x, y))

    # y = h-1
    y = float(h - 1)
    if abs(cos_t) > 1e-8:
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if 0.0 <= x <= (w - 1.0):
            pts.append((x, y))

    return pts

def farthest_pair(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    best = (pts[0], pts[1])
    best_d2 = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                best = (pts[i], pts[j])
    return best

def mean_point_to_line_distance(
    rho_pred: float, theta_pred_deg: float,
    rho_gt: float, theta_gt_deg: float,
    w: int, h: int, n_samples: int = 50
) -> float:
    """
    Sample points on GT line segment inside image and compute mean perpendicular distance to predicted line.
    All in UNet space (w x h).
    """
    pts = line_intersections_in_image(rho_gt, theta_gt_deg, w, h)
    if len(pts) >= 2:
        p0, p1 = farthest_pair(pts)
    else:
        # fallback: clamp 2 points
        theta = math.radians(theta_gt_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = w / 2.0, h / 2.0
        x0, x1 = 0.0, w - 1.0
        if abs(sin_t) < 1e-8:
            y0 = y1 = cy
        else:
            y0 = cy + (rho_gt - ((x0 - cx) * cos_t)) / sin_t
            y1 = cy + (rho_gt - ((x1 - cx) * cos_t)) / sin_t
        y0 = float(np.clip(y0, 0, h - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        p0, p1 = (x0, y0), (x1, y1)

    xs = np.linspace(p0[0], p1[0], n_samples)
    ys = np.linspace(p0[1], p1[1], n_samples)

    theta_p = math.radians(theta_pred_deg)
    cos_p, sin_p = math.cos(theta_p), math.sin(theta_p)
    cx, cy = w / 2.0, h / 2.0

    x_c = xs - cx
    y_c = ys - cy
    d = np.abs(x_c * cos_p + y_c * sin_p - rho_pred)
    return float(np.mean(d))

def edge_y_at_x(rho: float, theta_deg: float, x: float, w: int, h: int) -> float:
    """Compute y on the line at a given x (top-left coords), clamped to image bounds."""
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    if abs(sin_t) < 1e-8:
        # near-horizontal normal => line ~ vertical; y is ill-defined. Return mid.
        return float(cy)
    y = cy + (rho - ((x - cx) * cos_t)) / sin_t
    return float(np.clip(y, 0.0, h - 1.0))

def edge_y_error(
    rho_pred: float, theta_pred_deg: float,
    rho_gt: float, theta_gt_deg: float,
    w: int, h: int
) -> float:
    """Mean abs y difference at left and right edges."""
    ypl = edge_y_at_x(rho_pred, theta_pred_deg, 0.0, w, h)
    ypr = edge_y_at_x(rho_pred, theta_pred_deg, w - 1.0, w, h)
    ygl = edge_y_at_x(rho_gt, theta_gt_deg, 0.0, w, h)
    ygr = edge_y_at_x(rho_gt, theta_gt_deg, w - 1.0, w, h)
    return float(0.5 * (abs(ypl - ygl) + abs(ypr - ygr)))


# =========================
# ===== Data loading ======
# =========================

class FusionCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.files = sorted([str(p) for p in Path(cache_dir).glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def _load_one(self, path: str):
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()

        x = y = None
        img_name = None

        if isinstance(obj, dict):
            x = obj.get("x", obj.get("input", None))
            y = obj.get("y", obj.get("label", None))
            img_name = obj.get("img_name", obj.get("image_name", obj.get("img", None)))
        else:
            # pattern: saved as tuple/list
            if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                x, y = obj[0], obj[1]
                if len(obj) >= 3:
                    img_name = obj[2]
            elif isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) == 2 and isinstance(obj[0], np.ndarray):
                x, y = obj[0], obj[1]

        if x is None or y is None:
            raise RuntimeError(f"Cannot parse cache npy: {path}")

        if img_name is None:
            img_name = Path(path).stem + ".jpg"

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if y.size != 2:
            raise RuntimeError(f"Label must be size=2 [rho_norm, theta_norm], got shape={y.shape} in {path}")

        return x, y, str(img_name)

    def __getitem__(self, idx):
        p = self.files[idx]
        x, y, img_name = self._load_one(p)
        return torch.from_numpy(x), torch.from_numpy(y), img_name, os.path.basename(p)


def load_original_image(img_name: str, w: int, h: int) -> Optional[np.ndarray]:
    if not ORIGINAL_IMG_ROOT:
        return None
    cand = os.path.join(ORIGINAL_IMG_ROOT, img_name)
    if not os.path.exists(cand):
        return None
    im = cv2.imread(cand, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    return im


# =========================
# ===== Model loading =====
# =========================

def load_cnn():
    from cnn_model import HorizonResNet
    model = HorizonResNet(in_channels=4)
    ckpt = torch.load(CNN_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=True)
    model.eval().to(DEVICE)
    return model

def load_unet():
    if not UNET_WEIGHTS_PATH or not os.path.exists(UNET_WEIGHTS_PATH):
        print(f"[WARN] UNet weights not found: {UNET_WEIGHTS_PATH}. Will evaluate CNN only.")
        return None
    from unet_model import RestorationGuidedHorizonNet
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path="Epoch99.pth", require_dce=True)
    state = torch.load(UNET_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model


# =========================
# == Boundary extraction ==
# =========================

def extract_boundary_points_sky_to_sea(mask01: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract sky->sea boundary as one y per column:
        find smallest y where mask[y-1]==1 and mask[y]==0 (sky above, sea below)
    Returns:
        pts: (N,2) float32 with (x, y)
        leak_ratio: 1 - N / W
    """
    assert mask01.ndim == 2
    h, w = mask01.shape
    ys = np.full((w,), np.nan, dtype=np.float32)

    for x in range(w):
        col = mask01[:, x]
        # find transitions (y where sky->sea)
        # y from 1..h-1, check col[y-1]==1 and col[y]==0
        idx = np.where((col[1:] == 0) & (col[:-1] == 1))[0]
        if idx.size > 0:
            ys[x] = float(idx[0] + 1)

    valid = np.isfinite(ys)
    leak_ratio = 1.0 - float(np.mean(valid))

    if valid.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), leak_ratio

    xs = np.arange(w, dtype=np.float32)[valid]
    ys_v = ys[valid]

    # median smooth on y(x) to suppress spikes (ships/masts)
    k = int(BOUNDARY_SMOOTH_MEDIAN_K)
    if k >= 3 and (k % 2 == 1) and ys_v.size >= k:
        # build dense y array for smoothing, then re-sample valid positions
        y_dense = np.full((w,), np.nan, dtype=np.float32)
        y_dense[valid] = ys_v
        # fill missing with nearest to allow median filter
        # forward fill then backward fill
        for i in range(1, w):
            if not np.isfinite(y_dense[i]) and np.isfinite(y_dense[i-1]):
                y_dense[i] = y_dense[i-1]
        for i in range(w-2, -1, -1):
            if not np.isfinite(y_dense[i]) and np.isfinite(y_dense[i+1]):
                y_dense[i] = y_dense[i+1]
        # if still nan, skip
        if np.isfinite(y_dense).all():
            y_med = cv2.medianBlur(y_dense.reshape(1, -1), k).reshape(-1).astype(np.float32)
            ys_v = y_med[valid]

    pts = np.stack([xs, ys_v], axis=1).astype(np.float32)
    return pts, leak_ratio


def ransac_fit_line_yx(pts: np.ndarray, iters: int, thr: float) -> Optional[Dict]:
    """
    RANSAC fit y = m x + b on points (x,y).
    Inlier criterion: |y - (m x + b)| <= thr
    Returns dict with m,b,inlier_mask,inlier_ratio,rmse
    """
    if pts.shape[0] < BOUNDARY_MIN_POINTS:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]
    n = pts.shape[0]

    best_inliers = None
    best_cnt = -1
    best_m = 0.0
    best_b = 0.0

    # sample indices for speed
    for _ in range(iters):
        i1, i2 = np.random.randint(0, n, size=2)
        if i1 == i2:
            continue
        x1, y1 = xs[i1], ys[i1]
        x2, y2 = xs[i2], ys[i2]
        if abs(x2 - x1) < 1e-6:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        resid = np.abs(ys - (m * xs + b))
        inl = resid <= thr
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inl
            best_m, best_b = float(m), float(b)

    if best_inliers is None or best_cnt <= 0:
        return None

    inlier_ratio = best_cnt / float(n)
    if inlier_ratio < float(RANSAC_MIN_INLIER_RATIO):
        return None

    # refine with least squares on inliers
    xi = xs[best_inliers]
    yi = ys[best_inliers]
    if xi.size < 2:
        return None
    m_ls, b_ls = np.polyfit(xi, yi, deg=1)
    m_ls = float(m_ls); b_ls = float(b_ls)

    resid_all = ys - (m_ls * xs + b_ls)
    rmse = float(np.sqrt(np.mean(resid_all * resid_all)))

    return {
        "m": m_ls,
        "b": b_ls,
        "inlier_mask": best_inliers,
        "inlier_ratio": float(inlier_ratio),
        "rmse": rmse,
    }


def line_mb_to_polar(m: float, b: float, w: int, h: int) -> Tuple[float, float]:
    """
    Convert y = m x + b (top-left coords) -> (rho_real, theta_deg) in centered Hough form:
        (x-cx)*cos(theta) + (y-cy)*sin(theta) = rho
    """
    cx, cy = w / 2.0, h / 2.0

    # line in Ax + By + C = 0 (top-left)
    A = m
    B = -1.0
    C = b

    norm = math.sqrt(A * A + B * B) + 1e-12
    nx = A / norm
    ny = B / norm
    Cn = C / norm

    # centered rho
    rho = -Cn - (nx * cx + ny * cy)

    theta = math.degrees(math.atan2(ny, nx)) % 180.0
    return float(rho), float(theta)


# =========================
# ===== Visualization =====
# =========================

def polar_to_line_pts(theta_deg: float, rho: float, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    pts = line_intersections_in_image(rho, theta_deg, w, h)
    if len(pts) >= 2:
        p0, p1 = farthest_pair(pts)
    elif len(pts) == 1:
        p0 = pts[0]; p1 = pts[0]
    else:
        # fallback horizontal mid line
        p0 = (0.0, h / 2.0); p1 = (w - 1.0, h / 2.0)
    return (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))

def draw_overlay(im_bgr: np.ndarray,
                 rho_gt: float, theta_gt: float,
                 rho_cnn: float, theta_cnn: float,
                 rho_ref: Optional[float], theta_ref: Optional[float],
                 used_ref: bool,
                 text_lines: List[str]) -> np.ndarray:
    out = im_bgr.copy()

    # GT: green
    p1g, p2g = polar_to_line_pts(theta_gt, rho_gt, CFG.unet_w, CFG.unet_h)
    cv2.line(out, p1g, p2g, (0, 255, 0), DRAW_THICKNESS)

    # CNN: red
    p1c, p2c = polar_to_line_pts(theta_cnn, rho_cnn, CFG.unet_w, CFG.unet_h)
    cv2.line(out, p1c, p2c, (0, 0, 255), DRAW_THICKNESS)

    # REF: blue
    if rho_ref is not None and theta_ref is not None:
        p1r, p2r = polar_to_line_pts(theta_ref, rho_ref, CFG.unet_w, CFG.unet_h)
        cv2.line(out, p1r, p2r, (255, 0, 0), DRAW_THICKNESS)

    # header
    header = "GT=green  CNN=red  REF=blue  used=" + ("REF" if used_ref else "CNN")
    cv2.putText(out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    # text lines
    y0 = 48
    for i, t in enumerate(text_lines[:10]):
        yy = y0 + i * 18
        cv2.putText(out, t, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# =========================
# ======== Main ===========
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    outlier_dir = os.path.join(OUT_DIR, "outliers")
    if SAVE_OUTLIERS:
        os.makedirs(outlier_dir, exist_ok=True)

    ds = FusionCacheDataset(CACHE_DIR)
    if len(ds) == 0:
        raise RuntimeError(f"No .npy found in CACHE_DIR: {CACHE_DIR}")

    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    cnn = load_cnn()
    unet = load_unet()

    print("========== Full Pipeline Evaluation ==========")
    print(f"Split: test | N={len(ds)} | Device={DEVICE}")
    print(f"CNN Weights:  {CNN_WEIGHTS_PATH}")
    print(f"UNet Weights: {UNET_WEIGHTS_PATH if unet is not None else '(disabled)'}")
    print(f"Cache:        {CACHE_DIR}")
    print(f"Images:       {ORIGINAL_IMG_ROOT}")
    print(f"Out:          {OUT_DIR}")
    print("")

    # original-scale factor (approx)
    sx = CFG.orig_w / float(CFG.unet_w)
    sy = CFG.orig_h / float(CFG.unet_h)
    scale_orig = float(0.5 * (sx + sy))

    # record arrays
    rho_err_cnn = []
    theta_err_cnn = []
    line_dist_cnn = []
    edgey_cnn = []

    rho_err_final = []
    theta_err_final = []
    line_dist_final = []
    edgey_final = []

    gate_rows = []
    rows = []

    # for outlier selection
    per_sample_for_outlier = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, y, img_name, cache_file = batch
            # batch=1
            x = x.to(DEVICE, non_blocking=True).float()
            y = y.to(DEVICE, non_blocking=True).float()

            # CNN pred + confidence
            try:
                pred_norm, conf = cnn(x, return_conf=True)
                conf_val = float(conf.detach().cpu().numpy().reshape(-1)[0])
            except TypeError:
                pred_norm = cnn(x)
                conf_val = float("nan")

            pred_norm = pred_norm.detach().cpu().numpy().reshape(-1)
            gt_norm = y.detach().cpu().numpy().reshape(-1)

            rho_gt, theta_gt = denorm_rho_theta(gt_norm[0], gt_norm[1], CFG)
            rho_c, theta_c = denorm_rho_theta(pred_norm[0], pred_norm[1], CFG)

            # baseline metrics (CNN)
            rho_abs = float(abs(rho_c - rho_gt))
            th_abs = float(wrap_angle_deg(theta_c - theta_gt, period=180.0))
            ld = mean_point_to_line_distance(rho_c, theta_c, rho_gt, theta_gt, CFG.unet_w, CFG.unet_h)
            ey = edge_y_error(rho_c, theta_c, rho_gt, theta_gt, CFG.unet_w, CFG.unet_h)

            rho_err_cnn.append(rho_abs)
            theta_err_cnn.append(th_abs)
            line_dist_cnn.append(ld)
            edgey_cnn.append(ey)

            # default: final = cnn
            used_ref = False
            rho_f, theta_f = rho_c, theta_c

            # RANSAC refine if enabled
            leak_ratio = float("nan")
            b_rmse = float("nan")
            inlier_ratio = float("nan")
            rho_ref = None
            theta_ref = None
            refine_ok = False

            if (unet is not None) and (not math.isnan(conf_val)) and (conf_val < CONF_REFINE_THRESH):
                im = load_original_image(img_name[0], CFG.unet_w, CFG.unet_h)
                if im is not None:
                    # UNet forward
                    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im_t = torch.from_numpy(im_rgb).permute(2, 0, 1).float() / 255.0
                    im_t = im_t.unsqueeze(0).to(DEVICE)

                    _, seg_logits, _ = unet(im_t, enable_restoration=True, enable_segmentation=True)
                    if seg_logits is not None:
                        mask = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # 0 sea, 1 sky

                        pts, leak_ratio = extract_boundary_points_sky_to_sea(mask)
                        if pts.shape[0] >= BOUNDARY_MIN_POINTS:
                            fit = ransac_fit_line_yx(pts, RANSAC_ITERS, RANSAC_RESIDUAL_PX)
                            if fit is not None:
                                inlier_ratio = float(fit["inlier_ratio"])
                                b_rmse = float(fit["rmse"])

                                # gate
                                if (leak_ratio <= GATE_LEAK_RATIO_MAX) and (b_rmse <= GATE_BOUNDARY_RMSE_MAX) and (inlier_ratio >= RANSAC_MIN_INLIER_RATIO):
                                    rho_ref, theta_ref = line_mb_to_polar(fit["m"], fit["b"], CFG.unet_w, CFG.unet_h)
                                    refine_ok = True

            if refine_ok and (rho_ref is not None) and (theta_ref is not None):
                rho_f, theta_f = rho_ref, theta_ref
                used_ref = True

            # final metrics
            rho_abs_f = float(abs(rho_f - rho_gt))
            th_abs_f = float(wrap_angle_deg(theta_f - theta_gt, period=180.0))
            ld_f = mean_point_to_line_distance(rho_f, theta_f, rho_gt, theta_gt, CFG.unet_w, CFG.unet_h)
            ey_f = edge_y_error(rho_f, theta_f, rho_gt, theta_gt, CFG.unet_w, CFG.unet_h)

            rho_err_final.append(rho_abs_f)
            theta_err_final.append(th_abs_f)
            line_dist_final.append(ld_f)
            edgey_final.append(ey_f)

            # save row
            row = {
                "idx": int(i),
                "cache_file": str(cache_file[0]),
                "img_name": str(img_name[0]),
                "conf": conf_val,
                "used_ref": int(used_ref),
                "leak_ratio": leak_ratio,
                "boundary_rmse": b_rmse,
                "inlier_ratio": inlier_ratio,
                "rho_abs_px_unet_cnn": rho_abs,
                "theta_abs_deg_cnn": th_abs,
                "line_dist_px_unet_cnn": ld,
                "edgey_px_unet_cnn": ey,
                "rho_abs_px_unet_final": rho_abs_f,
                "theta_abs_deg_final": th_abs_f,
                "line_dist_px_unet_final": ld_f,
                "edgey_px_unet_final": ey_f,
                "rho_abs_px_orig_cnn": rho_abs * scale_orig,
                "line_dist_px_orig_cnn": ld * scale_orig,
                "edgey_px_orig_cnn": ey * scale_orig,
                "rho_abs_px_orig_final": rho_abs_f * scale_orig,
                "line_dist_px_orig_final": ld_f * scale_orig,
                "edgey_px_orig_final": ey_f * scale_orig,
                "rho_gt": float(rho_gt),
                "theta_gt": float(theta_gt),
                "rho_cnn": float(rho_c),
                "theta_cnn": float(theta_c),
                "rho_ref": float(rho_ref) if rho_ref is not None else float("nan"),
                "theta_ref": float(theta_ref) if theta_ref is not None else float("nan"),
                "rho_final": float(rho_f),
                "theta_final": float(theta_f),
            }
            rows.append(row)

            gate_rows.append({
                "idx": int(i),
                "img_name": str(img_name[0]),
                "conf": conf_val,
                "leak_ratio": leak_ratio,
                "boundary_rmse": b_rmse,
                "inlier_ratio": inlier_ratio,
                "refine_ok": int(refine_ok),
                "used_ref": int(used_ref),
            })

            per_sample_for_outlier.append({
                "idx": int(i),
                "img_name": str(img_name[0]),
                "cache_file": str(cache_file[0]),
                "theta_err_deg": th_abs_f,
                "rho_err_orig": rho_abs_f * scale_orig,
                "edgey_err_orig": ey_f * scale_orig,
                "line_dist_orig": ld_f * scale_orig,
                "conf": conf_val,
                "used_ref": int(used_ref),
                "rho_gt": float(rho_gt),
                "theta_gt": float(theta_gt),
                "rho_cnn": float(rho_c),
                "theta_cnn": float(theta_c),
                "rho_ref": float(rho_ref) if rho_ref is not None else float("nan"),
                "theta_ref": float(theta_ref) if theta_ref is not None else float("nan"),
                "rho_final": float(rho_f),
                "theta_final": float(theta_f),
            })

    # --------------------
    # Print summaries
    # --------------------
    rho_err_cnn = np.array(rho_err_cnn, dtype=np.float64)
    theta_err_cnn = np.array(theta_err_cnn, dtype=np.float64)
    line_dist_cnn = np.array(line_dist_cnn, dtype=np.float64)
    edgey_cnn = np.array(edgey_cnn, dtype=np.float64)

    rho_err_final = np.array(rho_err_final, dtype=np.float64)
    theta_err_final = np.array(theta_err_final, dtype=np.float64)
    line_dist_final = np.array(line_dist_final, dtype=np.float64)
    edgey_final = np.array(edgey_final, dtype=np.float64)

    print("---- CNN only ----")
    print(summarize("Rho abs error (px, UNet)", rho_err_cnn))
    print(summarize("Theta error (deg)", theta_err_cnn))
    print(summarize("Mean point->line dist (px, UNet)", line_dist_cnn))
    print(summarize("Edge-Y error (px, UNet)", edgey_cnn))
    print(summarize("Rho abs error (px, original-scale approx)", rho_err_cnn * scale_orig))
    print(summarize("Line dist (px, original-scale approx)", line_dist_cnn * scale_orig))
    print(summarize("EdgeY (px, original-scale approx)", edgey_cnn * scale_orig))
    print("")

    print("---- Final (conf-gated RANSAC refine) ----")
    print(summarize("Rho abs error (px, UNet)", rho_err_final))
    print(summarize("Theta error (deg)", theta_err_final))
    print(summarize("Mean point->line dist (px, UNet)", line_dist_final))
    print(summarize("Edge-Y error (px, UNet)", edgey_final))
    print(summarize("Rho abs error (px, original-scale approx)", rho_err_final * scale_orig))
    print(summarize("Line dist (px, original-scale approx)", line_dist_final * scale_orig))
    print(summarize("EdgeY (px, original-scale approx)", edgey_final * scale_orig))
    print("")

    # threshold stats (final)
    def pct_le(arr, t): return 100.0 * float(np.mean(arr <= t))
    print("---- Threshold stats (FINAL) ----")
    print(f"Theta <=1°: {pct_le(theta_err_final, 1.0):.2f}% | <=2°: {pct_le(theta_err_final, 2.0):.2f}% | <=5°: {pct_le(theta_err_final, 5.0):.2f}%")
    print(f"Rho(orig) <=5px: {pct_le(rho_err_final * scale_orig, 5.0):.2f}% | <=10px: {pct_le(rho_err_final * scale_orig, 10.0):.2f}% | <=20px: {pct_le(rho_err_final * scale_orig, 20.0):.2f}%")
    print(f"LineDist(orig) <=5px: {pct_le(line_dist_final * scale_orig, 5.0):.2f}% | <=10px: {pct_le(line_dist_final * scale_orig, 10.0):.2f}% | <=20px: {pct_le(line_dist_final * scale_orig, 20.0):.2f}%")
    print(f"EdgeY(orig) <=5px: {pct_le(edgey_final * scale_orig, 5.0):.2f}% | <=10px: {pct_le(edgey_final * scale_orig, 10.0):.2f}% | <=20px: {pct_le(edgey_final * scale_orig, 20.0):.2f}%")
    print("")

    # --------------------
    # Save CSVs
    # --------------------
    csv_path = os.path.join(OUT_DIR, "full_eval_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Saved] per-sample full metrics -> {csv_path}")

    gate_path = os.path.join(OUT_DIR, "gate_stats.csv")
    with open(gate_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        w.writeheader()
        for r in gate_rows:
            w.writerow(r)
    print(f"[Saved] gate stats -> {gate_path} (use this to tune GATE_* thresholds)")

    # --------------------
    # Outliers
    # --------------------
    if SAVE_OUTLIERS:
        # select by threshold OR topK on line_dist
        flags = []
        for s in per_sample_for_outlier:
            is_bad = (s["theta_err_deg"] > THRESH_THETA_DEG) or (s["rho_err_orig"] > THRESH_RHO_ORIG_PX) or (s["edgey_err_orig"] > THRESH_EDGEY_ORIG_PX)
            flags.append(bool(is_bad))

        # topK by line_dist
        order = sorted(per_sample_for_outlier, key=lambda d: float(d["line_dist_orig"]), reverse=True)
        topk_set = set([d["idx"] for d in order[:TOPK_OUTLIERS]])

        outliers = []
        for s, flg in zip(per_sample_for_outlier, flags):
            if flg or (s["idx"] in topk_set):
                outliers.append(s)

        # deterministic order by line_dist desc
        outliers = sorted(outliers, key=lambda d: float(d["line_dist_orig"]), reverse=True)

        txt_path = os.path.join(OUT_DIR, "outliers.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for s in outliers:
                f.write(
                    f'{s["idx"]}\t{s["img_name"]}\t'
                    f'theta={s["theta_err_deg"]:.3f}\t'
                    f'rho~{s["rho_err_orig"]:.2f}\t'
                    f'lineDist~{s["line_dist_orig"]:.2f}\t'
                    f'edgeY~{s["edgey_err_orig"]:.2f}\t'
                    f'conf={s["conf"]:.3f}\tused_ref={s["used_ref"]}\n'
                )
        print(f"[Saved] outlier list -> {txt_path}")
        print(f"[Outliers] selected={len(outliers)} | saving overlay images to: {outlier_dir}")

        # draw overlays
        for s in outliers:
            im = load_original_image(s["img_name"], CFG.unet_w, CFG.unet_h)
            if im is None:
                im = np.zeros((CFG.unet_h, CFG.unet_w, 3), dtype=np.uint8)

            text_lines = [
                f'conf={s["conf"]:.3f} used_ref={s["used_ref"]}',
                f'theta_err={s["theta_err_deg"]:.3f} deg',
                f'rho_err(orig)={s["rho_err_orig"]:.2f} px',
                f'lineDist(orig)={s["line_dist_orig"]:.2f} px',
                f'edgeY(orig)={s["edgey_err_orig"]:.2f} px',
            ]

            rho_gt = s["rho_gt"]; theta_gt = s["theta_gt"]
            rho_cnn = s["rho_cnn"]; theta_cnn = s["theta_cnn"]
            rho_ref = s["rho_ref"]; theta_ref = s["theta_ref"]
            used_ref = bool(s["used_ref"])

            # nan check
            if not np.isfinite(rho_ref) or not np.isfinite(theta_ref):
                rho_ref_v = None; theta_ref_v = None
            else:
                rho_ref_v = float(rho_ref); theta_ref_v = float(theta_ref)

            vis = draw_overlay(im, rho_gt, theta_gt, rho_cnn, theta_cnn, rho_ref_v, theta_ref_v, used_ref, text_lines)

            out_name = Path(s["img_name"]).stem + ".png"
            out_path = os.path.join(outlier_dir, out_name)
            cv2.imwrite(out_path, vis)

        print("[Saved] outlier overlays done.")

    print("Done.")


if __name__ == "__main__":
    main()
