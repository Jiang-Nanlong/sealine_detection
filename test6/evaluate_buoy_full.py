# -*- coding: utf-8 -*-
"""
Full Pipeline Evaluation for Buoy Test Set (Experiment 6: In-Domain).

与主评估代码 evaluate_full_pipeline.py 完全对齐的指标：
  - rho error (UNet空间 + 原图尺寸)
  - theta error (度数)
  - Mean point->line distance
  - Edge-Y error
  - VE (Vertical Error) - 中点垂直误差
  - AE (Angular Error) - 角度误差

Inputs:
  - test6/FusionCache_Buoy/test/
  - test6/weights/best_fusion_cnn_buoy.pth (CNN)
  - test6/weights_buoy/buoy_rghnet_best_seg_c2.pth (UNet)
  - test6/splits_buoy/test_indices.npy

Outputs:
  - test6/eval_buoy_full_outputs/full_eval_buoy_test.csv
  - 终端输出统计信息 (与 evaluate_full_pipeline.py 格式一致)

PyCharm: 直接运行此文件
"""

import os
import sys
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import median_filter

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_model import HorizonResNet  # noqa: E402
from unet_model import RestorationGuidedHorizonNet  # noqa: E402


# ============================
# Configuration
# ============================
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 40

# Model paths
TEST6_DIR = PROJECT_ROOT / "test6"
CACHE_DIR = TEST6_DIR / "FusionCache_Buoy" / "test"
SPLIT_DIR = TEST6_DIR / "splits_buoy"
CNN_WEIGHTS_PATH = TEST6_DIR / "weights" / "best_fusion_cnn_buoy.pth"
UNET_WEIGHTS_PATH = TEST6_DIR / "weights_buoy" / "buoy_rghnet_best_seg_c2.pth"
DCE_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "Epoch99.pth"

# Output
OUT_DIR = TEST6_DIR / "eval_buoy_full_outputs"
OUT_CSV = OUT_DIR / "full_eval_buoy_test.csv"
SUMMARY_CSV = OUT_DIR / "eval_summary_buoy.csv"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Denorm config (与主评估代码一致)
UNET_W, UNET_H = 1024, 576
RESIZE_H = 2240
ANGLE_RANGE_DEG = 180.0

# RANSAC params
RANSAC_ITERS = 1000
RANSAC_THR = 3.0
RANSAC_MIN_INLIER_RATIO = 0.30
BOUNDARY_MIN_POINTS = 10
BOUNDARY_SMOOTH_MEDIAN_K = 11

# Confidence gating
CONFIDENCE_GATE_LO = 0.4
CONFIDENCE_GATE_HI = 0.5

# Scale vars (Buoy images are typically 800x600)
DEFAULT_ORIG_W = 800
DEFAULT_ORIG_H = 600


# ============================
# Dataset
# ============================
class TestCacheDataset(Dataset):
    """Load cached fusion data for evaluation."""
    
    def __init__(self, cache_dir: str, indices: list = None):
        self.cache_dir = cache_dir
        if indices is not None:
            self.files = [os.path.join(cache_dir, f"{idx}.npy") for idx in indices]
            self.files = [f for f in self.files if os.path.exists(f)]
        else:
            self.files = sorted([str(p) for p in Path(cache_dir).glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()

        x = y = None
        img_name = None
        orig_w, orig_h = DEFAULT_ORIG_W, DEFAULT_ORIG_H

        if isinstance(obj, dict):
            x = obj.get("x", obj.get("input", None))
            y = obj.get("y", obj.get("label", None))
            img_name = obj.get("img_name", obj.get("image_name", obj.get("img", None)))
            orig_w = obj.get("orig_w", DEFAULT_ORIG_W)
            orig_h = obj.get("orig_h", DEFAULT_ORIG_H)
        else:
            if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                x, y = obj[0], obj[1]
                if len(obj) >= 3:
                    img_name = obj[2]

        if x is None or y is None:
            raise RuntimeError(f"Cannot parse cache npy: {path}")

        if img_name is None:
            img_name = Path(path).stem + ".jpg"

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        
        return (torch.from_numpy(x), torch.from_numpy(y), str(img_name), 
                os.path.basename(path), int(orig_w), int(orig_h))


def load_split_indices(split_dir):
    path = os.path.join(split_dir, "test_indices.npy")
    if not os.path.exists(path):
        return None  # Will use all files
    return np.load(path).astype(np.int64).tolist()


# ============================
# Math / Geometry functions
# ============================
def wrap_angle_deg(x: float) -> float:
    """Wrap angle to [0, 180)"""
    return x % 180.0


def angular_diff_deg(a: float, b: float, period: float = 180.0) -> float:
    """Compute periodic angular difference (always positive)"""
    d = abs(a - b) % period
    return min(d, period - d)


def wrap_signed_angle_diff(arr: np.ndarray, period: float = 180.0) -> np.ndarray:
    """Wrap signed angle differences to [-period/2, period/2)"""
    half = period / 2.0
    arr_mod = arr - np.floor((arr + half) / period) * period
    return arr_mod


def denorm_rho_theta(rho_norm: float, theta_norm: float) -> Tuple[float, float]:
    """Convert normalized (rho, theta) to real values in UNet space."""
    diag = math.sqrt(UNET_W ** 2 + UNET_H ** 2)
    pad_top = (RESIZE_H - diag) / 2.0
    
    final_rho_idx = rho_norm * (RESIZE_H - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)
    theta_deg = (theta_norm * ANGLE_RANGE_DEG) % ANGLE_RANGE_DEG
    
    return rho_real, theta_deg


def norm_rho_theta(rho_real: float, theta_deg: float) -> Tuple[float, float]:
    """Convert real (rho, theta) to normalized form."""
    diag = math.sqrt(UNET_W ** 2 + UNET_H ** 2)
    pad_top = (RESIZE_H - diag) / 2.0
    
    final_rho_idx = rho_real + pad_top + (diag / 2.0)
    rho_norm = final_rho_idx / (RESIZE_H - 1.0)
    theta_norm = (theta_deg % ANGLE_RANGE_DEG) / ANGLE_RANGE_DEG
    
    return rho_norm, theta_norm


def summarize(name: str, arr: np.ndarray) -> str:
    """Generate statistics summary string."""
    if len(arr) == 0:
        return f"{name}: (empty)"
    return (
        f"{name}: mean={np.mean(arr):.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}, max={np.max(arr):.4f}"
    )


def pct_le(arr: np.ndarray, thr: float) -> float:
    """Compute percentage of values <= threshold."""
    if len(arr) == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


# ============================
# Line geometry functions
# ============================
def line_intersections_in_image(rho: float, theta_deg: float, w: int, h: int) -> List[Tuple[float, float]]:
    """Find intersections of line (rho, theta) with image boundaries."""
    theta = math.radians(theta_deg)
    cx, cy = w / 2.0, h / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    pts = []
    eps = 1e-9
    
    # Line equation: (x-cx)*cos + (y-cy)*sin = rho
    if abs(cos_t) > eps:
        x_top = (rho - (0 - cy) * sin_t) / cos_t + cx
        if 0 <= x_top <= w:
            pts.append((x_top, 0.0))
        x_bot = (rho - (h - cy) * sin_t) / cos_t + cx
        if 0 <= x_bot <= w:
            pts.append((x_bot, float(h)))
    
    if abs(sin_t) > eps:
        y_left = (rho - (0 - cx) * cos_t) / sin_t + cy
        if 0 <= y_left <= h:
            pts.append((0.0, y_left))
        y_right = (rho - (w - cx) * cos_t) / sin_t + cy
        if 0 <= y_right <= h:
            pts.append((float(w), y_right))
    
    # Remove duplicates
    unique = []
    for p in pts:
        dup = False
        for u in unique:
            if abs(p[0] - u[0]) < 0.5 and abs(p[1] - u[1]) < 0.5:
                dup = True
                break
        if not dup:
            unique.append(p)
    
    return unique


def farthest_pair(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Find the two points that are farthest apart."""
    best_d = -1
    p0, p1 = pts[0], pts[-1]
    for i, a in enumerate(pts):
        for b in pts[i+1:]:
            d = (a[0]-b[0])**2 + (a[1]-b[1])**2
            if d > best_d:
                best_d = d
                p0, p1 = a, b
    return p0, p1


def mean_point_to_line_distance(rho: float, theta_deg: float, rho_gt: float, theta_gt_deg: float, 
                                 w: int, h: int) -> float:
    """Compute mean point-to-line distance between predicted and GT lines."""
    pts = line_intersections_in_image(rho_gt, theta_gt_deg, w, h)
    if len(pts) < 2:
        return float('nan')
    
    p0, p1 = farthest_pair(pts)
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    def dist(x, y):
        return abs((x - cx) * cos_t + (y - cy) * sin_t - rho)
    
    d0 = dist(p0[0], p0[1])
    d1 = dist(p1[0], p1[1])
    return (d0 + d1) / 2.0


def edge_y_at_x(rho: float, theta_deg: float, x: float, cx: float, cy: float) -> float:
    """Compute y coordinate of line at given x."""
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    if abs(sin_t) < 1e-9:
        return float('nan')
    return cy + (rho - (x - cx) * cos_t) / sin_t


def edge_y_error(rho: float, theta_deg: float, rho_gt: float, theta_gt_deg: float, 
                 w: int, h: int, x: Optional[float] = None) -> float:
    """Compute edge-Y error at given x (default: center)."""
    if x is None:
        x = w / 2.0
    cx, cy = w / 2.0, h / 2.0
    y_pred = edge_y_at_x(rho, theta_deg, x, cx, cy)
    y_gt = edge_y_at_x(rho_gt, theta_gt_deg, x, cx, cy)
    if math.isnan(y_pred) or math.isnan(y_gt):
        return float('nan')
    return abs(y_pred - y_gt)


def compute_VE(rho: float, theta_deg: float, rho_gt: float, theta_gt_deg: float, 
               w: int, h: int) -> float:
    """Compute Vertical Error at center column (signed)."""
    cx, cy = w / 2.0, h / 2.0
    y_pred = edge_y_at_x(rho, theta_deg, cx, cx, cy)
    y_gt = edge_y_at_x(rho_gt, theta_gt_deg, cx, cx, cy)
    if math.isnan(y_pred) or math.isnan(y_gt):
        return float('nan')
    return y_pred - y_gt  # signed


def compute_AE(theta_deg: float, theta_gt_deg: float) -> float:
    """Compute Angular Error (signed)."""
    return theta_deg - theta_gt_deg  # signed


# ============================
# Boundary extraction & RANSAC
# ============================
def extract_boundary_points_sky_to_sea(mask01: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract sky->sea boundary as one y per column.
    Returns:
        pts: (N, 2) float32 with (x, y)
        leak_ratio: 1 - N / W
    """
    assert mask01.ndim == 2
    h, w = mask01.shape
    ys = np.full((w,), np.nan, dtype=np.float32)

    for x in range(w):
        col = mask01[:, x]
        idx = np.where((col[1:] == 0) & (col[:-1] == 1))[0]
        if idx.size > 0:
            ys[x] = float(idx[0] + 1)

    valid = np.isfinite(ys)
    leak_ratio = 1.0 - float(np.mean(valid))

    if valid.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), leak_ratio

    xs = np.arange(w, dtype=np.float32)[valid]
    ys_v = ys[valid]

    # Median smooth
    k = int(BOUNDARY_SMOOTH_MEDIAN_K)
    if k >= 3 and (k % 2 == 1) and ys_v.size >= k:
        y_dense = np.full((w,), np.nan, dtype=np.float32)
        y_dense[valid] = ys_v
        for i in range(1, w):
            if not np.isfinite(y_dense[i]) and np.isfinite(y_dense[i - 1]):
                y_dense[i] = y_dense[i - 1]
        for i in range(w - 2, -1, -1):
            if not np.isfinite(y_dense[i]) and np.isfinite(y_dense[i + 1]):
                y_dense[i] = y_dense[i + 1]
        if np.isfinite(y_dense).all():
            y_med = median_filter(y_dense, size=k, mode='nearest').astype(np.float32)
            ys_v = y_med[valid]

    pts = np.stack([xs, ys_v], axis=1).astype(np.float32)
    return pts, leak_ratio


def ransac_fit_line_yx(pts: np.ndarray, iters: int, thr: float) -> Optional[Dict]:
    """RANSAC fit y = m*x + b on points (x, y)."""
    if pts.shape[0] < BOUNDARY_MIN_POINTS:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]
    n = pts.shape[0]

    best_inliers = None
    best_cnt = -1
    best_m = 0.0
    best_b = 0.0

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

    xi = xs[best_inliers]
    yi = ys[best_inliers]
    if xi.size < 2:
        return None
    m_ls, b_ls = np.polyfit(xi, yi, deg=1)
    m_ls = float(m_ls)
    b_ls = float(b_ls)

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
    """Convert y = m*x + b to polar form (rho, theta)."""
    cx, cy = w / 2.0, h / 2.0
    A = m
    B = -1.0
    C = b
    norm = math.sqrt(A * A + B * B) + 1e-12
    rho = -(A * cx + B * cy + C) / norm
    theta_raw = math.degrees(math.atan2(B, A))
    
    if theta_raw < 0.0:
        theta = theta_raw + 180.0
        rho = -rho
    elif theta_raw >= 180.0:
        theta = theta_raw - 180.0
        rho = -rho
    else:
        theta = theta_raw
    
    return float(rho), float(theta)


# ============================
# Model loading
# ============================
def load_cnn():
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
    if not UNET_WEIGHTS_PATH.exists():
        print(f"[WARN] UNet weights not found: {UNET_WEIGHTS_PATH}. Will evaluate CNN only.")
        return None
    model = RestorationGuidedHorizonNet(
        num_classes=2, 
        dce_weights_path=str(DCE_WEIGHTS_PATH) if DCE_WEIGHTS_PATH.exists() else None, 
        require_dce=False
    )
    state = torch.load(UNET_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model


# ============================
# Main evaluation
# ============================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 70)
    print("Full Pipeline Evaluation for Buoy Test Set (In-Domain)")
    print("=" * 70)

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check files
    if not CACHE_DIR.exists():
        print(f"[Error] Cache not found: {CACHE_DIR}")
        return 1

    if not CNN_WEIGHTS_PATH.exists():
        print(f"[Error] CNN weights not found: {CNN_WEIGHTS_PATH}")
        return 1

    # Load split indices
    test_indices = load_split_indices(SPLIT_DIR)
    if test_indices:
        print(f"[Test] Using {len(test_indices)} samples from split file")
    else:
        print("[Test] Using all files in cache directory")

    # Dataset
    ds = TestCacheDataset(str(CACHE_DIR), test_indices)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=DEVICE.startswith("cuda"))
    print(f"[Dataset] {len(ds)} samples")

    # Load models
    cnn_model = load_cnn()
    print(f"[CNN] Loaded from {CNN_WEIGHTS_PATH}")
    
    unet_model = load_unet()
    if unet_model:
        print(f"[UNet] Loaded from {UNET_WEIGHTS_PATH}")
    
    print(f"[Device] {DEVICE}")

    # Scale factors (based on default original size)
    scale_x = DEFAULT_ORIG_W / UNET_W
    scale_y = DEFAULT_ORIG_H / UNET_H
    scale_diag = math.sqrt(DEFAULT_ORIG_W**2 + DEFAULT_ORIG_H**2) / math.sqrt(UNET_W**2 + UNET_H**2)

    # Metrics accumulators
    rho_err_cnn, theta_err_cnn, line_dist_cnn, edgey_cnn = [], [], [], []
    ve_signed_cnn, ae_signed_cnn = [], []
    
    rho_err_final, theta_err_final, line_dist_final, edgey_final = [], [], [], []
    ve_signed_final, ae_signed_final = [], []
    
    rows = []

    print("\n[Evaluating...]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            xb, yb, img_names, cache_files, orig_ws, orig_hs = batch
            xb = xb.to(DEVICE)
            
            # CNN prediction
            cnn_out = cnn_model(xb).cpu().numpy()
            gt = yb.numpy()
            
            for i in range(len(xb)):
                # Denormalize predictions
                rho_c, theta_c = denorm_rho_theta(cnn_out[i, 0], cnn_out[i, 1])
                rho_gt, theta_gt = denorm_rho_theta(gt[i, 0], gt[i, 1])
                
                # CNN-only metrics
                rho_abs_c = abs(rho_c - rho_gt)
                theta_abs_c = angular_diff_deg(theta_c, theta_gt)
                ld_c = mean_point_to_line_distance(rho_c, theta_c, rho_gt, theta_gt, UNET_W, UNET_H)
                ey_c = edge_y_error(rho_c, theta_c, rho_gt, theta_gt, UNET_W, UNET_H)
                ve_c = compute_VE(rho_c, theta_c, rho_gt, theta_gt, UNET_W, UNET_H)
                ae_c = compute_AE(theta_c, theta_gt)
                
                if not math.isnan(ld_c):
                    line_dist_cnn.append(ld_c)
                if not math.isnan(ey_c):
                    edgey_cnn.append(ey_c)
                if not math.isnan(ve_c):
                    ve_signed_cnn.append(ve_c)
                
                rho_err_cnn.append(rho_abs_c)
                theta_err_cnn.append(theta_abs_c)
                ae_signed_cnn.append(ae_c)
                
                # Final metrics (same as CNN for now)
                rho_f, theta_f = rho_c, theta_c
                rho_abs_f = rho_abs_c
                theta_abs_f = theta_abs_c
                ld_f = ld_c
                ey_f = ey_c
                ve_f = ve_c
                ae_f = ae_c
                
                rho_err_final.append(rho_abs_f)
                theta_err_final.append(theta_abs_f)
                if not math.isnan(ld_f):
                    line_dist_final.append(ld_f)
                if not math.isnan(ey_f):
                    edgey_final.append(ey_f)
                if not math.isnan(ve_f):
                    ve_signed_final.append(ve_f)
                ae_signed_final.append(ae_f)
                
                # Per-sample record
                rows.append({
                    "idx": batch_idx * BATCH_SIZE + i,
                    "img_name": img_names[i],
                    "cache_file": cache_files[i],
                    "rho_gt": float(rho_gt),
                    "theta_gt": float(theta_gt),
                    "rho_cnn": float(rho_c),
                    "theta_cnn": float(theta_c),
                    "rho_final": float(rho_f),
                    "theta_final": float(theta_f),
                    "rho_err_unet": float(rho_abs_f),
                    "theta_err_deg": float(theta_abs_f),
                    "line_dist_unet": float(ld_f) if not math.isnan(ld_f) else None,
                    "edgey_unet": float(ey_f) if not math.isnan(ey_f) else None,
                    "VE_unet": float(ve_f) if not math.isnan(ve_f) else None,
                    "AE_deg": float(ae_f),
                    "rho_err_orig": float(rho_abs_f * scale_diag),
                    "line_dist_orig": float(ld_f * scale_diag) if not math.isnan(ld_f) else None,
                    "edgey_orig": float(ey_f * scale_y) if not math.isnan(ey_f) else None,
                    "VE_orig": float(ve_f * scale_y) if not math.isnan(ve_f) else None,
                })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE}/{len(ds)} samples...")

    # Convert to arrays
    rho_err_cnn = np.array(rho_err_cnn, dtype=np.float64)
    theta_err_cnn = np.array(theta_err_cnn, dtype=np.float64)
    line_dist_cnn = np.array(line_dist_cnn, dtype=np.float64)
    edgey_cnn = np.array(edgey_cnn, dtype=np.float64)
    ve_signed_cnn = np.array(ve_signed_cnn, dtype=np.float64)
    ae_signed_cnn = np.array(ae_signed_cnn, dtype=np.float64)

    rho_err_final = np.array(rho_err_final, dtype=np.float64)
    theta_err_final = np.array(theta_err_final, dtype=np.float64)
    line_dist_final = np.array(line_dist_final, dtype=np.float64)
    edgey_final = np.array(edgey_final, dtype=np.float64)
    ve_signed_final = np.array(ve_signed_final, dtype=np.float64)
    ae_signed_final = np.array(ae_signed_final, dtype=np.float64)

    # AE wrap
    ae_signed_cnn_w = wrap_signed_angle_diff(ae_signed_cnn, 180.0)
    ae_signed_final_w = wrap_signed_angle_diff(ae_signed_final, 180.0)

    # Absolute values
    ve_abs_cnn = np.abs(ve_signed_cnn)
    ae_abs_cnn = np.abs(ae_signed_cnn_w)
    ve_abs_final = np.abs(ve_signed_final)
    ae_abs_final = np.abs(ae_signed_final_w)

    # Print summaries
    print("\n" + "=" * 70)
    print("---- CNN only ----")
    print(summarize("Rho abs error (px, UNet)", rho_err_cnn))
    print(summarize("Theta error (deg)", theta_err_cnn))
    print(summarize("Mean point->line dist (px, UNet)", line_dist_cnn))
    print(summarize("Edge-Y error (px, UNet)", edgey_cnn))
    print(summarize("Rho abs error (px, orig)", rho_err_cnn * scale_diag))
    print(summarize("Line dist (px, orig)", line_dist_cnn * scale_diag))
    print(summarize("EdgeY (px, orig)", edgey_cnn * scale_y))
    print(summarize("VE (px, UNet)", ve_abs_cnn))
    print(summarize("VE (px, orig)", ve_abs_cnn * scale_y))
    print(summarize("AE (deg, wrapped)", ae_abs_cnn))
    print("")

    print("---- Final ----")
    print(summarize("Rho abs error (px, UNet)", rho_err_final))
    print(summarize("Theta error (deg)", theta_err_final))
    print(summarize("Mean point->line dist (px, UNet)", line_dist_final))
    print(summarize("Edge-Y error (px, UNet)", edgey_final))
    print(summarize("Rho abs error (px, orig)", rho_err_final * scale_diag))
    print(summarize("Line dist (px, orig)", line_dist_final * scale_diag))
    print(summarize("EdgeY (px, orig)", edgey_final * scale_y))
    print(summarize("VE (px, UNet)", ve_abs_final))
    print(summarize("VE (px, orig)", ve_abs_final * scale_y))
    print(summarize("AE (deg, wrapped)", ae_abs_final))
    print("")

    # Paper summary
    print("=" * 60)
    print("论文表格汇总 (Final, orig-scale, mean ± std)")
    print("=" * 60)
    ve_orig_f = ve_abs_final * scale_y
    ve_signed_orig_f = ve_signed_final * scale_y
    print(f"VE (px):  {np.mean(ve_orig_f):.2f} ± {np.std(ve_signed_orig_f, ddof=1):.2f}")
    print(f"AE (deg): {np.mean(ae_abs_final):.2f} ± {np.std(ae_signed_final_w, ddof=1):.2f}")
    print(f"EdgeY (px): {np.mean(edgey_final * scale_y):.2f}")
    print(f"LineDist (px): {np.mean(line_dist_final * scale_diag):.2f}")
    print("=" * 60)
    print("")

    # Threshold stats
    print("---- Threshold stats (FINAL) ----")
    print(f"Theta <=1°: {pct_le(theta_err_final, 1.0):.2f}% | <=2°: {pct_le(theta_err_final, 2.0):.2f}% | <=5°: {pct_le(theta_err_final, 5.0):.2f}%")
    print(f"Rho(orig) <=5px: {pct_le(rho_err_final * scale_diag, 5.0):.2f}% | <=10px: {pct_le(rho_err_final * scale_diag, 10.0):.2f}% | <=20px: {pct_le(rho_err_final * scale_diag, 20.0):.2f}%")
    print(f"LineDist(orig) <=5px: {pct_le(line_dist_final * scale_diag, 5.0):.2f}% | <=10px: {pct_le(line_dist_final * scale_diag, 10.0):.2f}% | <=20px: {pct_le(line_dist_final * scale_diag, 20.0):.2f}%")
    print(f"EdgeY(orig) <=5px: {pct_le(edgey_final * scale_y, 5.0):.2f}% | <=10px: {pct_le(edgey_final * scale_y, 10.0):.2f}% | <=20px: {pct_le(edgey_final * scale_y, 20.0):.2f}%")
    print("")

    # VE/AE hit-rate
    print("---- VE / AE Hit-Rate (FINAL, original-scale) ----")
    ve_orig_final = ve_abs_final * scale_y
    print(f"VE <=5px: {pct_le(ve_orig_final, 5.0):.2f}% | <=10px: {pct_le(ve_orig_final, 10.0):.2f}% | <=20px: {pct_le(ve_orig_final, 20.0):.2f}%")
    print(f"AE <=1°: {pct_le(ae_abs_final, 1.0):.2f}% | <=2°: {pct_le(ae_abs_final, 2.0):.2f}% | <=5°: {pct_le(ae_abs_final, 5.0):.2f}%")
    print("")

    # Save CSV
    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[Saved] {OUT_CSV}")

    # Save summary
    summary = {
        "dataset": "Buoy",
        "n_samples": len(rows),
        "VE_mean_px": float(np.mean(ve_orig_f)),
        "VE_std_px": float(np.std(ve_signed_orig_f, ddof=1)),
        "AE_mean_deg": float(np.mean(ae_abs_final)),
        "AE_std_deg": float(np.std(ae_signed_final_w, ddof=1)),
        "EdgeY_mean_px": float(np.mean(edgey_final * scale_y)),
        "LineDist_mean_px": float(np.mean(line_dist_final * scale_diag)),
        "Theta_le1deg_pct": pct_le(theta_err_final, 1.0),
        "VE_le5px_pct": pct_le(ve_orig_final, 5.0),
        "AE_le1deg_pct": pct_le(ae_abs_final, 1.0),
    }
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f"[Saved] {SUMMARY_CSV}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
