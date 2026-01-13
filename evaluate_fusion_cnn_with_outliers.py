# -*- coding: utf-8 -*-
"""
Evaluate Fusion-CNN on cached FusionCache_* and save outlier visualizations.

- No CLI needed (PyCharm friendly).
- Deterministic outlier selection (topK + threshold).
- Saves:
    eval_outputs/eval_test.csv
    eval_outputs/outliers_test/*.png
    eval_outputs/outliers_test.txt
"""

import os
import math
import csv
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------
# User config (edit here)
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# weights from your training
WEIGHTS_PATH = r"splits_musid/best_fusion_cnn_1024x576.pth"

# fusion cache root for split
CACHE_DIR = r"Hashmani's Dataset/FusionCache_1024x576\test"

# original images root (for visualization). You MUST set this to your real image folder
# If you don't want to visualize on raw image, set to "" and it will visualize on a blank canvas.
ORIGINAL_IMG_ROOT = r"Hashmani's Dataset\MU-SID"  # <-- change to your image folder

# output dirs
OUT_DIR = r"eval_outputs"
OUTLIER_DIR = os.path.join(OUT_DIR, "outliers_test")
CSV_PATH = os.path.join(OUT_DIR, "eval_test.csv")
OUTLIER_TXT = os.path.join(OUT_DIR, "outliers_test.txt")

# cache sample format
# Each .npy expected to be dict-like or array:
#   - array shape (C,H,W) and label separately stored in same npy OR
#   - dict with keys {"input","label","img_path"} etc.
# This script supports two common patterns (see _load_one()).

# image size in UNet space (used to draw lines consistently)
UNET_W, UNET_H = 1024, 576

# outlier selection
TOPK_OUTLIERS = 50
THRESH_RHO_ORIG_PX = 20.0
THRESH_EDGEY_ORIG_PX = 20.0
THRESH_THETA_DEG = 2.0

# visualization options
DRAW_THICKNESS = 2
FONT_SCALE = 0.6

# -----------------------
# Helper math
# -----------------------
def wrap_angle_deg(err_deg: float, period: float = 180.0) -> float:
    """Wrap absolute error on a periodic angle domain (0..period)."""
    err = abs(err_deg) % period
    return min(err, period - err)

def polar_to_line_pts(theta_deg: float, rho: float, w: int, h: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Convert (theta, rho) in image coordinates to two endpoints within image.
    Here theta is in degrees, consistent with your pipeline (period 180).
    Using line normal form: x*cos + y*sin = rho
    """
    th = math.radians(theta_deg)
    a, b = math.cos(th), math.sin(th)
    # choose intersections with image borders
    pts = []

    # x=0 => y = (rho - 0*a)/b
    if abs(b) > 1e-6:
        y = (rho - 0 * a) / b
        if 0 <= y <= h-1:
            pts.append((0, int(round(y))))
        y = (rho - (w-1) * a) / b
        if 0 <= y <= h-1:
            pts.append((w-1, int(round(y))))

    # y=0 => x = (rho - 0*b)/a
    if abs(a) > 1e-6:
        x = (rho - 0 * b) / a
        if 0 <= x <= w-1:
            pts.append((int(round(x)), 0))
        x = (rho - (h-1) * b) / a
        if 0 <= x <= w-1:
            pts.append((int(round(x)), h-1))

    # dedup & pick two farthest
    pts_unique = []
    for p in pts:
        if p not in pts_unique:
            pts_unique.append(p)

    if len(pts_unique) >= 2:
        # choose farthest pair
        best = (pts_unique[0], pts_unique[1])
        best_d = -1
        for i in range(len(pts_unique)):
            for j in range(i+1, len(pts_unique)):
                dx = pts_unique[i][0] - pts_unique[j][0]
                dy = pts_unique[i][1] - pts_unique[j][1]
                d = dx*dx + dy*dy
                if d > best_d:
                    best_d = d
                    best = (pts_unique[i], pts_unique[j])
        return best
    # fallback: horizontal line
    return (0, int(h/2)), (w-1, int(h/2))

def point_line_dist_mean(theta_deg: float, rho: float, pts_xy: np.ndarray) -> float:
    """Mean distance from points to line x*cos+y*sin=rho."""
    th = math.radians(theta_deg)
    a, b = math.cos(th), math.sin(th)
    # distance = |ax+by-rho|
    d = np.abs(pts_xy[:,0]*a + pts_xy[:,1]*b - rho)
    return float(d.mean()) if len(d) else 0.0

def edge_y_error(theta_deg: float, rho_pred: float, rho_gt: float, x_ref: float = 0.0) -> float:
    """
    Convert rho error to y-intercept error at a reference x.
    For near-horizontal lines this approximates pixel shift at image edge.
    """
    th = math.radians(theta_deg)
    b = math.sin(th)
    a = math.cos(th)
    if abs(b) < 1e-6:
        return abs(rho_pred - rho_gt)  # degenerate
    y_pred = (rho_pred - x_ref * a) / b
    y_gt = (rho_gt - x_ref * a) / b
    return float(abs(y_pred - y_gt))

# -----------------------
# Dataset
# -----------------------
class FusionCacheDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.files = sorted([str(p) for p in Path(cache_dir).glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def _load_one(self, path: str):
        obj = np.load(path, allow_pickle=True)
        # pattern A: dict saved
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()

        if isinstance(obj, dict):
            x = obj.get("input", None)
            y = obj.get("label", None)
            img_name = obj.get("img_name", None) or obj.get("image_name", None)
        else:
            # pattern B: array saved directly; label may be in same file last element
            # If your cache format differs, adjust here.
            arr = obj
            if arr.ndim == 3:
                x = arr
                y = None
            elif arr.ndim == 1 and len(arr) == 2 and isinstance(arr[0], np.ndarray):
                x = arr[0]
                y = arr[1]
            else:
                x = arr
                y = None
            img_name = None

        # If label not found, try infer from separate label file (optional)
        if y is None:
            raise RuntimeError(f"Label not found in cache npy: {path}")

        if img_name is None:
            # fallback: use npy filename with jpg extension guess
            img_name = Path(path).stem + ".jpg"

        x = x.astype(np.float32)
        y = np.array(y, dtype=np.float32)  # [rho_norm, theta_norm] or [rho,theta] depending your cache
        return x, y, img_name

    def __getitem__(self, idx):
        path = self.files[idx]
        x, y, img_name = self._load_one(path)
        return torch.from_numpy(x), torch.from_numpy(y), img_name, os.path.basename(path)

# -----------------------
# Model loader
# -----------------------
def load_model():
    # Import your model here
    from cnn_model import HorizonResNet  # must exist in your project
    model = HorizonResNet(in_channels=4)  # if your input channels differ, modify
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    # support either {'state_dict':...} or raw state dict
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # strip "module." if needed
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Label / prediction decoding
# -----------------------
def decode_label(label: np.ndarray) -> Tuple[float, float]:
    """
    Convert label array into (rho_px, theta_deg) in UNet space.
    Your existing pipeline seems to store normalized [rho_norm, theta_norm] in [0,1].
    If your cache uses different encoding, adjust here to match your training.
    """
    rho_norm = float(label[0])
    theta_norm = float(label[1])
    rho_px = rho_norm * math.sqrt(UNET_W**2 + UNET_H**2)  # same as your evaluation earlier (approx)
    theta_deg = theta_norm * 180.0
    return rho_px, theta_deg

def decode_pred(pred: np.ndarray) -> Tuple[float, float]:
    rho_norm = float(pred[0])
    theta_norm = float(pred[1])
    rho_px = rho_norm * math.sqrt(UNET_W**2 + UNET_H**2)
    theta_deg = theta_norm * 180.0
    return rho_px, theta_deg

# -----------------------
# Visualization
# -----------------------
def load_original_image(img_name: str) -> np.ndarray:
    if not ORIGINAL_IMG_ROOT:
        return np.zeros((UNET_H, UNET_W, 3), dtype=np.uint8)
    cand = os.path.join(ORIGINAL_IMG_ROOT, img_name)
    if not os.path.exists(cand):
        # fallback blank
        return np.zeros((UNET_H, UNET_W, 3), dtype=np.uint8)
    im = cv2.imread(cand, cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((UNET_H, UNET_W, 3), dtype=np.uint8)
    im = cv2.resize(im, (UNET_W, UNET_H))
    return im

def draw_lines_and_text(img: np.ndarray,
                        rho_gt: float, theta_gt: float,
                        rho_pr: float, theta_pr: float,
                        text_lines: List[str]) -> np.ndarray:
    out = img.copy()
    p1g, p2g = polar_to_line_pts(theta_gt, rho_gt, UNET_W, UNET_H)
    p1p, p2p = polar_to_line_pts(theta_pr, rho_pr, UNET_W, UNET_H)

    # GT green
    cv2.line(out, p1g, p2g, (0, 255, 0), DRAW_THICKNESS)
    # Pred red
    cv2.line(out, p1p, p2p, (0, 0, 255), DRAW_THICKNESS)

    y0 = 22
    for t in text_lines:
        cv2.putText(out, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), 1, cv2.LINE_AA)
        y0 += 20
    return out

# -----------------------
# Main evaluation
# -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUTLIER_DIR, exist_ok=True)

    ds = FusionCacheDataset(CACHE_DIR)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = load_model()

    diag = math.sqrt(UNET_W**2 + UNET_H**2)
    scale_to_orig = 1920.0 / UNET_W  # approx, consistent with your earlier script

    rows: List[Dict] = []

    with torch.no_grad():
        for xb, yb, img_names, npy_names in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            pred = model(xb)
            pred = pred.detach().cpu().numpy()
            yb = yb.numpy()

            for i in range(len(img_names)):
                rho_gt, theta_gt = decode_label(yb[i])
                rho_pr, theta_pr = decode_pred(pred[i])

                rho_err = abs(rho_pr - rho_gt)
                theta_err = wrap_angle_deg(theta_pr - theta_gt, 180.0)

                # mean point->line distance: sample points along GT line (approx)
                # We approximate by using endpoints of GT line and interpolate.
                p1, p2 = polar_to_line_pts(theta_gt, rho_gt, UNET_W, UNET_H)
                xs = np.linspace(p1[0], p2[0], 200)
                ys = np.linspace(p1[1], p2[1], 200)
                pts = np.stack([xs, ys], axis=1).astype(np.float32)
                line_dist = point_line_dist_mean(theta_pr, rho_pr, pts)

                edge_y = edge_y_error(theta_gt, rho_pr, rho_gt, x_ref=0.0)

                rows.append({
                    "img_name": img_names[i],
                    "npy": npy_names[i],
                    "rho_gt": rho_gt, "theta_gt": theta_gt,
                    "rho_pred": rho_pr, "theta_pred": theta_pr,
                    "rho_err": rho_err,
                    "theta_err_deg": theta_err,
                    "line_dist": line_dist,
                    "edge_y": edge_y,
                    "rho_err_orig": rho_err * scale_to_orig,
                    "line_dist_orig": line_dist * scale_to_orig,
                    "edge_y_orig": edge_y * scale_to_orig,
                })

    # write csv
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # select outliers
    # threshold based
    out_thr = [r for r in rows if
               (r["rho_err_orig"] >= THRESH_RHO_ORIG_PX) or
               (r["edge_y_orig"] >= THRESH_EDGEY_ORIG_PX) or
               (r["theta_err_deg"] >= THRESH_THETA_DEG)]
    # topK by edge_y_orig
    out_top = sorted(rows, key=lambda d: d["edge_y_orig"], reverse=True)[:TOPK_OUTLIERS]

    # merge
    out_map = {(r["img_name"], r["npy"]): r for r in (out_thr + out_top)}
    outliers = list(out_map.values())
    outliers = sorted(outliers, key=lambda d: d["edge_y_orig"], reverse=True)

    # save outlier txt + images
    with open(OUTLIER_TXT, "w", encoding="utf-8") as f:
        for k, r in enumerate(outliers):
            f.write(f"{k}\t{r['img_name']}\ttheta={r['theta_err_deg']:.3f}\t"
                    f"rho~{r['rho_err_orig']:.2f}\tlineDist~{r['line_dist_orig']:.2f}\tedgeY~{r['edge_y_orig']:.2f}\n")

    for r in outliers:
        img = load_original_image(r["img_name"])
        text = [
            f"{r['img_name']}",
            f"theta_err={r['theta_err_deg']:.3f} deg",
            f"rho_err={r['rho_err_orig']:.2f}px (orig approx)",
            f"edgeY_err={r['edge_y_orig']:.2f}px",
        ]
        vis = draw_lines_and_text(img, r["rho_gt"], r["theta_gt"], r["rho_pred"], r["theta_pred"], text)
        out_path = os.path.join(OUTLIER_DIR, Path(r["img_name"]).stem + ".png")
        cv2.imwrite(out_path, vis)

    # print summary
    rho = np.array([r["rho_err"] for r in rows], dtype=np.float32)
    rhoo = np.array([r["rho_err_orig"] for r in rows], dtype=np.float32)
    th = np.array([r["theta_err_deg"] for r in rows], dtype=np.float32)
    ld = np.array([r["line_dist"] for r in rows], dtype=np.float32)
    ldo = np.array([r["line_dist_orig"] for r in rows], dtype=np.float32)
    ey = np.array([r["edge_y"] for r in rows], dtype=np.float32)
    eyo = np.array([r["edge_y_orig"] for r in rows], dtype=np.float32)

    def stat(x):
        return float(x.mean()), float(np.median(x)), float(np.percentile(x, 90)), float(np.percentile(x, 95)), float(x.max())

    print("========== Fusion-CNN Evaluation (with outliers) ==========")
    print(f"Split: test | N={len(rows)} | Device={DEVICE}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Cache:   {CACHE_DIR}\n")

    m, md, p90, p95, mx = stat(rho)
    print(f"Rho abs error (px, UNet {UNET_W}x{UNET_H}): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")
    m, md, p90, p95, mx = stat(rhoo)
    print(f"Rho abs error (px, original-scale approx): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")

    m, md, p90, p95, mx = stat(th)
    print(f"Theta error (deg, wrap-aware, period=180): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")

    m, md, p90, p95, mx = stat(ld)
    print(f"Mean point->line distance (px, UNet): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")
    m, md, p90, p95, mx = stat(ldo)
    print(f"Mean point->line distance (px, original-scale approx): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")

    m, md, p90, p95, mx = stat(ey)
    print(f"Edge-Y error (px, UNet): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")
    m, md, p90, p95, mx = stat(eyo)
    print(f"Edge-Y error (px, original-scale approx): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}\n")

    # threshold stats
    theta1 = float((th <= 1.0).mean() * 100)
    theta2 = float((th <= 2.0).mean() * 100)
    theta5 = float((th <= 5.0).mean() * 100)
    rho5 = float((rhoo <= 5.0).mean() * 100)
    rho10 = float((rhoo <= 10.0).mean() * 100)
    rho20 = float((rhoo <= 20.0).mean() * 100)
    ld5 = float((ldo <= 5.0).mean() * 100)
    ld10 = float((ldo <= 10.0).mean() * 100)
    ld20 = float((ldo <= 20.0).mean() * 100)
    ey5 = float((eyo <= 5.0).mean() * 100)
    ey10 = float((eyo <= 10.0).mean() * 100)
    ey20 = float((eyo <= 20.0).mean() * 100)

    print("---- Threshold stats (helpful for论文表格) ----")
    print(f"Theta <=1°: {theta1:.2f}% | <=2°: {theta2:.2f}% | <=5°: {theta5:.2f}%")
    print(f"Rho(orig) <=5px: {rho5:.2f}% | <=10px: {rho10:.2f}% | <=20px: {rho20:.2f}%")
    print(f"LineDist(orig) <=5px: {ld5:.2f}% | <=10px: {ld10:.2f}% | <=20px: {ld20:.2f}%")
    print(f"EdgeY(orig) <=5px: {ey5:.2f}% | <=10px: {ey10:.2f}% | <=20px: {ey20:.2f}%\n")

    print(f"[Saved] per-sample metrics -> {CSV_PATH}")
    print(f"[Outliers] selected={len(outliers)} (threshold-or-topK)")
    print(f"[Outliers] saving visualization to: {OUTLIER_DIR}")
    print(f"[Saved] outlier list -> {OUTLIER_TXT}")
    print("Done.")

if __name__ == "__main__":
    main()
