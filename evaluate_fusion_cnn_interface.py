"""
evaluate_fusion_cnn.py
------------------------------------------------------------
Evaluate the final Fusion-CNN (trained on FusionCache) on a split (default: test).

Metrics (all computed against cached labels):
  1) |rho_pred - rho_gt| in pixels (in the UNet-resized image space, default 1024x576)
  2) Angular error in degrees (theta in [0,180), wrap-around aware)
  3) Mean perpendicular distance (px) from GT line points to predicted line (more "line-centric")

Optional:
  - Also report rho error in original image scale (default assumes original 1920x1080)
  - Save per-sample errors to CSV for analysis.

Usage example:
  python evaluate_fusion_cnn.py ^
    --weights "splits_musid/best_fusion_cnn_1024x576.pth" ^
    --cache_root "Hashmani's Dataset/FusionCache_1024x576" ^
    --split test ^
    --batch_size 64 ^
    --out_csv "eval_test.csv"
"""

import argparse
import os
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import your model
from cnn_model import HorizonResNet


# -------------------------
# Cache dataset
# -------------------------
class SplitCacheDataset(Dataset):
    """
    Expects cache files saved by make_fusion_cache.py:
      each {idx}.npy is a dict with keys:
        - "input":  (4, 2240, 180) float32
        - "label":  (2,) [rho_norm, theta_norm] float32
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".npy")],
                            key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        fn = self.files[i]
        idx = int(os.path.splitext(fn)[0])
        path = os.path.join(self.cache_dir, fn)
        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()   # [4,2240,180]
        y = torch.from_numpy(data["label"]).float()   # [2]
        return x, y, idx


# -------------------------
# Param denormalization
# -------------------------
@dataclass
class DenormConfig:
    unet_w: int = 1024
    unet_h: int = 576
    resize_h: int = 2240          # rho axis length in cache
    angle_range_deg: float = 180.0
    # if you want original-scale pixel errors (e.g., 1920x1080)
    orig_w: int = 1920
    orig_h: int = 1080


def denorm_rho_theta(
    rho_norm: np.ndarray,
    theta_norm: np.ndarray,
    cfg: DenormConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized label/pred into:
      rho_real: signed distance (px) in centered coords of UNet-resized image (cfg.unet_w x cfg.unet_h)
      theta_deg: normal angle in degrees in [0, 180)
    This matches make_fusion_cache.py::calculate_radon_label and test.py::get_line_ends.
    """
    w, h = cfg.unet_w, cfg.unet_h
    diag = math.sqrt(w * w + h * h)
    pad_top = (cfg.resize_h - diag) / 2.0

    # final_rho_idx = rho_norm * (resize_h - 1)
    final_rho_idx = rho_norm * (cfg.resize_h - 1.0)

    # invert: rho_real = final_rho_idx - pad_top - diag/2
    rho_real = final_rho_idx - pad_top - (diag / 2.0)

    # theta in [0,180)
    theta_deg = (theta_norm * cfg.angle_range_deg) % cfg.angle_range_deg
    return rho_real, theta_deg


def angular_diff_deg(a: np.ndarray, b: np.ndarray, period: float = 180.0) -> np.ndarray:
    """Minimal absolute angular difference for angles on a circle of given period."""
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


# -------------------------
# Line distance metric
# -------------------------
def line_intersections_in_image(rho: float, theta_deg: float, w: int, h: int, eps: float = 1e-8) -> List[Tuple[float, float]]:
    """
    Line in centered coords: (x-cx)*cos + (y-cy)*sin = rho
    Return intersection points with the image rectangle in absolute coords.
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    pts: List[Tuple[float, float]] = []

    # x = 0
    if abs(sin_t) > eps:
        y = cy + (rho - ((0 - cx) * cos_t)) / sin_t
        if -1 <= y <= h:  # loose tolerance
            pts.append((0.0, float(y)))

    # x = w-1
    if abs(sin_t) > eps:
        x = w - 1.0
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if -1 <= y <= h:
            pts.append((x, float(y)))

    # y = 0
    if abs(cos_t) > eps:
        y = 0.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if -1 <= x <= w:
            pts.append((float(x), y))

    # y = h-1
    if abs(cos_t) > eps:
        y = h - 1.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if -1 <= x <= w:
            pts.append((float(x), y))

    # keep only inside (strict)
    pts2 = []
    for x, y in pts:
        if 0.0 <= x <= (w - 1.0) and 0.0 <= y <= (h - 1.0):
            pts2.append((x, y))
    return pts2


def farthest_pair(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Pick the farthest pair among pts (>=2)."""
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
    rho_pred: float,
    theta_pred_deg: float,
    rho_gt: float,
    theta_gt_deg: float,
    w: int,
    h: int,
    n_samples: int = 50
) -> float:
    """
    Sample points on GT line segment inside the image and compute mean perpendicular distance to predicted line.
    Distance is computed in the UNet-resized pixel space (w x h).
    """
    # 1) get GT line segment endpoints
    pts = line_intersections_in_image(rho_gt, theta_gt_deg, w, h)
    if len(pts) >= 2:
        p0, p1 = farthest_pair(pts)
    else:
        # fallback: sample along x=0..w-1 using GT y(x), even if outside -> clamp
        theta = math.radians(theta_gt_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = w / 2.0, h / 2.0
        x0, x1 = 0.0, w - 1.0
        if abs(sin_t) < 1e-8:
            # near-vertical in normal angle -> line is near-horizontal? Actually sin small => normal near 0/180.
            # fallback: just pick mid-y
            y0 = y1 = cy
        else:
            y0 = cy + (rho_gt - ((x0 - cx) * cos_t)) / sin_t
            y1 = cy + (rho_gt - ((x1 - cx) * cos_t)) / sin_t
        y0 = float(np.clip(y0, 0, h - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        p0, p1 = (x0, y0), (x1, y1)

    # 2) sample points along GT segment
    xs = np.linspace(p0[0], p1[0], n_samples)
    ys = np.linspace(p0[1], p1[1], n_samples)

    # 3) distance to predicted line
    theta_p = math.radians(theta_pred_deg)
    cos_p, sin_p = math.cos(theta_p), math.sin(theta_p)
    cx, cy = w / 2.0, h / 2.0

    x_c = xs - cx
    y_c = ys - cy
    # perpendicular distance since cos^2 + sin^2 = 1
    d = np.abs(x_c * cos_p + y_c * sin_p - rho_pred)
    return float(np.mean(d))


# -------------------------
# Summary helpers
# -------------------------
def summarize(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    return (
        f"{name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}, max={arr.max():.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=r"splits_musid/best_fusion_cnn_1024x576.pth")
    ap.add_argument("--cache_root", type=str, default=r"Hashmani's Dataset/FusionCache_1024x576")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="")
    # denorm config
    ap.add_argument("--unet_w", type=int, default=1024)
    ap.add_argument("--unet_h", type=int, default=576)
    ap.add_argument("--resize_h", type=int, default=2240)
    ap.add_argument("--orig_w", type=int, default=1920)
    ap.add_argument("--orig_h", type=int, default=1080)
    ap.add_argument("--line_samples", type=int, default=50)

    args = ap.parse_args()

    cfg = DenormConfig(
        unet_w=args.unet_w,
        unet_h=args.unet_h,
        resize_h=args.resize_h,
        orig_w=args.orig_w,
        orig_h=args.orig_h,
    )
    scale = cfg.orig_w / cfg.unet_w  # uniform scale in your pipeline (1920->1024, 1080->576)

    split_dir = os.path.join(args.cache_root, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    ds = SplitCacheDataset(split_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))

    model = HorizonResNet(in_channels=4, img_h=cfg.resize_h, img_w=180).to(args.device)
    ckpt = torch.load(args.weights, map_location=args.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    rho_err_px = []
    rho_err_px_orig = []
    theta_err_deg = []
    line_dist_px = []

    rows = []  # optional csv rows

    with torch.no_grad():
        for xb, yb, idxb in dl:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)

            pred = model(xb)  # [B,2] in [0,1]
            pred_np = pred.detach().cpu().numpy()
            gt_np = yb.detach().cpu().numpy()
            idx_np = idxb.detach().cpu().numpy()

            # denorm
            rho_p, th_p = denorm_rho_theta(pred_np[:, 0], pred_np[:, 1], cfg)
            rho_g, th_g = denorm_rho_theta(gt_np[:, 0], gt_np[:, 1], cfg)

            # errors
            e_rho = np.abs(rho_p - rho_g)  # px in 1024x576
            e_theta = angular_diff_deg(th_p, th_g, period=180.0)

            for i in range(len(idx_np)):
                ld = mean_point_to_line_distance(
                    rho_pred=float(rho_p[i]),
                    theta_pred_deg=float(th_p[i]),
                    rho_gt=float(rho_g[i]),
                    theta_gt_deg=float(th_g[i]),
                    w=cfg.unet_w,
                    h=cfg.unet_h,
                    n_samples=args.line_samples,
                )
                line_dist_px.append(ld)
                rho_err_px.append(float(e_rho[i]))
                rho_err_px_orig.append(float(e_rho[i] * scale))
                theta_err_deg.append(float(e_theta[i]))

                if args.out_csv:
                    rows.append({
                        "idx": int(idx_np[i]),
                        "rho_gt_norm": float(gt_np[i, 0]),
                        "theta_gt_norm": float(gt_np[i, 1]),
                        "rho_pred_norm": float(pred_np[i, 0]),
                        "theta_pred_norm": float(pred_np[i, 1]),
                        "rho_err_px_unet": float(e_rho[i]),
                        "rho_err_px_orig": float(e_rho[i] * scale),
                        "theta_err_deg": float(e_theta[i]),
                        "line_dist_px_unet": float(ld),
                    })

    rho_err_px = np.asarray(rho_err_px, dtype=np.float64)
    rho_err_px_orig = np.asarray(rho_err_px_orig, dtype=np.float64)
    theta_err_deg = np.asarray(theta_err_deg, dtype=np.float64)
    line_dist_px = np.asarray(line_dist_px, dtype=np.float64)

    print("========== Evaluation ==========")
    print(f"Split: {args.split} | N={len(ds)}")
    print(f"Weights: {args.weights}")
    print(f"Cache:   {split_dir}")
    print("")
    print(summarize("Rho abs error (px, UNet space 1024x576)", rho_err_px))
    print(summarize(f"Rho abs error (px, original ~{cfg.orig_w}x{cfg.orig_h})", rho_err_px_orig))
    print(summarize("Theta error (deg, wrap-aware, period=180)", theta_err_deg))
    print(summarize("Mean point->line distance (px, UNet space)", line_dist_px))
    print("")

    # Useful threshold stats
    def pct_le(arr, thr):
        return 100.0 * float(np.mean(arr <= thr))

    print("---- Thresholds ----")
    print(f"theta <= 1°:  {pct_le(theta_err_deg, 1):.2f}% | <=2°: {pct_le(theta_err_deg, 2):.2f}% | <=5°: {pct_le(theta_err_deg, 5):.2f}%")
    print(f"rho_orig <= 5px: {pct_le(rho_err_px_orig, 5):.2f}% | <=10px: {pct_le(rho_err_px_orig, 10):.2f}% | <=20px: {pct_le(rho_err_px_orig, 20):.2f}%")
    print(f"line_dist <= 5px: {pct_le(line_dist_px, 5):.2f}% | <=10px: {pct_le(line_dist_px, 10):.2f}% | <=20px: {pct_le(line_dist_px, 20):.2f}%")

    if args.out_csv:
        out_path = args.out_csv
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("")
        print(f"[Saved] per-sample metrics -> {out_path}")


if __name__ == "__main__":
    main()
