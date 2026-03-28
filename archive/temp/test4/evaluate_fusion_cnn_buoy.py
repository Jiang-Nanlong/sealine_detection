# -*- coding: utf-8 -*-
"""
Evaluate Fusion-CNN on Buoy FusionCache (Experiment 4: zero-shot generalization).

This is a Buoy-specialized wrapper around evaluate_fusion_cnn.py with:
  - Sensible defaults pointing to ./test4/FusionCache_Buoy
  - Per-video breakdown (inferred from cached img_name)

Assumptions:
  - You already ran:
      python test4/prepare_buoy_testset.py
      python test4/make_fusion_cache_buoy.py
  - Cached .npy contains keys: input, label, img_name

Usage:
  python test4/evaluate_fusion_cnn_buoy.py
"""

import os
import sys
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Path setup
try:
    from cnn_model import HorizonResNet
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    from cnn_model import HorizonResNet


# -------------------------
# Cache dataset
# -------------------------
class SplitCacheDataset(Dataset):
    """Loads cache files produced by make_fusion_cache_buoy.py."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        if not os.path.isdir(cache_dir):
            self.files = []
        else:
            self.files = sorted(
                [f for f in os.listdir(cache_dir) if f.endswith(".npy")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        fn = self.files[i]
        idx = int(os.path.splitext(fn)[0])
        path = os.path.join(self.cache_dir, fn)
        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()
        y = torch.from_numpy(data["label"]).float()
        img_name = str(data.get("img_name", ""))
        # 读取原始图像尺寸（如果存在）
        orig_w = int(data.get("orig_w", 800))
        orig_h = int(data.get("orig_h", 600))
        return x, y, idx, img_name, orig_w, orig_h


# -------------------------
# Param denormalization
# -------------------------
@dataclass
class DenormConfig:
    unet_w: int = 1024
    unet_h: int = 576
    resize_h: int = 2240
    angle_range_deg: float = 180.0
    orig_w: int = 800  # Buoy video typical resolution
    orig_h: int = 600


def denorm_rho_theta(rho_norm: np.ndarray, theta_norm: np.ndarray, cfg: DenormConfig) -> Tuple[np.ndarray, np.ndarray]:
    w, h = cfg.unet_w, cfg.unet_h
    diag = math.sqrt(w * w + h * h)
    pad_top = (cfg.resize_h - diag) / 2.0

    final_rho_idx = rho_norm * (cfg.resize_h - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)

    theta_deg = (theta_norm * cfg.angle_range_deg) % cfg.angle_range_deg
    return rho_real, theta_deg


def angular_diff_deg(a: np.ndarray, b: np.ndarray, period: float = 180.0) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


# -------------------------
# Line distance metric
# -------------------------
def line_intersections_in_image(
    rho: float,
    theta_deg: float,
    w: int,
    h: int,
    eps: float = 1e-8,
) -> List[Tuple[float, float]]:
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    pts: List[Tuple[float, float]] = []

    # x = 0
    if abs(sin_t) > eps:
        y = cy + (rho - ((0 - cx) * cos_t)) / sin_t
        if -1 <= y <= h:
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

    pts2 = []
    for x, y in pts:
        if 0.0 <= x <= (w - 1.0) and 0.0 <= y <= (h - 1.0):
            pts2.append((x, y))
    return pts2


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
    rho_pred: float,
    theta_pred_deg: float,
    rho_gt: float,
    theta_gt_deg: float,
    w: int,
    h: int,
    n_samples: int = 50,
) -> float:
    pts = line_intersections_in_image(rho_gt, theta_gt_deg, w, h)
    if len(pts) >= 2:
        p0, p1 = farthest_pair(pts)
    else:
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


# -------------------------
# Reporting helpers
# -------------------------
def summarize(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return f"{name}: N=0"
    return (
        f"{name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}, max={arr.max():.4f}"
    )


def video_from_img_name(img_name: str) -> str:
    """Extract video name from img_name (format: <video_stem>__<frame>.jpg)."""
    parts = img_name.rsplit("__", 1)
    if len(parts) >= 1:
        return parts[0]
    return "Unknown"


def pct_le(arr: np.ndarray, thr: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


def main():
    """Main evaluation function."""
    # -------------------------------------------------------------
    # Path configuration
    # -------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # ============================
    # Configuration
    # ============================
    WEIGHTS = os.path.join(project_root, "weights", "best_fusion_cnn_1024x576.pth")
    CACHE_ROOT = os.path.join(script_dir, "FusionCache_Buoy")
    SPLIT = "test"
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_CSV = os.path.join(script_dir, "eval_buoy_test_per_sample.csv")

    # Denorm config (Buoy videos are typically 800x600)
    UNET_W = 1024
    UNET_H = 576
    RESIZE_H = 2240
    ORIG_W = 800
    ORIG_H = 600
    LINE_SAMPLES = 50
    # ============================

    cfg = DenormConfig(
        unet_w=UNET_W,
        unet_h=UNET_H,
        resize_h=RESIZE_H,
        orig_w=ORIG_W,
        orig_h=ORIG_H,
    )
    # scale 现在在循环中按每张图计算，这里不再使用全局scale

    split_dir = os.path.join(CACHE_ROOT, SPLIT)
    if not os.path.isdir(split_dir):
        print(f"[Error] Directory not found: {split_dir}")
        print(f"  - CACHE_ROOT: {CACHE_ROOT}")
        print("  Please run make_fusion_cache_buoy.py first.")
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    ds = SplitCacheDataset(split_dir)
    if len(ds) == 0:
        print(f"[Warning] No .npy files found in {split_dir}")
        return

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )

    # Check weights
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS}")

    model = HorizonResNet(in_channels=4, img_h=cfg.resize_h, img_w=180).to(DEVICE)
    ckpt = torch.load(WEIGHTS, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    # Global arrays
    g_rho_err = []
    g_rho_err_orig = []
    g_theta_err = []
    g_line_dist = []

    # Per-video metrics
    per_video: Dict[str, Dict[str, list]] = {}

    rows = []

    print("=" * 60)
    print("Buoy Evaluation (Fusion-CNN)")
    print("=" * 60)
    print(f"Split: {SPLIT} | N={len(ds)}")
    print(f"Weights: {WEIGHTS}")
    print(f"Cache:   {split_dir}")
    print("")

    with torch.no_grad():
        for xb, yb, idxb, names, orig_ws, orig_hs in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            pred = model(xb)
            pred_np = pred.detach().cpu().numpy()
            gt_np = yb.detach().cpu().numpy()
            idx_np = idxb.detach().cpu().numpy()
            names = list(names)
            orig_ws = orig_ws.numpy()
            orig_hs = orig_hs.numpy()

            rho_p, th_p = denorm_rho_theta(pred_np[:, 0], pred_np[:, 1], cfg)
            rho_g, th_g = denorm_rho_theta(gt_np[:, 0], gt_np[:, 1], cfg)

            e_rho = np.abs(rho_p - rho_g)
            e_theta = angular_diff_deg(th_p, th_g, period=180.0)

            for i in range(len(idx_np)):
                # 使用每张图实际的原始尺寸计算scale
                img_orig_w = int(orig_ws[i])
                img_orig_h = int(orig_hs[i])
                img_scale = img_orig_w / cfg.unet_w

                ld = mean_point_to_line_distance(
                    rho_pred=float(rho_p[i]),
                    theta_pred_deg=float(th_p[i]),
                    rho_gt=float(rho_g[i]),
                    theta_gt_deg=float(th_g[i]),
                    w=cfg.unet_w,
                    h=cfg.unet_h,
                    n_samples=LINE_SAMPLES,
                )

                video = video_from_img_name(names[i])

                # Global (使用每张图实际的scale)
                g_rho_err.append(float(e_rho[i]))
                g_rho_err_orig.append(float(e_rho[i] * img_scale))
                g_theta_err.append(float(e_theta[i]))
                g_line_dist.append(float(ld))

                # Per-video
                if video not in per_video:
                    per_video[video] = {"rho": [], "rho_o": [], "theta": [], "line": []}
                per_video[video]["rho"].append(float(e_rho[i]))
                per_video[video]["rho_o"].append(float(e_rho[i] * img_scale))
                per_video[video]["theta"].append(float(e_theta[i]))
                per_video[video]["line"].append(float(ld))

                if OUT_CSV:
                    rows.append({
                        "idx": int(idx_np[i]),
                        "img_name": names[i],
                        "video": video,
                        "orig_w": img_orig_w,
                        "orig_h": img_orig_h,
                        "rho_gt_norm": float(gt_np[i, 0]),
                        "theta_gt_norm": float(gt_np[i, 1]),
                        "rho_pred_norm": float(pred_np[i, 0]),
                        "theta_pred_norm": float(pred_np[i, 1]),
                        "rho_err_px_unet": float(e_rho[i]),
                        "rho_err_px_orig": float(e_rho[i] * img_scale),
                        "theta_err_deg": float(e_theta[i]),
                        "line_dist_px_unet": float(ld),
                    })

    # Convert to arrays
    g_rho_err = np.asarray(g_rho_err, dtype=np.float64)
    g_rho_err_orig = np.asarray(g_rho_err_orig, dtype=np.float64)
    g_theta_err = np.asarray(g_theta_err, dtype=np.float64)
    g_line_dist = np.asarray(g_line_dist, dtype=np.float64)

    print("[Overall - 原图尺寸]")
    print(summarize("Rho abs error (px, 原图尺寸)", g_rho_err_orig))
    print(summarize("Theta error (deg)", g_theta_err))
    print("---- 阈值统计 (原图尺寸) ----")
    print(f"rho <= 5px: {pct_le(g_rho_err_orig, 5):.2f}% | <=10px: {pct_le(g_rho_err_orig, 10):.2f}% | <=20px: {pct_le(g_rho_err_orig, 20):.2f}%")
    print(f"theta <= 1°: {pct_le(g_theta_err, 1):.2f}% | <=2°: {pct_le(g_theta_err, 2):.2f}% | <=5°: {pct_le(g_theta_err, 5):.2f}%")

    print("\n[Overall - UNet空间 (仅供参考)]")
    print(summarize("Rho abs error (px, 1024x576)", g_rho_err))
    print(summarize("Line distance (px, 1024x576)", g_line_dist))

    # Per-video report
    print("\n[Per-video breakdown - 原图尺寸]")
    for video in sorted(per_video.keys()):
        arr_rho = np.asarray(per_video[video]["rho"], dtype=np.float64)
        arr_rho_o = np.asarray(per_video[video]["rho_o"], dtype=np.float64)
        arr_theta = np.asarray(per_video[video]["theta"], dtype=np.float64)
        arr_line = np.asarray(per_video[video]["line"], dtype=np.float64)

        if arr_rho.size == 0:
            continue
        print(f"\n--- {video} | N={arr_rho.size} ---")
        print(summarize("Rho error (px, 原图)", arr_rho_o))
        print(summarize("Theta error (deg)", arr_theta))
        print(f"rho<=10px: {pct_le(arr_rho_o, 10):.2f}% | theta<=2°: {pct_le(arr_theta, 2):.2f}%")

    if OUT_CSV and rows:
        out_path = OUT_CSV
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] per-sample metrics -> {out_path}")


if __name__ == "__main__":
    main()
