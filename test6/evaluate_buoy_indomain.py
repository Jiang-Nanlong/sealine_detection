# -*- coding: utf-8 -*-
"""
Evaluate Fusion-CNN on Buoy test set (Experiment 6: In-Domain).

Inputs:
  - test6/FusionCache_Buoy/test/
  - test6/weights/best_fusion_cnn_buoy.pth

Outputs:
  - test6/eval_buoy_indomain.csv
  - 终端输出统计信息

PyCharm: 直接运行此文件
"""

import os
import sys
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_model import HorizonResNet  # noqa: E402


# ============================
# PyCharm 配置区
# ============================
BATCH_SIZE = 64
NUM_WORKERS = 0
# ============================

# Paths
TEST6_DIR = PROJECT_ROOT / "test6"
CACHE_DIR = TEST6_DIR / "FusionCache_Buoy" / "test"
SPLIT_DIR = TEST6_DIR / "splits_buoy"
WEIGHTS_PATH = TEST6_DIR / "weights" / "best_fusion_cnn_buoy.pth"
OUT_CSV = TEST6_DIR / "eval_buoy_indomain.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Denorm config
UNET_W, UNET_H = 1024, 576
RESIZE_H = 2240


# ============================
# Dataset
# ============================
class TestCacheDataset(Dataset):
    def __init__(self, cache_dir: str, indices: list):
        self.cache_dir = cache_dir
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()
        y = torch.from_numpy(data["label"]).float()
        img_name = str(data.get("img_name", ""))
        orig_w = int(data.get("orig_w", 1920))
        orig_h = int(data.get("orig_h", 1080))
        return x, y, idx, img_name, orig_w, orig_h


def load_split_indices(split_dir):
    return np.load(os.path.join(split_dir, "test_indices.npy")).astype(np.int64).tolist()


# ============================
# Metrics
# ============================
def denorm_rho_theta(rho_norm, theta_norm):
    diag = math.sqrt(UNET_W ** 2 + UNET_H ** 2)
    pad_top = (RESIZE_H - diag) / 2.0
    final_rho_idx = rho_norm * (RESIZE_H - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)
    theta_deg = (theta_norm * 180.0) % 180.0
    return rho_real, theta_deg


def angular_diff_deg(a, b, period=180.0):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def pct_le(arr, thr):
    if len(arr) == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


def summarize(name, arr):
    if len(arr) == 0:
        return f"{name}: N=0"
    return f"{name}: mean={np.mean(arr):.4f}, median={np.median(arr):.4f}, p95={np.percentile(arr, 95):.4f}"


# ============================
# Main
# ============================
def main():
    print("=" * 60)
    print("Evaluate Fusion-CNN on Buoy Test Set (In-Domain)")
    print("=" * 60)

    # Check files
    if not CACHE_DIR.exists():
        print(f"[Error] Cache not found: {CACHE_DIR}")
        return 1

    if not WEIGHTS_PATH.exists():
        print(f"[Error] Weights not found: {WEIGHTS_PATH}")
        return 1

    # Load data
    test_indices = load_split_indices(SPLIT_DIR)
    print(f"[Test] {len(test_indices)} samples")

    ds = TestCacheDataset(str(CACHE_DIR), test_indices)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=DEVICE.startswith("cuda"))

    # Load model
    model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=180).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # Evaluate
    g_rho_err_orig = []
    g_theta_err = []
    rows = []

    with torch.no_grad():
        for xb, yb, idxb, names, orig_ws, orig_hs in dl:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            gt = yb.numpy()
            idxb = idxb.numpy()
            orig_ws = orig_ws.numpy()
            orig_hs = orig_hs.numpy()

            for i in range(len(idxb)):
                rho_p, th_p = denorm_rho_theta(pred[i, 0], pred[i, 1])
                rho_g, th_g = denorm_rho_theta(gt[i, 0], gt[i, 1])

                e_rho = abs(rho_p - rho_g)
                e_theta = angular_diff_deg(th_p, th_g)

                # Scale to original size
                scale = orig_ws[i] / UNET_W
                e_rho_orig = e_rho * scale

                g_rho_err_orig.append(e_rho_orig)
                g_theta_err.append(e_theta)

                rows.append({
                    "idx": int(idxb[i]),
                    "img_name": names[i],
                    "orig_w": int(orig_ws[i]),
                    "orig_h": int(orig_hs[i]),
                    "rho_err_px_orig": float(e_rho_orig),
                    "theta_err_deg": float(e_theta),
                })

    g_rho_err_orig = np.array(g_rho_err_orig)
    g_theta_err = np.array(g_theta_err)

    # Print results
    print("\n[Results - 原图尺寸]")
    print(summarize("Rho error (px)", g_rho_err_orig))
    print(summarize("Theta error (deg)", g_theta_err))
    print("---- 阈值统计 ----")
    print(f"rho <= 5px: {pct_le(g_rho_err_orig, 5):.2f}% | <=10px: {pct_le(g_rho_err_orig, 10):.2f}% | <=20px: {pct_le(g_rho_err_orig, 20):.2f}%")
    print(f"theta <= 1°: {pct_le(g_theta_err, 1):.2f}% | <=2°: {pct_le(g_theta_err, 2):.2f}% | <=5°: {pct_le(g_theta_err, 5):.2f}%")

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Saved] {OUT_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
