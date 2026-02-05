# -*- coding: utf-8 -*-
"""
Evaluate Fusion-CNN on SMD test set (Experiment 6: In-Domain).

与主评估代码对齐的指标：
  - rho error (UNet空间 + 原图尺寸)
  - theta error (度数)
  - 阈值统计 (参考 evaluate_full_pipeline.py)

Inputs:
  - test6/FusionCache_SMD/test/
  - test6/weights/best_fusion_cnn_smd.pth
  - test6/splits_smd/test_indices.npy

Outputs:
  - test6/eval_smd_indomain.csv
  - 终端输出统计信息

PyCharm: 直接运行此文件
"""

import os
import sys
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_model import HorizonResNet  # noqa: E402


# ============================
# PyCharm 配置区
# ============================
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 40
# ============================

# Paths
TEST6_DIR = PROJECT_ROOT / "test6"
CACHE_DIR = TEST6_DIR / "FusionCache_SMD" / "test"
SPLIT_DIR = TEST6_DIR / "splits_smd"
WEIGHTS_PATH = TEST6_DIR / "weights" / "best_fusion_cnn_smd.pth"
OUT_CSV = TEST6_DIR / "eval_smd_indomain.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Denorm config (与主评估代码一致)
UNET_W, UNET_H = 1024, 576
RESIZE_H = 2240
ANGLE_RANGE_DEG = 180.0


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
        img_name = str(data.get("img_name", f"{idx}.jpg"))
        orig_w = int(data.get("orig_w", 1920))
        orig_h = int(data.get("orig_h", 1080))
        return x, y, idx, img_name, orig_w, orig_h


def load_split_indices(split_dir):
    path = os.path.join(split_dir, "test_indices.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test indices not found: {path}")
    return np.load(path).astype(np.int64).tolist()


# ============================
# Metrics (与主评估代码对齐)
# ============================
def denorm_rho_theta(rho_norm, theta_norm):
    """将归一化的 (rho, theta) 转换为实际值"""
    diag = math.sqrt(UNET_W ** 2 + UNET_H ** 2)
    pad_top = (RESIZE_H - diag) / 2.0
    
    final_rho_idx = rho_norm * (RESIZE_H - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)
    theta_deg = (theta_norm * ANGLE_RANGE_DEG) % ANGLE_RANGE_DEG
    
    return rho_real, theta_deg


def angular_diff_deg(a, b, period=180.0):
    """计算周期性角度差（wrap-aware）"""
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def pct_le(arr, thr):
    """计算小于等于阈值的样本百分比"""
    if len(arr) == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


def summarize(name, arr):
    """统计摘要（与主评估代码对齐）"""
    if len(arr) == 0:
        return f"{name}: (empty)"
    return (
        f"{name}: mean={np.mean(arr):.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}, max={np.max(arr):.4f}"
    )


# ============================
# Main
# ============================
def main():
    # Set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 70)
    print("Evaluate Fusion-CNN on SMD Test Set (In-Domain)")
    print("=" * 70)

    # Check files
    if not CACHE_DIR.exists():
        print(f"[Error] Cache not found: {CACHE_DIR}")
        print("  Please run make_fusion_cache_smd_train.py first.")
        return 1

    if not WEIGHTS_PATH.exists():
        print(f"[Error] Weights not found: {WEIGHTS_PATH}")
        print("  Please run train_fusion_cnn_smd.py first.")
        return 1

    # Load data
    test_indices = load_split_indices(SPLIT_DIR)
    print(f"[Test] {len(test_indices)} samples")

    ds = TestCacheDataset(str(CACHE_DIR), test_indices)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=DEVICE.startswith("cuda"))

    # Load model
    model = HorizonResNet(in_channels=4).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded from {WEIGHTS_PATH}")
    print(f"[Device] {DEVICE}")

    # Evaluate
    rho_err_unet = []
    rho_err_orig = []
    theta_err = []
    rows = []

    print("\n[Evaluating...]")
    with torch.no_grad():
        for xb, yb, idxb, names, orig_ws, orig_hs in dl:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            gt = yb.numpy()
            idxb = idxb.numpy()
            orig_ws = orig_ws.numpy()
            orig_hs = orig_hs.numpy()

            for i in range(len(idxb)):
                # Denormalize
                rho_p, th_p = denorm_rho_theta(pred[i, 0], pred[i, 1])
                rho_g, th_g = denorm_rho_theta(gt[i, 0], gt[i, 1])

                # Errors in UNet space
                e_rho_unet = abs(rho_p - rho_g)
                e_theta = angular_diff_deg(th_p, th_g)

                # Scale to original size
                scale_x = orig_ws[i] / UNET_W
                e_rho_orig = e_rho_unet * scale_x

                rho_err_unet.append(e_rho_unet)
                rho_err_orig.append(e_rho_orig)
                theta_err.append(e_theta)

                rows.append({
                    "idx": int(idxb[i]),
                    "img_name": names[i],
                    "orig_w": int(orig_ws[i]),
                    "orig_h": int(orig_hs[i]),
                    "rho_err_px_unet": float(e_rho_unet),
                    "rho_err_px_orig": float(e_rho_orig),
                    "theta_err_deg": float(e_theta),
                })

    rho_err_unet = np.array(rho_err_unet)
    rho_err_orig = np.array(rho_err_orig)
    theta_err = np.array(theta_err)

    # Print results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    print("\n[UNet Space (1024x576)]")
    print(summarize("Rho error (px)", rho_err_unet))
    print(summarize("Theta error (deg)", theta_err))
    
    print("\n[Original Image Scale]")
    print(summarize("Rho error (px)", rho_err_orig))
    
    print("\n[Threshold Statistics]")
    print(f"Rho (orig): <=5px: {pct_le(rho_err_orig, 5):.2f}% | <=10px: {pct_le(rho_err_orig, 10):.2f}% | <=20px: {pct_le(rho_err_orig, 20):.2f}%")
    print(f"Theta:      <=1°: {pct_le(theta_err, 1):.2f}% | <=2°: {pct_le(theta_err, 2):.2f}% | <=5°: {pct_le(theta_err, 5):.2f}%")

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n[Saved] {OUT_CSV}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
