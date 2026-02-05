# -*- coding: utf-8 -*-
"""
Evaluate Fusion-CNN on Buoy test set (Experiment 6: In-Domain).

海天线统一评价指标:
  - VE (Vertical Error): 中心列垂直误差 mean(abs(dy))
  - SVE: std(dy, ddof=1)
  - AE (Angular Error): 角度误差 mean(abs(da_w))
  - SA: std(da_w, ddof=1)

Inputs:
  - test6/FusionCache_Buoy/test/
  - test6/weights/best_fusion_cnn_buoy.pth
  - test6/splits_buoy/test_indices.npy

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
CACHE_DIR = TEST6_DIR / "FusionCache_Buoy" / "test"
SPLIT_DIR = TEST6_DIR / "splits_buoy"
WEIGHTS_PATH = TEST6_DIR / "weights" / "best_fusion_cnn_buoy.pth"
OUT_CSV = TEST6_DIR / "eval_buoy_indomain.csv"

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
# Metrics (海天线统一评价指标)
# ============================
def denorm_rho_theta(rho_norm, theta_norm):
    """将归一化的 (rho, theta) 转换为实际值"""
    diag = math.sqrt(UNET_W ** 2 + UNET_H ** 2)
    pad_top = (RESIZE_H - diag) / 2.0
    
    final_rho_idx = rho_norm * (RESIZE_H - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)
    theta_deg = (theta_norm * ANGLE_RANGE_DEG) % ANGLE_RANGE_DEG
    
    return rho_real, theta_deg


def edge_y_at_x(rho: float, theta_deg: float, x: float, w: int, h: int) -> float:
    """计算直线在给定x处的y坐标"""
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    if abs(sin_t) < 1e-8:
        return float(cy)
    y = cy + (rho - ((x - cx) * cos_t)) / sin_t
    return float(np.clip(y, 0.0, h - 1.0))


def compute_VE(rho_pred: float, theta_pred_deg: float,
               rho_gt: float, theta_gt_deg: float,
               w: int, h: int) -> float:
    """
    Vertical Error (VE): 在图像中心列 xc = (W-1)/2 处的y值差
    返回带符号的 Δy
    """
    xc = (w - 1.0) / 2.0
    y_pred = edge_y_at_x(rho_pred, theta_pred_deg, xc, w, h)
    y_gt = edge_y_at_x(rho_gt, theta_gt_deg, xc, w, h)
    return float(y_pred - y_gt)


def compute_AE(rho_pred: float, theta_pred_deg: float,
               rho_gt: float, theta_gt_deg: float,
               w: int, h: int) -> float:
    """
    Angular Error (AE): 预测线与GT线的角度差
    使用端点计算角度: α = atan2(y2-y1, x2-x1) * 180/π
    返回带符号的 Δα
    """
    ypl = edge_y_at_x(rho_pred, theta_pred_deg, 0.0, w, h)
    ypr = edge_y_at_x(rho_pred, theta_pred_deg, w - 1.0, w, h)
    alpha_pred = math.degrees(math.atan2(ypr - ypl, (w - 1.0)))

    ygl = edge_y_at_x(rho_gt, theta_gt_deg, 0.0, w, h)
    ygr = edge_y_at_x(rho_gt, theta_gt_deg, w - 1.0, w, h)
    alpha_gt = math.degrees(math.atan2(ygr - ygl, (w - 1.0)))

    return float(alpha_pred - alpha_gt)


def wrap_ae(da: float) -> float:
    """将角度差 wrap 到 [-90, 90]: da_w = ((da + 90) % 180) - 90"""
    return ((da + 90.0) % 180.0) - 90.0


def safe_std(arr: np.ndarray, ddof: int = 1) -> float:
    """安全计算标准差，样本数<=1时返回0.0"""
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if len(valid) <= 1:
        return 0.0
    return float(np.std(valid, ddof=ddof))


def pct_le(arr, thr):
    """计算小于等于阈值的样本百分比"""
    if len(arr) == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


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
    print("Evaluate Fusion-CNN on Buoy Test Set (In-Domain)")
    print("=" * 70)

    # Check files
    if not CACHE_DIR.exists():
        print(f"[Error] Cache not found: {CACHE_DIR}")
        print("  Please run make_fusion_cache_buoy_train.py first.")
        return 1

    if not WEIGHTS_PATH.exists():
        print(f"[Error] Weights not found: {WEIGHTS_PATH}")
        print("  Please run train_fusion_cnn_buoy.py first.")
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
    dy_signed = []  # VE signed
    da_wrapped = []  # AE wrapped signed
    rows = []

    # Scale factors (Buoy: 800x600)
    DEFAULT_ORIG_W, DEFAULT_ORIG_H = 800, 600
    scale_y = DEFAULT_ORIG_H / UNET_H

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

                # VE/AE 计算
                dy = compute_VE(rho_p, th_p, rho_g, th_g, UNET_W, UNET_H)
                da = compute_AE(rho_p, th_p, rho_g, th_g, UNET_W, UNET_H)
                da_w = wrap_ae(da)
                
                dy_signed.append(dy)
                da_wrapped.append(da_w)

                # Scale to original size
                scale_y_sample = orig_hs[i] / UNET_H
                dy_orig = dy * scale_y_sample

                rows.append({
                    "idx": int(idxb[i]),
                    "img_name": names[i],
                    "orig_w": int(orig_ws[i]),
                    "orig_h": int(orig_hs[i]),
                    "dy_signed_unet": float(dy),
                    "dy_abs_unet": float(abs(dy)),
                    "da_wrapped_deg": float(da_w),
                    "da_abs_deg": float(abs(da_w)),
                    "dy_signed_orig": float(dy_orig),
                    "dy_abs_orig": float(abs(dy_orig)),
                })

    dy_signed = np.array(dy_signed, dtype=np.float64)
    da_wrapped = np.array(da_wrapped, dtype=np.float64)

    # 计算统一指标
    ve_abs_unet = np.abs(dy_signed)
    ae_abs = np.abs(da_wrapped)
    ve_abs_orig = ve_abs_unet * scale_y

    VE_mean = float(np.mean(ve_abs_orig))
    SVE = safe_std(dy_signed * scale_y)
    AE_mean = float(np.mean(ae_abs))
    SA = safe_std(da_wrapped)
    VE_p95 = float(np.percentile(ve_abs_orig, 95)) if len(ve_abs_orig) > 0 else 0.0
    AE_p95 = float(np.percentile(ae_abs, 95)) if len(ae_abs) > 0 else 0.0

    # Print results
    print("\n" + "=" * 60)
    print("海天线统一评价指标 (orig-scale, N={:d})".format(len(dy_signed)))
    print("=" * 60)
    
    print(f"\nVE  (px):  {VE_mean:.2f}    SVE: {SVE:.2f}    P95: {VE_p95:.2f}")
    print(f"AE (deg):  {AE_mean:.2f}    SA:  {SA:.2f}    P95: {AE_p95:.2f}")
    
    print("\n" + "=" * 60)
    print("论文表格汇总 (mean ± std)")
    print("=" * 60)
    print(f"VE (px):  {VE_mean:.2f} ± {SVE:.2f}")
    print(f"AE (deg): {AE_mean:.2f} ± {SA:.2f}")
    print("=" * 60)
    
    print("\n---- Hit-Rate (orig-scale) ----")
    print(f"VE <=5px: {pct_le(ve_abs_orig, 5):.2f}% | <=10px: {pct_le(ve_abs_orig, 10):.2f}% | <=20px: {pct_le(ve_abs_orig, 20):.2f}%")
    print(f"AE <=1°: {pct_le(ae_abs, 1):.2f}% | <=2°: {pct_le(ae_abs, 2):.2f}% | <=5°: {pct_le(ae_abs, 5):.2f}%")

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n[Saved] {OUT_CSV}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
