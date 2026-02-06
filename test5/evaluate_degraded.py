# -*- coding: utf-8 -*-
"""
evaluate_degraded.py

Evaluate Fusion-CNN on degraded MU-SID test images.

This script evaluates model performance across all degradation types
and compares against clean (baseline) performance.

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
import math
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_model import HorizonResNet

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 0
# 选择数据集: "musid", "smd", "buoy"
DATASET = "musid"
# ============================

# 命令行参数覆盖 (支持 run_experiment5.py 一键调用)
if "--dataset" in sys.argv:
    _idx = sys.argv.index("--dataset")
    if _idx + 1 < len(sys.argv):
        DATASET = sys.argv[_idx + 1]

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
TEST6_DIR = PROJECT_ROOT / "test6"

# 数据集配置
DATASET_CONFIGS = {
    "musid": {
        "cache_root": TEST5_DIR / "FusionCache_Degraded",
        "weights_path": PROJECT_ROOT / "weights" / "best_fusion_cnn_1024x576.pth",
        "out_dir": TEST5_DIR / "eval_results",
        "orig_w": 1920,
        "orig_h": 1080,
    },
    "smd": {
        "cache_root": TEST5_DIR / "FusionCache_Degraded_SMD",
        "weights_path": TEST6_DIR / "weights" / "best_fusion_cnn_smd.pth",
        "out_dir": TEST5_DIR / "eval_results_smd",
        "orig_w": 1920,
        "orig_h": 1080,
    },
    "buoy": {
        "cache_root": TEST5_DIR / "FusionCache_Degraded_Buoy",
        "weights_path": TEST6_DIR / "weights" / "best_fusion_cnn_buoy.pth",
        "out_dir": TEST5_DIR / "eval_results_buoy",
        "orig_w": 800,
        "orig_h": 600,
    },
}

# Legacy compatibility
CACHE_ROOT = TEST5_DIR / "FusionCache_Degraded"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_fusion_cnn_1024x576.pth"
OUT_DIR = TEST5_DIR / "eval_results"

# Denorm config
UNET_W = 1024
UNET_H = 576
RESIZE_H = 2240
ORIG_W = 1920
ORIG_H = 1080


@dataclass
class DenormConfig:
    unet_w: int = UNET_W
    unet_h: int = UNET_H
    resize_h: int = RESIZE_H
    orig_w: int = ORIG_W
    orig_h: int = ORIG_H


class CacheDataset(Dataset):
    """Load cached .npy files."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.npy"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]), allow_pickle=True).item()
        x = torch.from_numpy(data["input"])
        y = torch.from_numpy(data["label"])
        img_name = data.get("img_name", "")
        return x, y, idx, img_name


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def denorm_rho_theta(rho_norm, theta_norm, cfg: DenormConfig):
    """Convert normalized (rho, theta) to real values."""
    diag = math.sqrt(cfg.unet_w ** 2 + cfg.unet_h ** 2)
    pad_top = (cfg.resize_h - diag) / 2.0
    
    rho_idx = rho_norm * (cfg.resize_h - 1.0)
    rho_real = rho_idx - pad_top - (diag / 2.0)
    
    theta_deg = (theta_norm * 180.0) % 180.0
    return rho_real, theta_deg


def angular_diff_deg(a, b, period=180.0):
    """Compute angular difference with wrap-around."""
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def evaluate_single(model, dataloader, cfg: DenormConfig, device: str):
    """Evaluate model on a single degradation type."""
    model.eval()
    
    rho_errors = []
    theta_errors = []
    scale = cfg.orig_w / cfg.unet_w
    
    with torch.no_grad():
        for xb, yb, _, _ in dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            pred = model(xb)
            pred_np = pred.cpu().numpy()
            gt_np = yb.cpu().numpy()
            
            rho_p, th_p = denorm_rho_theta(pred_np[:, 0], pred_np[:, 1], cfg)
            rho_g, th_g = denorm_rho_theta(gt_np[:, 0], gt_np[:, 1], cfg)
            
            e_rho = np.abs(rho_p - rho_g) * scale  # In original pixel space
            e_theta = angular_diff_deg(th_p, th_g)
            
            rho_errors.extend(e_rho.tolist())
            theta_errors.extend(e_theta.tolist())
    
    rho_errors = np.array(rho_errors)
    theta_errors = np.array(theta_errors)
    
    return {
        "n": len(rho_errors),
        "rho_mean": float(np.mean(rho_errors)),
        "rho_median": float(np.median(rho_errors)),
        "rho_p95": float(np.percentile(rho_errors, 95)),
        "rho_le_5": float(np.mean(rho_errors <= 5) * 100),
        "rho_le_10": float(np.mean(rho_errors <= 10) * 100),
        "rho_le_20": float(np.mean(rho_errors <= 20) * 100),
        "theta_mean": float(np.mean(theta_errors)),
        "theta_median": float(np.median(theta_errors)),
        "theta_le_1": float(np.mean(theta_errors <= 1) * 100),
        "theta_le_2": float(np.mean(theta_errors <= 2) * 100),
    }


def main():
    cfg_ds = DATASET_CONFIGS[DATASET]
    cache_root = cfg_ds["cache_root"]
    weights_path = cfg_ds["weights_path"]
    out_dir = cfg_ds["out_dir"]
    
    print("=" * 60)
    print("Experiment 5: Evaluate Degraded Images")
    print(f"Dataset: {DATASET.upper()}")
    print("=" * 60)
    
    ensure_dir(out_dir)
    cfg = DenormConfig(
        unet_w=UNET_W, unet_h=UNET_H, resize_h=RESIZE_H,
        orig_w=cfg_ds["orig_w"], orig_h=cfg_ds["orig_h"]
    )
    
    # Load model
    print(f"\n[Device] {DEVICE}")
    print(f"[Load] Fusion-CNN: {weights_path}")
    
    model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=180).to(DEVICE)
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    
    # Get degradation folders
    if not cache_root.exists():
        print(f"[Error] Cache not found: {cache_root}")
        print("Please run make_fusion_cache_degraded.py first")
        sys.exit(1)
    
    deg_folders = sorted([d for d in cache_root.iterdir() if d.is_dir()])
    print(f"\n[Evaluate] {len(deg_folders)} degradation types")
    
    results = []
    clean_metrics = None
    degraded_metrics_list = []
    
    for deg_folder in deg_folders:
        deg_name = deg_folder.name
        
        ds = CacheDataset(str(deg_folder))
        if len(ds) == 0:
            print(f"  [Skip] {deg_name}: no cache files")
            continue
        
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=DEVICE.startswith("cuda"),
        )
        
        metrics = evaluate_single(model, dl, cfg, DEVICE)
        metrics["degradation"] = deg_name
        results.append(metrics)
        
        if deg_name == "clean":
            clean_metrics = metrics
        else:
            degraded_metrics_list.append(metrics)
        
        print(f"\n[{deg_name}] N={metrics['n']}")
        print(f"  ρ: mean={metrics['rho_mean']:.2f}px, ≤5px={metrics['rho_le_5']:.1f}%, ≤10px={metrics['rho_le_10']:.1f}%, ≤20px={metrics['rho_le_20']:.1f}%")
        print(f"  θ: mean={metrics['theta_mean']:.3f}°, ≤1°={metrics['theta_le_1']:.1f}%, ≤2°={metrics['theta_le_2']:.1f}%")
    
    # Print clean vs degraded comparison
    if clean_metrics and degraded_metrics_list:
        print("\n" + "=" * 60)
        print("[Drop] Clean vs Degraded (Average)")
        print("=" * 60)
        
        avg_rho = np.mean([m["rho_mean"] for m in degraded_metrics_list])
        avg_theta = np.mean([m["theta_mean"] for m in degraded_metrics_list])
        
        rho_c = clean_metrics["rho_mean"]
        theta_c = clean_metrics["theta_mean"]
        
        d_rho = avg_rho - rho_c
        d_theta = avg_theta - theta_c
        pct_rho = (d_rho / rho_c * 100) if rho_c > 0 else 0
        pct_theta = (d_theta / theta_c * 100) if theta_c > 0 else 0
        
        print(f"[Drop] rho_mean: {rho_c:.2f} -> {avg_rho:.2f} ({d_rho:+.2f}, {pct_rho:+.1f}%)")
        print(f"[Drop] theta_mean: {theta_c:.3f} -> {avg_theta:.3f} ({d_theta:+.3f}, {pct_theta:+.1f}%)")
    
    # Save detailed results
    out_csv = out_dir / "degradation_results.csv"
    if results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Saved] Results -> {out_csv}")
    
    print("\n" + "=" * 60)
    print("[Done] Evaluation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
