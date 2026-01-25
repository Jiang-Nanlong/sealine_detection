# -*- coding: utf-8 -*-
"""
make_fusion_cache_degraded.py

Build FusionCache for degraded MU-SID test images.

This generates cache files for each degradation type, enabling
fast batch evaluation without re-running UNet/Radon each time.

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.amp as amp
from tqdm import tqdm

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 选择要处理的退化类型，None 表示处理全部
SELECTED_DEGRADATIONS = None  # 或 ["gaussian_noise_25", "low_light_2.0"]
# ============================

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
DEGRADED_IMG_DIR = TEST5_DIR / "degraded_images"
MUSID_GT_CSV = PROJECT_ROOT / "Hashmani's Dataset" / "GroundTruth.csv"
SPLITS_DIR = PROJECT_ROOT / "splits_musid"

CACHE_ROOT = TEST5_DIR / "FusionCache_Degraded"

# MU-SID trained weights
RGHNET_CKPT = str(PROJECT_ROOT / "weights" / "rghnet_best_c2.pth")
DCE_WEIGHTS = str(PROJECT_ROOT / "weights" / "Epoch99.pth")

# Image sizes
UNET_IN_W = 1024
UNET_IN_H = 576
RESIZE_H = 2240


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_gt_dict():
    """Load ground truth as dict: img_name -> (x1, y1, x2, y2)."""
    df = pd.read_csv(MUSID_GT_CSV)
    gt = {}
    for _, row in df.iterrows():
        gt[row["img_name"]] = (row["x1"], row["y1"], row["x2"], row["y2"])
    return gt


def load_test_split():
    """Load test split image names."""
    test_split_file = SPLITS_DIR / "test.txt"
    with open(test_split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def compute_rho_theta_label(x1, y1, x2, y2, orig_w, orig_h, unet_w, unet_h, resize_h):
    """Convert line endpoints to normalized (rho, theta) in Radon space."""
    import math
    
    # Scale to UNet space
    sx = unet_w / orig_w
    sy = unet_h / orig_h
    x1_u, y1_u = x1 * sx, y1 * sy
    x2_u, y2_u = x2 * sx, y2 * sy
    
    # Compute theta
    dx = x2_u - x1_u
    dy = y2_u - y1_u
    if abs(dx) < 1e-9:
        theta_deg = 90.0
    else:
        theta_deg = math.degrees(math.atan(dy / dx))
    if theta_deg < 0:
        theta_deg += 180.0
    
    # Compute rho
    theta_rad = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    cx, cy = unet_w / 2.0, unet_h / 2.0
    rho = (x1_u - cx) * cos_t + (y1_u - cy) * sin_t
    
    # Normalize
    diag = math.sqrt(unet_w ** 2 + unet_h ** 2)
    pad_top = (resize_h - diag) / 2.0
    rho_idx = rho + (diag / 2.0) + pad_top
    rho_norm = rho_idx / (resize_h - 1.0)
    theta_norm = (theta_deg % 180.0) / 180.0
    
    return float(rho_norm), float(theta_norm)


def main():
    print("=" * 60)
    print("Experiment 5: Build FusionCache for Degraded Images")
    print("=" * 60)
    
    # Load models
    print(f"\n[Device] {DEVICE}")
    print("[Load] UNet (RestorationGuidedHorizonNet)...")
    unet = RestorationGuidedHorizonNet(
        dce_pretrained=DCE_WEIGHTS,
        encoder_name="timm-mobilenetv3_small_100",
    ).to(DEVICE)
    
    ckpt = torch.load(RGHNET_CKPT, map_location=DEVICE, weights_only=False)
    unet.load_state_dict(ckpt, strict=True)
    unet.eval()
    
    print("[Load] Radon Transform...")
    radon = TextureSuppressedMuSCoWERT(
        num_angles=180,
        resize_h=RESIZE_H,
        img_w=UNET_IN_W,
        img_h=UNET_IN_H,
        mid_core_size=9,
        device=DEVICE,
    )
    
    # Load data
    gt_dict = load_gt_dict()
    test_images = load_test_split()
    print(f"[Load] {len(test_images)} test images")
    
    # Get degradation folders
    if not DEGRADED_IMG_DIR.exists():
        print(f"[Error] Degraded images not found: {DEGRADED_IMG_DIR}")
        print("Please run generate_degraded_images.py first")
        sys.exit(1)
    
    deg_folders = sorted([d for d in DEGRADED_IMG_DIR.iterdir() if d.is_dir()])
    
    if SELECTED_DEGRADATIONS:
        deg_folders = [d for d in deg_folders if d.name in SELECTED_DEGRADATIONS]
    
    print(f"[Process] {len(deg_folders)} degradation types")
    
    ensure_dir(CACHE_ROOT)
    
    for deg_folder in deg_folders:
        deg_name = deg_folder.name
        cache_dir = CACHE_ROOT / deg_name
        ensure_dir(cache_dir)
        
        print(f"\n[Cache] {deg_name}...")
        
        idx = 0
        with torch.no_grad(), amp.autocast("cuda", enabled=DEVICE.startswith("cuda")):
            for img_name in tqdm(test_images, desc=deg_name):
                img_path = deg_folder / img_name
                if not img_path.exists():
                    continue
                
                if img_name not in gt_dict:
                    continue
                
                # Load image
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                
                orig_h, orig_w = img_bgr.shape[:2]
                
                # Resize to UNet input
                img_resized = cv2.resize(img_bgr, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # To tensor
                img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)
                
                # UNet forward
                restored, seg_logits = unet(img_t)
                seg_prob = torch.sigmoid(seg_logits)
                
                # Radon transform
                radon_out = radon(restored)  # (1, 3, H, 180)
                
                # Concat: (3 Radon + 1 Seg) = 4 channels
                seg_radon = radon.forward_grayscale(seg_prob)  # (1, 1, H, 180)
                fusion_input = torch.cat([radon_out, seg_radon], dim=1)  # (1, 4, H, 180)
                
                # Compute label
                x1, y1, x2, y2 = gt_dict[img_name]
                rho_norm, theta_norm = compute_rho_theta_label(
                    x1, y1, x2, y2, orig_w, orig_h, UNET_IN_W, UNET_IN_H, RESIZE_H
                )
                
                # Save
                cache_data = {
                    "input": fusion_input.squeeze(0).cpu().numpy().astype(np.float32),
                    "label": np.array([rho_norm, theta_norm], dtype=np.float32),
                    "img_name": img_name,
                    "degradation": deg_name,
                }
                np.save(str(cache_dir / f"{idx}.npy"), cache_data)
                idx += 1
        
        print(f"  -> Saved {idx} cache files to {cache_dir}")
    
    print("\n" + "=" * 60)
    print("[Done] FusionCache for degraded images saved to:")
    print(f"  {CACHE_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
