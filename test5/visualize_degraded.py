# -*- coding: utf-8 -*-
"""
visualize_degraded.py

Generate visualization images comparing clean vs degraded predictions.

Creates side-by-side comparisons showing:
  - Clean image with prediction
  - Degraded image with prediction
  - Error metrics overlay

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================
# PyCharm 配置�?(在这里修�?
# ============================
N_SAMPLES = 5                   # 每种退化可视化样本�?
SELECTED_DEGRADATIONS = [       # 选择要可视化的退化类型（海洋特有�?
    "rain_medium",
    "glare_heavy",
    "jpeg_q10",
    "lowres_0.25x",
    "fog_0.5",
]
SEED = 42
# 选择数据�? "musid", "smd", "buoy"
DATASET = "musid"
# ============================

# 命令行参数覆�?(支持 run_experiment5.py 一键调�?
if "--dataset" in sys.argv:
    _idx = sys.argv.index("--dataset")
    if _idx + 1 < len(sys.argv):
        DATASET = sys.argv[_idx + 1]

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
test1_DIR = PROJECT_ROOT / "test1"

# 数据集配�?
DATASET_CONFIGS = {
    "musid": {
        "degraded_img_dir": TEST5_DIR / "degraded_images",
        "cache_root": TEST5_DIR / "FusionCache_Degraded",
        "out_dir": TEST5_DIR / "visualization",
        "cnn_weights": "weights/best_fusion_cnn_1024x576.pth",
    },
    "smd": {
        "degraded_img_dir": TEST5_DIR / "degraded_images_smd",
        "cache_root": TEST5_DIR / "FusionCache_Degraded_SMD",
        "out_dir": TEST5_DIR / "visualization_smd",
        "cnn_weights": "test1/weights/best_fusion_cnn_smd.pth",
    },
    "buoy": {
        "degraded_img_dir": TEST5_DIR / "degraded_images_buoy",
        "cache_root": TEST5_DIR / "FusionCache_Degraded_Buoy",
        "out_dir": TEST5_DIR / "visualization_buoy",
        "cnn_weights": "test1/weights/best_fusion_cnn_buoy.pth",
    },
}

# Legacy compatibility
DEGRADED_IMG_DIR = TEST5_DIR / "degraded_images"
CACHE_ROOT = TEST5_DIR / "FusionCache_Degraded"
OUT_DIR = TEST5_DIR / "visualization"

# Image size
UNET_W, UNET_H = 1024, 576
RESIZE_H = 2240

# Colors (BGR)
COLOR_GT = (0, 255, 0)       # Green
COLOR_PRED = (0, 0, 255)     # Red
COLOR_TEXT = (255, 255, 255) # White
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def denorm_rho_theta(rho_norm, theta_norm, w=UNET_W, h=UNET_H, resize_h=RESIZE_H):
    """Convert normalized (rho, theta) to real values."""
    diag = math.sqrt(w * w + h * h)
    pad_top = (resize_h - diag) / 2.0
    
    rho_idx = rho_norm * (resize_h - 1.0)
    rho_real = rho_idx - pad_top - (diag / 2.0)
    
    theta_deg = (theta_norm * 180.0) % 180.0
    return rho_real, theta_deg


def rho_theta_to_line(rho, theta_deg, w, h):
    """Convert (rho, theta) to line endpoints."""
    theta_rad = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    
    cx, cy = w / 2.0, h / 2.0
    x0 = cx + rho * cos_t
    y0 = cy + rho * sin_t
    
    length = max(w, h) * 2
    dx, dy = -sin_t * length, cos_t * length
    
    x1, y1 = int(x0 - dx), int(y0 - dy)
    x2, y2 = int(x0 + dx), int(y0 + dy)
    
    return (x1, y1), (x2, y2)


def draw_line_on_image(img, rho, theta_deg, color, thickness=2):
    """Draw horizon line on image."""
    h, w = img.shape[:2]
    pt1, pt2 = rho_theta_to_line(rho, theta_deg, w, h)
    cv2.line(img, pt1, pt2, color, thickness)
    return img


def create_comparison(clean_img, degraded_img, pred_clean, pred_deg, gt, deg_name):
    """Create side-by-side comparison image."""
    h, w = clean_img.shape[:2]
    
    # Resize both to same size
    clean_resized = cv2.resize(clean_img, (UNET_W, UNET_H))
    deg_resized = cv2.resize(degraded_img, (UNET_W, UNET_H))
    
    # Draw lines
    rho_gt, theta_gt = denorm_rho_theta(gt[0], gt[1])
    rho_clean, theta_clean = denorm_rho_theta(pred_clean[0], pred_clean[1])
    rho_deg, theta_deg = denorm_rho_theta(pred_deg[0], pred_deg[1])
    
    # Clean image
    draw_line_on_image(clean_resized, rho_gt, theta_gt, COLOR_GT, LINE_THICKNESS)
    draw_line_on_image(clean_resized, rho_clean, theta_clean, COLOR_PRED, LINE_THICKNESS)
    
    # Degraded image
    draw_line_on_image(deg_resized, rho_gt, theta_gt, COLOR_GT, LINE_THICKNESS)
    draw_line_on_image(deg_resized, rho_deg, theta_deg, COLOR_PRED, LINE_THICKNESS)
    
    # Compute errors
    scale = 1920 / UNET_W  # Original scale
    rho_err_clean = abs(rho_clean - rho_gt) * scale
    rho_err_deg = abs(rho_deg - rho_gt) * scale
    
    # Add labels
    cv2.putText(clean_resized, "Clean", (10, 25), FONT, FONT_SCALE, COLOR_TEXT, 1)
    cv2.putText(clean_resized, f"Rho Err: {rho_err_clean:.1f}px", (10, 50), FONT, FONT_SCALE, COLOR_TEXT, 1)
    
    cv2.putText(deg_resized, deg_name, (10, 25), FONT, FONT_SCALE, COLOR_TEXT, 1)
    cv2.putText(deg_resized, f"Rho Err: {rho_err_deg:.1f}px", (10, 50), FONT, FONT_SCALE, COLOR_TEXT, 1)
    
    # Legend
    cv2.putText(clean_resized, "GT", (UNET_W - 60, 25), FONT, FONT_SCALE, COLOR_GT, 2)
    cv2.putText(clean_resized, "Pred", (UNET_W - 60, 50), FONT, FONT_SCALE, COLOR_PRED, 2)
    
    # Combine side by side
    combined = np.hstack([clean_resized, deg_resized])
    
    return combined


def load_cache_data(cache_dir):
    """Load all cache data from a directory."""
    data = {}
    for f in Path(cache_dir).glob("*.npy"):
        item = np.load(str(f), allow_pickle=True).item()
        img_name = item.get("img_name", "")
        if img_name:
            data[img_name] = {
                "label": item["label"],
                "input": item["input"],
            }
    return data


def main():
    np.random.seed(SEED)
    
    cfg = DATASET_CONFIGS[DATASET]
    degraded_img_dir = cfg["degraded_img_dir"]
    cache_root = cfg["cache_root"]
    out_dir = cfg["out_dir"]
    
    print("=" * 60)
    print("Experiment 5: Visualize Degraded Predictions")
    print(f"Dataset: {DATASET.upper()}")
    print("=" * 60)
    
    ensure_dir(out_dir)
    
    # Check directories exist
    if not degraded_img_dir.exists():
        print(f"[Error] Degraded images not found: {degraded_img_dir}")
        sys.exit(1)
    
    if not cache_root.exists():
        print(f"[Error] Cache not found: {cache_root}")
        sys.exit(1)
    
    # Load clean cache for GT and clean predictions
    clean_cache_dir = cache_root / "clean"
    if not clean_cache_dir.exists():
        print(f"[Error] Clean cache not found: {clean_cache_dir}")
        sys.exit(1)
    
    clean_data = load_cache_data(clean_cache_dir)
    img_names = list(clean_data.keys())
    
    print(f"\n[Load] {len(img_names)} images from clean cache")
    
    # Load model for inference (or use cached predictions)
    # For simplicity, we'll run inference here
    from cnn_model import HorizonResNet
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=180).to(device)
    weights_path = PROJECT_ROOT / cfg["cnn_weights"]
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    
    # Process each degradation type
    for deg_name in SELECTED_DEGRADATIONS:
        print(f"\n[Visualize] {deg_name}...")
        
        deg_cache_dir = cache_root / deg_name
        deg_img_dir = degraded_img_dir / deg_name
        clean_img_dir = degraded_img_dir / "clean"
        
        if not deg_cache_dir.exists():
            print(f"  [Skip] Cache not found: {deg_cache_dir}")
            continue
        
        deg_data = load_cache_data(deg_cache_dir)
        
        # Select random samples
        common_imgs = list(set(clean_data.keys()) & set(deg_data.keys()))
        n_samples = min(N_SAMPLES, len(common_imgs))
        selected = np.random.choice(common_imgs, n_samples, replace=False)
        
        out_subdir = out_dir / deg_name
        ensure_dir(out_subdir)
        
        for i, img_name in enumerate(selected):
            clean_img_path = clean_img_dir / img_name
            deg_img_path = deg_img_dir / img_name
            
            if not clean_img_path.exists() or not deg_img_path.exists():
                continue
            
            clean_img = cv2.imread(str(clean_img_path))
            deg_img = cv2.imread(str(deg_img_path))
            
            # Get predictions
            with torch.no_grad():
                clean_input = torch.from_numpy(clean_data[img_name]["input"]).unsqueeze(0).to(device)
                deg_input = torch.from_numpy(deg_data[img_name]["input"]).unsqueeze(0).to(device)
                
                pred_clean = model(clean_input).cpu().numpy()[0]
                pred_deg = model(deg_input).cpu().numpy()[0]
            
            gt = clean_data[img_name]["label"]
            
            # Create comparison
            comparison = create_comparison(clean_img, deg_img, pred_clean, pred_deg, gt, deg_name)
            
            out_path = out_subdir / f"{i:03d}_{img_name}"
            cv2.imwrite(str(out_path), comparison)
        
        print(f"  -> Saved {n_samples} comparisons to {out_subdir}")
    
    print("\n" + "=" * 60)
    print("[Done] Visualizations saved to:")
    print(f"  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # CLI override for dataset
    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            DATASET = sys.argv[idx + 1].lower()
            if DATASET not in DATASET_CONFIGS:
                print(f"[Error] Unknown dataset: {DATASET}")
                print(f"  Available: {list(DATASET_CONFIGS.keys())}")
                sys.exit(1)
    main()
