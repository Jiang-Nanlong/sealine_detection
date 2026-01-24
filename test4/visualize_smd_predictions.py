# -*- coding: utf-8 -*-
"""
visualize_smd_predictions.py

Visualize horizon detection predictions on SMD dataset for Experiment 4.

This script generates visualization images showing:
  - Original input image
  - Predicted horizon line (red) vs Ground truth (green)
  - Per-image error metrics overlay

Useful for:
  1. Qualitative analysis in thesis
  2. Identifying failure cases
  3. Comparing domain performance visually

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
# PyCharm 配置区 (在这里修改)
# ============================
N_SAMPLES = 20                  # 可视化样本数量
MODE = "random"                 # 模式: "random" / "best" / "worst" / "per_domain"
SEED = 42                       # 随机种子
# ============================

# ----------------------------
# Config
# ----------------------------
TEST4_DIR = PROJECT_ROOT / "test4"
FRAMES_DIR = TEST4_DIR / "smd_frames"
CACHE_DIR = TEST4_DIR / "FusionCache_SMD_1024x576" / "test"
EVAL_CSV = TEST4_DIR / "eval_smd_test_per_sample.csv"
GT_CSV = TEST4_DIR / "SMD_GroundTruth.csv"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_fusion_cnn_1024x576.pth"
OUT_DIR = TEST4_DIR / "visualization"

# Image size (UNet space)
UNET_W, UNET_H = 1024, 576
RESIZE_H = 2240

# Colors (BGR)
COLOR_GT = (0, 255, 0)       # Green for ground truth
COLOR_PRED = (0, 0, 255)     # Red for prediction
COLOR_TEXT = (255, 255, 255) # White for text
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def denorm_rho_theta(rho_norm, theta_norm, w=UNET_W, h=UNET_H, resize_h=RESIZE_H):
    """Convert normalized (rho, theta) to real values in UNet space."""
    diag = math.sqrt(w * w + h * h)
    pad_top = (resize_h - diag) / 2.0
    
    final_rho_idx = rho_norm * (resize_h - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)
    
    theta_deg = (theta_norm * 180.0) % 180.0
    return rho_real, theta_deg


def draw_horizon_line(img, rho, theta_deg, color, thickness=LINE_THICKNESS, w=UNET_W, h=UNET_H):
    """Draw horizon line on image given rho and theta."""
    theta_rad = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    cx, cy = w / 2.0, h / 2.0
    
    # Compute line endpoints at x=0 and x=w-1
    pts = []
    
    # x = 0
    if abs(sin_t) > 1e-8:
        y = cy + (rho - ((0 - cx) * cos_t)) / sin_t
        if -50 <= y <= h + 50:
            pts.append((0, int(y)))
    
    # x = w - 1
    if abs(sin_t) > 1e-8:
        x = w - 1
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if -50 <= y <= h + 50:
            pts.append((int(x), int(y)))
    
    # y = 0
    if abs(cos_t) > 1e-8:
        x = cx + (rho - ((0 - cy) * sin_t)) / cos_t
        if -50 <= x <= w + 50:
            pts.append((int(x), 0))
    
    # y = h - 1
    if abs(cos_t) > 1e-8:
        x = cx + (rho - ((h - 1 - cy) * sin_t)) / cos_t
        if -50 <= x <= w + 50:
            pts.append((int(x), int(h - 1)))
    
    if len(pts) >= 2:
        # Find two most distant points
        best_pair = (pts[0], pts[1])
        best_dist = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (pts[i], pts[j])
        
        cv2.line(img, best_pair[0], best_pair[1], color, thickness)
    
    return img


def create_visualization(img_path, rho_pred, theta_pred, rho_gt, theta_gt, metrics, domain):
    """Create a single visualization image with GT and prediction."""
    # Read original image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Resize to UNet space
    img = cv2.resize(img, (UNET_W, UNET_H))
    
    # Draw lines
    img = draw_horizon_line(img, rho_gt, theta_gt, COLOR_GT, thickness=LINE_THICKNESS + 1)
    img = draw_horizon_line(img, rho_pred, theta_pred, COLOR_PRED, thickness=LINE_THICKNESS)
    
    # Add metrics overlay
    rho_err = metrics.get("rho_err_px_orig", 0)
    theta_err = metrics.get("theta_err_deg", 0)
    
    # Background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (350, 95), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    # Text
    cv2.putText(img, f"Domain: {domain}", (10, 25), FONT, FONT_SCALE, COLOR_TEXT, 1)
    cv2.putText(img, f"Rho Error: {rho_err:.2f} px", (10, 50), FONT, FONT_SCALE, COLOR_TEXT, 1)
    cv2.putText(img, f"Theta Error: {theta_err:.3f} deg", (10, 75), FONT, FONT_SCALE, COLOR_TEXT, 1)
    
    # Legend
    cv2.putText(img, "GT", (UNET_W - 80, 25), FONT, FONT_SCALE, COLOR_GT, 2)
    cv2.putText(img, "Pred", (UNET_W - 80, 50), FONT, FONT_SCALE, COLOR_PRED, 2)
    
    return img


def main():
    """
    PyCharm: 修改文件顶部的配置区即可:
        N_SAMPLES = 20       # 可视化样本数量
        MODE = "random"      # "random" / "best" / "worst" / "per_domain"
        SEED = 42            # 随机种子
    """
    n_samples = N_SAMPLES
    mode = MODE
    seed = SEED
    out_dir = OUT_DIR
    
    ensure_dir(out_dir)
    np.random.seed(seed)
    
    # Check files exist
    if not EVAL_CSV.exists():
        print(f"[Error] Evaluation CSV not found: {EVAL_CSV}")
        print("Please run evaluate_fusion_cnn_smd.py first")
        sys.exit(1)
    
    if not GT_CSV.exists():
        print(f"[Error] Ground truth CSV not found: {GT_CSV}")
        sys.exit(1)
    
    # Load data
    print(f"[Load] Evaluation results: {EVAL_CSV}")
    eval_df = pd.read_csv(EVAL_CSV)
    
    print(f"[Load] Ground truth: {GT_CSV}")
    gt_df = pd.read_csv(GT_CSV)
    
    # Select samples based on mode
    if mode == "worst":
        # Sort by rho error descending
        eval_df = eval_df.sort_values("rho_err_px_orig", ascending=False)
        selected = eval_df.head(n_samples)
        suffix = "worst"
    elif mode == "best":
        # Sort by rho error ascending
        eval_df = eval_df.sort_values("rho_err_px_orig", ascending=True)
        selected = eval_df.head(n_samples)
        suffix = "best"
    elif mode == "per_domain":
        # Select n_samples from each domain
        selected_list = []
        for domain in ["NIR", "VIS_Onboard", "VIS_Onshore"]:
            domain_df = eval_df[eval_df["domain"] == domain]
            n = min(n_samples, len(domain_df))
            selected_list.append(domain_df.sample(n=n, random_state=seed))
        selected = pd.concat(selected_list)
        suffix = "per_domain"
    else:
        # Random
        n = min(n_samples, len(eval_df))
        selected = eval_df.sample(n=n, random_state=seed)
        suffix = "random"
    
    print(f"[Select] {len(selected)} samples ({mode} mode)")
    
    # Generate visualizations
    out_subdir = Path(out_dir) / suffix
    ensure_dir(out_subdir)
    
    success_count = 0
    for i, row in enumerate(selected.itertuples()):
        img_name = row.img_name
        img_path = FRAMES_DIR / img_name
        
        if not img_path.exists():
            print(f"  [Skip] Image not found: {img_path}")
            continue
        
        # Get prediction (from normalized values)
        rho_pred, theta_pred = denorm_rho_theta(row.rho_pred_norm, row.theta_pred_norm)
        rho_gt, theta_gt = denorm_rho_theta(row.rho_gt_norm, row.theta_gt_norm)
        
        metrics = {
            "rho_err_px_orig": row.rho_err_px_orig,
            "theta_err_deg": row.theta_err_deg,
        }
        
        vis = create_visualization(img_path, rho_pred, theta_pred, rho_gt, theta_gt, metrics, row.domain)
        
        if vis is not None:
            out_path = out_subdir / f"{i:03d}_{img_name}"
            cv2.imwrite(str(out_path), vis)
            success_count += 1
    
    print(f"[Done] Saved {success_count} visualizations to: {out_subdir}")
    
    # Create grid visualization for thesis
    if success_count > 0 and mode in ["worst", "best"]:
        print(f"[Grid] Creating grid visualization...")
        grid_images = []
        for img_file in sorted(out_subdir.glob("*.jpg"))[:min(9, success_count)]:
            img = cv2.imread(str(img_file))
            if img is not None:
                grid_images.append(img)
        
        if len(grid_images) >= 4:
            # Create 2x2 or 3x3 grid
            n_cols = 3 if len(grid_images) >= 9 else 2
            n_rows = (len(grid_images) + n_cols - 1) // n_cols
            
            # Resize all to same size
            h, w = grid_images[0].shape[:2]
            
            # Pad with blank if needed
            while len(grid_images) < n_rows * n_cols:
                grid_images.append(np.zeros((h, w, 3), dtype=np.uint8))
            
            rows = []
            for r in range(n_rows):
                row_imgs = grid_images[r * n_cols:(r + 1) * n_cols]
                rows.append(np.hstack(row_imgs))
            
            grid = np.vstack(rows)
            grid_path = out_subdir / f"grid_{suffix}.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"  -> Grid saved: {grid_path}")


if __name__ == "__main__":
    main()
