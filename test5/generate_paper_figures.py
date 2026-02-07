# -*- coding: utf-8 -*-
"""
generate_paper_figures.py

Generate paper figures for Experiment 5: Degradation Robustness.

Figure A: 3x5 qualitative visualization grid
  Rows: MU-SID / SMD / Buoy
  Cols: Clean + fog_0.5 + gaussian_noise_30 + low_light_2.5 + rain_heavy

Figure B: Top-5 hardest degradation bar chart
  Grouped by dataset, sorted by rho_le_10 (lower = harder)

Usage:
  python test5/generate_paper_figures.py [--seed 123] [--device cuda]

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

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
SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 要显示的退化类型（固定顺序）
DEGRADATION_COLS = ["clean", "fog_0.5", "gaussian_noise_30", "low_light_2.5", "rain_heavy"]
DEGRADATION_LABELS = ["Clean", "Fog 0.5", "Noise σ=30", "Low Light γ=2.5", "Rain Heavy"]

# 数据集配置（行顺序）
DATASET_ROWS = ["musid", "smd", "buoy"]
DATASET_LABELS = ["MU-SID", "SMD", "Buoy"]
# ============================

# 命令行参数覆盖
if "--seed" in sys.argv:
    _idx = sys.argv.index("--seed")
    if _idx + 1 < len(sys.argv):
        SEED = int(sys.argv[_idx + 1])

if "--device" in sys.argv:
    _idx = sys.argv.index("--device")
    if _idx + 1 < len(sys.argv):
        DEVICE = sys.argv[_idx + 1]

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
TEST6_DIR = PROJECT_ROOT / "test6"

# UNet input size (used for Radon label generation)
UNET_W = 1024
UNET_H = 576
RESIZE_H = 2240

# Drawing config
COLOR_GT = (0, 255, 0)      # Green for GT
COLOR_PRED = (0, 0, 255)    # Red for Prediction
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 255, 255)

# 数据集配置
DATASET_CONFIGS = {
    "musid": {
        "degraded_img_dir": TEST5_DIR / "degraded_images",
        "cache_root": TEST5_DIR / "FusionCache_Degraded",
        "eval_csv": TEST5_DIR / "eval_results" / "degradation_results.csv",
        "cnn_weights": PROJECT_ROOT / "weights" / "best_fusion_cnn_1024x576.pth",
        "orig_w": 1920,
        "orig_h": 1080,
    },
    "smd": {
        "degraded_img_dir": TEST5_DIR / "degraded_images_smd",
        "cache_root": TEST5_DIR / "FusionCache_Degraded_SMD",
        "eval_csv": TEST5_DIR / "eval_results_smd" / "degradation_results.csv",
        "cnn_weights": TEST6_DIR / "weights" / "best_fusion_cnn_smd.pth",
        "orig_w": 1920,
        "orig_h": 1080,
    },
    "buoy": {
        "degraded_img_dir": TEST5_DIR / "degraded_images_buoy",
        "cache_root": TEST5_DIR / "FusionCache_Degraded_Buoy",
        "eval_csv": TEST5_DIR / "eval_results_buoy" / "degradation_results.csv",
        "cnn_weights": TEST6_DIR / "weights" / "best_fusion_cnn_buoy.pth",
        "orig_w": 800,
        "orig_h": 600,
    },
}

# Output directory
OUT_DIR = TEST5_DIR / "experiment5_results"


@dataclass
class DenormConfig:
    unet_w: int = UNET_W
    unet_h: int = UNET_H
    resize_h: int = RESIZE_H
    orig_w: int = 1920
    orig_h: int = 1080


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ----------------------------
# Utility functions (复用自 visualize_degraded.py 和 evaluate_degraded.py)
# ----------------------------
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
    d = abs(a - b) % period
    return min(d, period - d)


def rho_theta_to_line(rho, theta_deg, w, h):
    """Convert (rho, theta) to line endpoints for drawing."""
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


def draw_horizon_lines(img, rho_gt, theta_gt, rho_pred, theta_pred, cfg: DenormConfig):
    """Draw GT and prediction lines on image (scaled to original size)."""
    h, w = img.shape[:2]
    
    # Scale rho from UNET space to original image space
    scale = cfg.orig_w / cfg.unet_w
    rho_gt_scaled = rho_gt * scale
    rho_pred_scaled = rho_pred * scale
    
    # Draw GT line (green)
    pt1, pt2 = rho_theta_to_line(rho_gt_scaled, theta_gt, w, h)
    cv2.line(img, pt1, pt2, COLOR_GT, LINE_THICKNESS, cv2.LINE_AA)
    
    # Draw prediction line (red)
    pt1, pt2 = rho_theta_to_line(rho_pred_scaled, theta_pred, w, h)
    cv2.line(img, pt1, pt2, COLOR_PRED, LINE_THICKNESS, cv2.LINE_AA)
    
    return img


def load_cache_item(cache_path: Path) -> Optional[dict]:
    """Load a single cache .npy file."""
    if not cache_path.exists():
        return None
    return np.load(str(cache_path), allow_pickle=True).item()


def load_model(weights_path: Path, device: str):
    """Load Fusion-CNN model."""
    model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=180).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def get_prediction(model, input_tensor, device: str):
    """Get model prediction for a single input."""
    with torch.no_grad():
        x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
        pred = model(x).cpu().numpy()[0]
    return pred  # (rho_norm, theta_norm)


def find_representative_sample(cache_dir: Path, n_samples: int = 10) -> Optional[str]:
    """
    Find a representative sample from cache directory.
    
    Strategy: Pick a sample with moderate error (not the best, not the worst)
    to show realistic performance.
    """
    npy_files = sorted(cache_dir.glob("*.npy"))
    if not npy_files:
        return None
    
    # Take a deterministic sample from middle of the sorted list
    np.random.seed(SEED)
    idx = len(npy_files) // 3  # ~33% position
    return npy_files[idx].stem


def get_common_sample_across_degradations(cache_root: Path, degradations: List[str]) -> Optional[str]:
    """
    Find a sample that exists in all degradation folders.
    
    Returns the sample stem (without extension).
    """
    sample_sets = []
    for deg in degradations:
        deg_dir = cache_root / deg
        if not deg_dir.exists():
            return None
        samples = {f.stem for f in deg_dir.glob("*.npy")}
        sample_sets.append(samples)
    
    if not sample_sets:
        return None
    
    # Find intersection
    common = sample_sets[0]
    for s in sample_sets[1:]:
        common = common & s
    
    if not common:
        return None
    
    # Pick one deterministically
    np.random.seed(SEED)
    common_list = sorted(list(common))
    idx = len(common_list) // 3  # ~33% position
    return common_list[idx]


# ----------------------------
# Figure A: Qualitative Visualization Grid (3x5)
# ----------------------------
def generate_figure_a():
    """
    Generate 3x5 qualitative visualization grid.
    
    Rows: MU-SID / SMD / Buoy
    Cols: Clean + fog_0.5 + gaussian_noise_30 + low_light_2.5 + rain_heavy
    """
    print("\n" + "=" * 60)
    print("Generating Figure A: Qualitative Visualization Grid (3x5)")
    print("=" * 60)
    
    n_rows = len(DATASET_ROWS)
    n_cols = len(DEGRADATION_COLS)
    
    # Determine cell size (we'll resize images to uniform size)
    cell_w, cell_h = 384, 216  # 16:9 aspect ratio
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    fig.suptitle("Degradation Robustness: Qualitative Comparison", fontsize=14, fontweight='bold')
    
    selected_samples = {}
    
    for row_idx, dataset in enumerate(DATASET_ROWS):
        cfg_ds = DATASET_CONFIGS[dataset]
        cache_root = cfg_ds["cache_root"]
        degraded_img_dir = cfg_ds["degraded_img_dir"]
        weights_path = cfg_ds["cnn_weights"]
        
        # Create denorm config for this dataset
        denorm_cfg = DenormConfig(
            unet_w=UNET_W, unet_h=UNET_H, resize_h=RESIZE_H,
            orig_w=cfg_ds["orig_w"], orig_h=cfg_ds["orig_h"]
        )
        
        # Check if weights exist
        if not weights_path.exists():
            print(f"[Warning] Weights not found for {dataset}: {weights_path}")
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].text(0.5, 0.5, "Weights\nNot Found", 
                                            ha='center', va='center', fontsize=10)
                axes[row_idx, col_idx].axis('off')
            continue
        
        # Load model
        print(f"\n[{dataset.upper()}] Loading model from {weights_path}")
        model = load_model(weights_path, DEVICE)
        
        # Find a common sample across all degradations
        sample_stem = get_common_sample_across_degradations(cache_root, DEGRADATION_COLS)
        
        if sample_stem is None:
            print(f"[Warning] No common sample found for {dataset}")
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].text(0.5, 0.5, "No Data", 
                                            ha='center', va='center', fontsize=10)
                axes[row_idx, col_idx].axis('off')
            continue
        
        selected_samples[dataset] = sample_stem
        print(f"[{dataset.upper()}] Selected sample: {sample_stem}")
        
        for col_idx, deg_name in enumerate(DEGRADATION_COLS):
            ax = axes[row_idx, col_idx]
            
            # Load cache for this sample + degradation
            cache_path = cache_root / deg_name / f"{sample_stem}.npy"
            cache_item = load_cache_item(cache_path)
            
            if cache_item is None:
                ax.text(0.5, 0.5, "No Cache", ha='center', va='center', fontsize=10)
                ax.axis('off')
                continue
            
            # Get GT and prediction
            gt_label = cache_item["label"]  # (rho_norm, theta_norm)
            input_tensor = cache_item["input"]
            pred = get_prediction(model, input_tensor, DEVICE)
            
            rho_gt, theta_gt = denorm_rho_theta(gt_label[0], gt_label[1], denorm_cfg)
            rho_pred, theta_pred = denorm_rho_theta(pred[0], pred[1], denorm_cfg)
            
            # Compute errors (in original pixel space)
            scale = cfg_ds["orig_w"] / UNET_W
            rho_err = abs(rho_pred - rho_gt) * scale
            theta_err = angular_diff_deg(theta_pred, theta_gt)
            
            # Load degraded image
            img_name = cache_item.get("img_name", f"{sample_stem}.jpg")
            img_path = degraded_img_dir / deg_name / img_name
            
            if not img_path.exists():
                # Try other extensions
                for ext in [".JPG", ".jpg", ".jpeg", ".png"]:
                    candidate = degraded_img_dir / deg_name / f"{sample_stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Draw lines
                img_with_lines = draw_horizon_lines(
                    img.copy(), rho_gt, theta_gt, rho_pred, theta_pred, denorm_cfg
                )
                
                # Resize for display
                img_display = cv2.resize(img_with_lines, (cell_w, cell_h))
                ax.imshow(img_display)
            else:
                ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', fontsize=10)
            
            # Add error annotation
            err_text = f"ρ: {rho_err:.1f}px | θ: {theta_err:.2f}°"
            ax.text(0.02, 0.02, err_text, transform=ax.transAxes, fontsize=8,
                   color='white', backgroundcolor='black', alpha=0.7,
                   verticalalignment='bottom')
            
            # Column title (only first row)
            if row_idx == 0:
                ax.set_title(DEGRADATION_LABELS[col_idx], fontsize=10, fontweight='bold')
            
            # Row label (only first column)
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS[row_idx], fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='Ground Truth'),
        mpatches.Patch(color='red', label='Prediction')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
              bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    ensure_dir(OUT_DIR)
    out_path = OUT_DIR / "fig_degradation_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Saved] Figure A -> {out_path}")
    
    # Print selected samples for caption
    print("\n[Selected Samples for Figure Caption]")
    for dataset, sample in selected_samples.items():
        print(f"  {dataset.upper()}: {sample}")
    
    return selected_samples


# ----------------------------
# Figure B: Top-5 Hardest Degradation Bar Chart
# ----------------------------
def generate_figure_b():
    """
    Generate Top-5 hardest degradation bar chart.
    
    Hardness metric: rho_le_10 (lower = harder)
    """
    print("\n" + "=" * 60)
    print("Generating Figure B: Top-5 Hardest Degradation Bar Chart")
    print("=" * 60)
    
    all_top5 = {}
    
    for dataset in DATASET_ROWS:
        cfg_ds = DATASET_CONFIGS[dataset]
        eval_csv = cfg_ds["eval_csv"]
        
        if not eval_csv.exists():
            print(f"[Warning] Eval CSV not found for {dataset}: {eval_csv}")
            continue
        
        # Load evaluation results
        df = pd.read_csv(eval_csv)
        
        # Exclude 'clean' from ranking
        df_deg = df[df["degradation"] != "clean"].copy()
        
        if df_deg.empty:
            print(f"[Warning] No degradation results for {dataset}")
            continue
        
        # Sort by rho_le_10 (ascending = harder first)
        df_sorted = df_deg.sort_values("rho_le_10", ascending=True)
        
        # Get Top-5
        top5 = df_sorted.head(5)[["degradation", "rho_le_10"]].values.tolist()
        all_top5[dataset] = top5
        
        print(f"\n[{dataset.upper()}] Top-5 Hardest Degradations (by ρ≤10px%):")
        for i, (deg, val) in enumerate(top5, 1):
            print(f"  {i}. {deg}: {val:.1f}%")
    
    if not all_top5:
        print("[Error] No evaluation data found. Please run evaluate_degraded.py first.")
        return
    
    # Create bar chart
    n_datasets = len(all_top5)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True)
    
    if n_datasets == 1:
        axes = [axes]
    
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.4, 5))  # Red-ish colors for "hard"
    
    for ax_idx, dataset in enumerate(DATASET_ROWS):
        if dataset not in all_top5:
            continue
        
        ax = axes[ax_idx]
        top5 = all_top5[dataset]
        
        degradations = [item[0] for item in top5]
        values = [item[1] for item in top5]
        
        y_pos = np.arange(len(degradations))
        
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                   f'{val:.1f}%', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(degradations, fontsize=9)
        ax.invert_yaxis()  # Top-1 at top
        ax.set_xlabel('ρ ≤ 10px (%)', fontsize=10)
        ax.set_title(f'{DATASET_LABELS[ax_idx]}', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(values) * 1.2)
        
        # Add rank numbers
        for i, (y, deg) in enumerate(zip(y_pos, degradations)):
            ax.text(-1, y, f'#{i+1}', va='center', ha='right', fontsize=9, fontweight='bold')
    
    fig.suptitle('Top-5 Hardest Degradations by Dataset\n(Lower ρ≤10px% = Harder)', 
                fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    ensure_dir(OUT_DIR)
    out_path = OUT_DIR / "fig_top5_bar.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Saved] Figure B -> {out_path}")
    
    return all_top5


# ----------------------------
# Main
# ----------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    print("=" * 60)
    print("Experiment 5: Generate Paper Figures")
    print("=" * 60)
    print(f"[Config]")
    print(f"  SEED   = {SEED}")
    print(f"  DEVICE = {DEVICE}")
    print(f"  Output = {OUT_DIR}")
    
    # Generate Figure A (Qualitative Grid)
    selected_samples = generate_figure_a()
    
    # Generate Figure B (Top-5 Bar Chart)
    top5_results = generate_figure_b()
    
    print("\n" + "=" * 60)
    print("[Done] Paper figures generated!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUT_DIR / 'fig_degradation_grid.png'}")
    print(f"  - {OUT_DIR / 'fig_top5_bar.png'}")


if __name__ == "__main__":
    main()
