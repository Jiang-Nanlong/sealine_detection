# -*- coding: utf-8 -*-
"""
generate_paper_figures.py

Generate paper figures for Experiment 5: Degradation Robustness.

Figure A: 3x5 qualitative visualization grid
  Rows: MU-SID / SMD / Buoy
  Cols: Clean + fog_0.5 + gaussian_noise_30 + low_light_2.5 + rain_heavy

Figure B: Top-5 hardest degradation bar chart
  Grouped by dataset, sorted by rho_le_10 (lower = harder)

Figure C: Buoy fog intensity trend (Clean -> Fog 0.3 -> Fog 0.5)

Usage:
  python test5/generate_paper_figures.py [--seed 123] [--device cuda]
  python test5/generate_paper_figures.py --mu_idx 10 --smd_idx 20 --buoy_idx 30

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
# PyCharm 配置�?(在这里修�?
# ============================
SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 要显示的退化类型（固定顺序�?
DEGRADATION_COLS = ["clean", "fog_0.5", "gaussian_noise_30", "low_light_2.5", "rain_heavy"]
DEGRADATION_LABELS = ["Clean", "Fog 0.5", "Noise σ=30", "Low Light γ=2.5", "Rain Heavy"]

# 数据集配置（行顺序）
DATASET_ROWS = ["musid", "smd", "buoy"]
DATASET_LABELS = ["MU-SID", "SMD", "Buoy"]

# 手动指定样本索引（None表示自动选择�?
MANUAL_SAMPLE_IDX = {
    "musid": None,
    "smd": None,
    "buoy": None,
}
# ============================

# 命令行参数覆�?
if "--seed" in sys.argv:
    _idx = sys.argv.index("--seed")
    if _idx + 1 < len(sys.argv):
        SEED = int(sys.argv[_idx + 1])

if "--device" in sys.argv:
    _idx = sys.argv.index("--device")
    if _idx + 1 < len(sys.argv):
        DEVICE = sys.argv[_idx + 1]

# 手动样本索引覆盖
if "--mu_idx" in sys.argv:
    _idx = sys.argv.index("--mu_idx")
    if _idx + 1 < len(sys.argv):
        MANUAL_SAMPLE_IDX["musid"] = int(sys.argv[_idx + 1])

if "--smd_idx" in sys.argv:
    _idx = sys.argv.index("--smd_idx")
    if _idx + 1 < len(sys.argv):
        MANUAL_SAMPLE_IDX["smd"] = int(sys.argv[_idx + 1])

if "--buoy_idx" in sys.argv:
    _idx = sys.argv.index("--buoy_idx")
    if _idx + 1 < len(sys.argv):
        MANUAL_SAMPLE_IDX["buoy"] = int(sys.argv[_idx + 1])

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
test1_DIR = PROJECT_ROOT / "test1"

# UNet input size (used for Radon label generation)
UNET_W = 1024
UNET_H = 576
RESIZE_H = 2240

# Drawing config - RGB format (用于 matplotlib 显示)
# cv2.line �?RGB 图像上绘制时，颜色需要按 RGB 顺序
COLOR_GT_RGB = (0, 255, 0)      # Green for GT (RGB)
COLOR_PRED_RGB = (255, 0, 0)    # Red for Prediction (RGB)
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 255, 255)

# 数据集配�?
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
        "cnn_weights": test1_DIR / "weights" / "best_fusion_cnn_smd.pth",
        "orig_w": 1920,
        "orig_h": 1080,
    },
    "buoy": {
        "degraded_img_dir": TEST5_DIR / "degraded_images_buoy",
        "cache_root": TEST5_DIR / "FusionCache_Degraded_Buoy",
        "eval_csv": TEST5_DIR / "eval_results_buoy" / "degradation_results.csv",
        "cnn_weights": test1_DIR / "weights" / "best_fusion_cnn_buoy.pth",
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
# Utility functions (复用�?visualize_degraded.py �?evaluate_degraded.py)
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


def draw_horizon_lines_rgb(img_rgb, rho_gt, theta_gt, rho_pred, theta_pred, cfg: DenormConfig):
    """
    Draw GT and prediction lines on RGB image.
    
    注意：img_rgb 已经�?RGB 格式（从 BGR 转换而来�?
    cv2.line 直接按通道顺序绘制，所以传�?RGB 颜色即可
    """
    h, w = img_rgb.shape[:2]
    
    # Scale rho from UNET space to original image space
    scale = cfg.orig_w / cfg.unet_w
    rho_gt_scaled = rho_gt * scale
    rho_pred_scaled = rho_pred * scale
    
    # Draw GT line (green) - 直接使用 RGB 颜色
    pt1, pt2 = rho_theta_to_line(rho_gt_scaled, theta_gt, w, h)
    cv2.line(img_rgb, pt1, pt2, COLOR_GT_RGB, LINE_THICKNESS, cv2.LINE_AA)
    
    # Draw prediction line (red) - 直接使用 RGB 颜色
    pt1, pt2 = rho_theta_to_line(rho_pred_scaled, theta_pred, w, h)
    cv2.line(img_rgb, pt1, pt2, COLOR_PRED_RGB, LINE_THICKNESS, cv2.LINE_AA)
    
    return img_rgb


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


def compute_all_errors_for_dataset(model, cache_root: Path, denorm_cfg: DenormConfig, 
                                   degradations: List[str], device: str) -> Dict:
    """
    Compute rho errors for all samples across all degradations.
    
    Returns:
        dict mapping sample_stem -> {deg_name: rho_error}
    """
    # First, find common samples across all degradations
    sample_sets = []
    for deg in degradations:
        deg_dir = cache_root / deg
        if not deg_dir.exists():
            return {}
        samples = {f.stem for f in deg_dir.glob("*.npy")}
        sample_sets.append(samples)
    
    if not sample_sets:
        return {}
    
    common = sample_sets[0]
    for s in sample_sets[1:]:
        common = common & s
    
    if not common:
        return {}
    
    common_list = sorted(list(common))
    scale = denorm_cfg.orig_w / denorm_cfg.unet_w
    
    # Compute errors for each sample
    all_errors = {}
    for sample_stem in common_list:
        sample_errors = {}
        for deg in degradations:
            cache_path = cache_root / deg / f"{sample_stem}.npy"
            cache_item = load_cache_item(cache_path)
            if cache_item is None:
                continue
            
            gt_label = cache_item["label"]
            input_tensor = cache_item["input"]
            pred = get_prediction(model, input_tensor, device)
            
            rho_gt, theta_gt = denorm_rho_theta(gt_label[0], gt_label[1], denorm_cfg)
            rho_pred, theta_pred = denorm_rho_theta(pred[0], pred[1], denorm_cfg)
            
            rho_err = abs(rho_pred - rho_gt) * scale
            sample_errors[deg] = rho_err
        
        if len(sample_errors) == len(degradations):
            all_errors[sample_stem] = sample_errors
    
    return all_errors


def select_representative_sample(all_errors: Dict, degradations: List[str]) -> Optional[str]:
    """
    Select representative sample using median-based criterion.
    
    Algorithm:
    1. Compute median error for each degradation
    2. For each sample, compute L1 distance to medians across all degradations
    3. Filter samples where clean error <= P75 of clean errors
    4. Return sample with minimum total L1 distance
    """
    if not all_errors:
        return None
    
    samples = list(all_errors.keys())
    
    # Compute per-degradation medians
    medians = {}
    for deg in degradations:
        errors = [all_errors[s][deg] for s in samples]
        medians[deg] = np.median(errors)
    
    # Compute P75 for clean
    clean_errors = [all_errors[s]["clean"] for s in samples]
    clean_p75 = np.percentile(clean_errors, 75)
    
    # Filter samples where clean error <= P75
    valid_samples = [s for s in samples if all_errors[s]["clean"] <= clean_p75]
    
    if not valid_samples:
        valid_samples = samples  # Fallback to all samples
    
    # Compute L1 distance to medians for each valid sample
    best_sample = None
    best_score = float('inf')
    
    for sample in valid_samples:
        score = sum(abs(all_errors[sample][deg] - medians[deg]) for deg in degradations)
        if score < best_score:
            best_score = score
            best_sample = sample
    
    return best_sample


def get_sample_by_index(cache_root: Path, degradations: List[str], idx: int) -> Optional[str]:
    """Get sample stem by index from common samples."""
    sample_sets = []
    for deg in degradations:
        deg_dir = cache_root / deg
        if not deg_dir.exists():
            return None
        samples = {f.stem for f in deg_dir.glob("*.npy")}
        sample_sets.append(samples)
    
    if not sample_sets:
        return None
    
    common = sample_sets[0]
    for s in sample_sets[1:]:
        common = common & s
    
    common_list = sorted(list(common))
    if idx < 0 or idx >= len(common_list):
        print(f"[Warning] Index {idx} out of range (0-{len(common_list)-1}), using idx=0")
        idx = 0
    
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
        
        # Select sample: manual or automatic
        manual_idx = MANUAL_SAMPLE_IDX.get(dataset)
        
        if manual_idx is not None:
            # Use manually specified index
            sample_stem = get_sample_by_index(cache_root, DEGRADATION_COLS, manual_idx)
            print(f"[{dataset.upper()}] Using manual index: {manual_idx}")
        else:
            # Automatic selection using median-based algorithm
            print(f"[{dataset.upper()}] Auto-selecting representative sample...")
            all_errors = compute_all_errors_for_dataset(
                model, cache_root, denorm_cfg, DEGRADATION_COLS, DEVICE
            )
            sample_stem = select_representative_sample(all_errors, DEGRADATION_COLS)
        
        if sample_stem is None:
            print(f"[Warning] No sample found for {dataset}")
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].text(0.5, 0.5, "No Data", 
                                            ha='center', va='center', fontsize=10)
                axes[row_idx, col_idx].axis('off')
            continue
        
        # Find index of selected sample for reporting
        sample_sets = []
        for deg in DEGRADATION_COLS:
            deg_dir = cache_root / deg
            if deg_dir.exists():
                samples = {f.stem for f in deg_dir.glob("*.npy")}
                sample_sets.append(samples)
        common = sample_sets[0] if sample_sets else set()
        for s in sample_sets[1:]:
            common = common & s
        common_list = sorted(list(common))
        sample_idx = common_list.index(sample_stem) if sample_stem in common_list else -1
        
        selected_samples[dataset] = {"stem": sample_stem, "idx": sample_idx}
        print(f"[{dataset.upper()}] Selected sample: {sample_stem} (idx={sample_idx})")
        
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
                # Read image as BGR, convert to RGB for matplotlib
                img_bgr = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Draw lines on RGB image
                img_with_lines = draw_horizon_lines_rgb(
                    img_rgb.copy(), rho_gt, theta_gt, rho_pred, theta_pred, denorm_cfg
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
    
    # Add legend with correct colors (matplotlib 使用 RGB hex)
    legend_elements = [
        mpatches.Patch(facecolor='#00FF00', edgecolor='black', label='Ground Truth'),
        mpatches.Patch(facecolor='#FF0000', edgecolor='black', label='Prediction')
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
    for dataset, info in selected_samples.items():
        print(f"  {dataset.upper()}: {info['stem']} (idx={info['idx']})")
    
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
        
        print(f"\n[{dataset.upper()}] Top-5 Hardest Degradations (by ρ�?0px%):")
        for i, (deg, val) in enumerate(top5, 1):
            print(f"  {i}. {deg}: {val:.1f}%")
    
    if not all_top5:
        print("[Error] No evaluation data found. Please run evaluate_degraded.py first.")
        return None
    
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
        
        # Merge rank into labels: "#1 gaussian_noise_30"
        degradations = [f"#{i+1} {item[0]}" for i, item in enumerate(top5)]
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
        ax.set_xlabel('ρ �?10px (%)', fontsize=10)
        ax.set_title(f'{DATASET_LABELS[ax_idx]}', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(values) * 1.25)
    
    fig.suptitle('Top-5 Hardest Degradations by Dataset\n(Lower ρ�?0px% = Harder)', 
                fontsize=12, fontweight='bold')
    
    # Adjust layout for long labels
    fig.subplots_adjust(left=0.22)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save figure
    ensure_dir(OUT_DIR)
    out_path = OUT_DIR / "fig_top5_bar.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Saved] Figure B -> {out_path}")
    
    return all_top5


# ----------------------------
# Figure C: Buoy Fog Intensity Trend
# ----------------------------
def generate_figure_c():
    """
    Generate Buoy fog intensity trend chart.
    
    X-axis: Clean, Fog 0.3, Fog 0.5
    Y-axis: rho�?0px(%) and theta�?°(%)
    """
    print("\n" + "=" * 60)
    print("Generating Figure C: Buoy Fog Intensity Trend")
    print("=" * 60)
    
    cfg_ds = DATASET_CONFIGS["buoy"]
    eval_csv = cfg_ds["eval_csv"]
    
    if not eval_csv.exists():
        print(f"[Error] Eval CSV not found for buoy: {eval_csv}")
        return None
    
    # Load evaluation results
    df = pd.read_csv(eval_csv)
    
    # Get fog degradations
    fog_levels = ["clean", "fog_0.3", "fog_0.5"]
    fog_labels = ["Clean", "Fog 0.3", "Fog 0.5"]
    
    rho_values = []
    theta_values = []
    
    for deg in fog_levels:
        row = df[df["degradation"] == deg]
        if row.empty:
            print(f"[Warning] No data for {deg}")
            rho_values.append(0)
            theta_values.append(0)
        else:
            rho_values.append(row["rho_le_10"].values[0])
            theta_values.append(row["theta_le_2"].values[0])
    
    # Print values
    print("\n[Buoy Fog Trend Values]")
    for i, (deg, rho, theta) in enumerate(zip(fog_labels, rho_values, theta_values)):
        print(f"  {deg}: ρ�?0px = {rho:.1f}%, θ�?° = {theta:.1f}%")
    
    # Create trend chart
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x = np.arange(len(fog_levels))
    
    # Plot lines with markers
    ax.plot(x, rho_values, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='ρ �?10px (%)')
    ax.plot(x, theta_values, 's--', color='#ff7f0e', linewidth=2, markersize=8, label='θ �?2° (%)')
    
    # Add value annotations
    for i, (rho, theta) in enumerate(zip(rho_values, theta_values)):
        ax.annotate(f'{rho:.1f}%', (x[i], rho), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, color='#1f77b4')
        ax.annotate(f'{theta:.1f}%', (x[i], theta), textcoords="offset points", 
                   xytext=(0, -15), ha='center', fontsize=9, color='#ff7f0e')
    
    ax.set_xticks(x)
    ax.set_xticklabels(fog_labels, fontsize=10)
    ax.set_xlabel('Fog Intensity', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title('Buoy Dataset: Performance vs Fog Intensity', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    ensure_dir(OUT_DIR)
    out_path = OUT_DIR / "fig_buoy_fog_trend.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Saved] Figure C -> {out_path}")
    
    return {"rho_le_10": rho_values, "theta_le_2": theta_values}


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
    print(f"  Manual indices: musid={MANUAL_SAMPLE_IDX['musid']}, "
          f"smd={MANUAL_SAMPLE_IDX['smd']}, buoy={MANUAL_SAMPLE_IDX['buoy']}")
    
    # Generate Figure A (Qualitative Grid)
    selected_samples = generate_figure_a()
    
    # Generate Figure B (Top-5 Bar Chart)
    top5_results = generate_figure_b()
    
    # Generate Figure C (Buoy Fog Trend)
    trend_results = generate_figure_c()
    
    print("\n" + "=" * 60)
    print("[Done] Paper figures generated!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUT_DIR / 'fig_degradation_grid.png'}")
    print(f"  - {OUT_DIR / 'fig_top5_bar.png'}")
    print(f"  - {OUT_DIR / 'fig_buoy_fog_trend.png'}")


if __name__ == "__main__":
    main()
