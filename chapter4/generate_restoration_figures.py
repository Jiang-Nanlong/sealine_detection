# -*- coding: utf-8 -*-
"""
generate_restoration_figures.py

Chapter 4: UNet Restoration Branch Evaluation - 论文用图生成脚本

功能:
  1. 对 MU-SID 测试集应用退化（Gaussian Noise / Fog / Low Light）
  2. 使用 UNet 复原分支进行去噪/复原
  3. 计算 PSNR/SSIM 指标（degraded vs clean, restored vs clean）
  4. 生成论文用拼图（K行 × 4列：Clean / Degraded / Restored / Diff）
  5. 输出汇总指标文件

使用方法:
  python chapter4/generate_restoration_figures.py --seed 123 --k 6 --device cuda
  python chapter4/generate_restoration_figures.py --indices 1,5,20,30,50,100 --device cuda

输出:
  chapter4_results/
    fig_restoration_gaussian_noise_30.png
    fig_restoration_fog_0.5.png
    fig_restoration_low_light_2.5.png
    restoration_metrics_summary.txt
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet_model import RestorationGuidedHorizonNet

# ----------------------------
# Configuration
# ----------------------------
# 数据路径
MUSID_IMG_DIR = PROJECT_ROOT / "Hashmani's Dataset" / "MU-SID"
GT_CSV = PROJECT_ROOT / "splits_musid" / "GroundTruth_test.csv"

# 模型权重路径 (第四章使用 MU-SID 训练的 Stage C2 权重)
UNET_WEIGHTS = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"
DCE_WEIGHTS = PROJECT_ROOT / "weights" / "Epoch99.pth"

# 备用路径（如果上面的不存在）
ALT_UNET_WEIGHTS = PROJECT_ROOT / "weights_new" / "rghnet_best_c2.pth"
ALT_DCE_WEIGHTS = PROJECT_ROOT / "weights_new" / "Epoch99.pth"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "chapter4_results"

# 模型输入尺寸
MODEL_INPUT_SIZE = (1024, 576)  # (width, height)

# 退化配置（只保留3种，与用户要求一致）
DEGRADATIONS = {
    "gaussian_noise_30": {"type": "gaussian_noise", "sigma": 30},
    "fog_0.5": {"type": "fog", "intensity": 0.5},
    "low_light_2.5": {"type": "low_light", "gamma": 2.5},
}

# 全局种子
GLOBAL_SEED = 123


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_deterministic_seed(img_name: str, deg_name: str, base_seed: int) -> int:
    """生成基于 (img_name + deg_name + base_seed) 的确定性种子"""
    key = f"{img_name}_{deg_name}_{base_seed}"
    hash_val = hashlib.md5(key.encode()).hexdigest()
    return int(hash_val[:8], 16)


# ============================================================
# 退化函数（复用 test5 中的实现）
# ============================================================

def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_fog(img: np.ndarray, intensity: float) -> np.ndarray:
    """Add fog/haze effect."""
    fog_layer = np.ones_like(img, dtype=np.float32) * 255
    foggy = img.astype(np.float32) * (1 - intensity) + fog_layer * intensity
    return np.clip(foggy, 0, 255).astype(np.uint8)


def add_low_light(img: np.ndarray, gamma: float) -> np.ndarray:
    """Simulate low light by gamma correction (gamma > 1 darkens)."""
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_degradation(img: np.ndarray, deg_name: str, deg_config: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply degradation to image based on config."""
    deg_type = deg_config["type"]
    
    if deg_type == "gaussian_noise":
        return add_gaussian_noise(img, deg_config["sigma"], rng)
    elif deg_type == "fog":
        return add_fog(img, deg_config["intensity"])
    elif deg_type == "low_light":
        return add_low_light(img, deg_config["gamma"])
    else:
        raise ValueError(f"Unknown degradation type: {deg_type}")


# ============================================================
# 图像处理函数
# ============================================================

def load_image(img_path: Path) -> np.ndarray:
    """Load image as BGR uint8."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    return img


def resize_for_model(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize image to model input size (width, height)."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def bgr_to_tensor(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    """Convert BGR uint8 image to RGB float tensor [0,1], shape [1,3,H,W]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [1,3,H,W] RGB float to BGR uint8."""
    img = tensor.squeeze(0).detach().cpu().float().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))  # [H,W,3]
    img = (img * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


# ============================================================
# 指标计算（PSNR / SSIM）
# ============================================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute PSNR between two images.
    Images should be uint8 with same shape.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two images.
    Uses simplified SSIM formula (no sliding window, full image).
    """
    # 转换为灰度图计算 SSIM
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray1 = img1.astype(np.float64)
        gray2 = img2.astype(np.float64)
    
    # SSIM 常数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim


# ============================================================
# 数据加载
# ============================================================

def load_test_samples(gt_csv: Path, img_dir: Path) -> list:
    """
    Load test samples from ground truth CSV.
    Returns list of (img_name, img_path).
    """
    samples = []
    with open(gt_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            img_name = parts[0]
            
            # 尝试常见扩展名
            for ext in [".JPG", ".jpg", ".jpeg", ".png"]:
                img_path = img_dir / (img_name + ext)
                if img_path.exists():
                    samples.append((img_name, img_path))
                    break
    
    return samples


def select_representative_samples(samples: list, k: int, seed: int) -> list:
    """
    自动选择 K 张代表性样本。
    策略：均匀抽样，保证覆盖测试集不同部分。
    """
    n = len(samples)
    if k >= n:
        return samples
    
    # 使用种子保证可复现
    rng = np.random.default_rng(seed)
    
    # 均匀间隔选择
    step = n / k
    indices = [int(i * step) for i in range(k)]
    
    # 添加一点随机性（在间隔内微调）
    jittered_indices = []
    for i, idx in enumerate(indices):
        start = int(i * step)
        end = int((i + 1) * step) - 1 if i < k - 1 else n - 1
        jittered_idx = rng.integers(max(0, start), min(n, end + 1))
        jittered_indices.append(jittered_idx)
    
    # 去重并排序
    jittered_indices = sorted(list(set(jittered_indices)))
    
    # 如果去重后不足 k 个，补充
    while len(jittered_indices) < k:
        new_idx = rng.integers(0, n)
        if new_idx not in jittered_indices:
            jittered_indices.append(new_idx)
            jittered_indices.sort()
    
    selected = [samples[i] for i in jittered_indices[:k]]
    return selected


# ============================================================
# 可视化函数
# ============================================================

def create_diff_heatmap(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Create difference heatmap |img1 - img2|.
    Returns BGR image with colormap applied.
    """
    # 转灰度
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray1 = img1.astype(np.float32)
        gray2 = img2.astype(np.float32)
    
    diff = np.abs(gray1 - gray2)
    
    # 归一化到 0-255
    diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    
    # 应用颜色映射
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    
    return heatmap


def generate_comparison_figure(
    samples_data: list,
    deg_name: str,
    output_path: Path,
    figsize_per_row: tuple = (16, 3)
):
    """
    Generate comparison figure for one degradation type.
    
    samples_data: list of dict, each with keys:
        - img_name, clean, degraded, restored
        - psnr_deg, ssim_deg, psnr_rest, ssim_rest
    """
    k = len(samples_data)
    ncols = 4  # Clean, Degraded, Restored, Diff
    
    fig, axes = plt.subplots(k, ncols, figsize=(figsize_per_row[0], figsize_per_row[1] * k))
    
    if k == 1:
        axes = axes.reshape(1, -1)
    
    # 列标题
    col_titles = ["Clean", "Degraded", "Restored", "Diff (|Restored-Clean|)"]
    
    for row_idx, data in enumerate(samples_data):
        clean_rgb = cv2.cvtColor(data["clean"], cv2.COLOR_BGR2RGB)
        deg_rgb = cv2.cvtColor(data["degraded"], cv2.COLOR_BGR2RGB)
        rest_rgb = cv2.cvtColor(data["restored"], cv2.COLOR_BGR2RGB)
        diff_rgb = cv2.cvtColor(data["diff"], cv2.COLOR_BGR2RGB)
        
        # Clean
        axes[row_idx, 0].imshow(clean_rgb)
        axes[row_idx, 0].axis("off")
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=12, fontweight='bold')
        
        # Degraded with PSNR/SSIM
        axes[row_idx, 1].imshow(deg_rgb)
        axes[row_idx, 1].axis("off")
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_titles[1], fontsize=12, fontweight='bold')
        # 添加指标标注
        psnr_str = f"PSNR: {data['psnr_deg']:.2f} dB"
        ssim_str = f"SSIM: {data['ssim_deg']:.4f}"
        axes[row_idx, 1].text(
            0.02, 0.02, f"{psnr_str}\n{ssim_str}",
            transform=axes[row_idx, 1].transAxes,
            fontsize=9, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        
        # Restored with PSNR/SSIM
        axes[row_idx, 2].imshow(rest_rgb)
        axes[row_idx, 2].axis("off")
        if row_idx == 0:
            axes[row_idx, 2].set_title(col_titles[2], fontsize=12, fontweight='bold')
        # 添加指标标注
        psnr_str = f"PSNR: {data['psnr_rest']:.2f} dB"
        ssim_str = f"SSIM: {data['ssim_rest']:.4f}"
        # 计算提升
        psnr_gain = data['psnr_rest'] - data['psnr_deg']
        ssim_gain = data['ssim_rest'] - data['ssim_deg']
        gain_str = f"Δ: +{psnr_gain:.2f} dB, +{ssim_gain:.4f}"
        axes[row_idx, 2].text(
            0.02, 0.02, f"{psnr_str}\n{ssim_str}\n{gain_str}",
            transform=axes[row_idx, 2].transAxes,
            fontsize=9, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7)
        )
        
        # Diff heatmap
        axes[row_idx, 3].imshow(diff_rgb)
        axes[row_idx, 3].axis("off")
        if row_idx == 0:
            axes[row_idx, 3].set_title(col_titles[3], fontsize=12, fontweight='bold')
        
        # 行标签（图像名）
        axes[row_idx, 0].text(
            -0.05, 0.5, data["img_name"][:15],
            transform=axes[row_idx, 0].transAxes,
            fontsize=10, rotation=90,
            verticalalignment='center',
            horizontalalignment='right'
        )
    
    # 大标题
    deg_display = deg_name.replace("_", " ").title()
    fig.suptitle(f"UNet Restoration Branch Evaluation: {deg_display}", fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"  [Saved] {output_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Chapter 4: UNet Restoration Evaluation")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--k", type=int, default=6, help="Number of samples to use")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated indices to use (overrides --k)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()
    
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    
    print("=" * 70)
    print("Chapter 4: UNet Restoration Branch Evaluation")
    print("=" * 70)
    print(f"[Config] Seed: {GLOBAL_SEED}")
    print(f"[Config] Device: {device}")
    print(f"[Config] K: {args.k}")
    if args.indices:
        print(f"[Config] Manual indices: {args.indices}")
    
    # 确保输出目录
    ensure_dir(OUTPUT_DIR)
    
    # ----------------------------------------
    # 1. 加载测试样本
    # ----------------------------------------
    print(f"\n[Step 1] Loading test samples from {GT_CSV}...")
    samples = load_test_samples(GT_CSV, MUSID_IMG_DIR)
    print(f"  -> Found {len(samples)} test images")
    
    if args.indices:
        # 手动指定索引
        indices = [int(x.strip()) for x in args.indices.split(",")]
        selected = [samples[i] for i in indices if i < len(samples)]
        print(f"  -> Using manual indices: {indices}")
    else:
        # 自动选择代表性样本
        selected = select_representative_samples(samples, args.k, GLOBAL_SEED)
        print(f"  -> Auto-selected {len(selected)} representative samples")
    
    for i, (name, _) in enumerate(selected):
        print(f"    [{i}] {name}")
    
    # ----------------------------------------
    # 2. 加载模型
    # ----------------------------------------
    print(f"\n[Step 2] Loading UNet model...")
    
    # 查找权重文件
    unet_path = None
    for p in [UNET_WEIGHTS, ALT_UNET_WEIGHTS]:
        if p.exists():
            unet_path = p
            break
    
    dce_path = None
    for p in [DCE_WEIGHTS, ALT_DCE_WEIGHTS]:
        if p.exists():
            dce_path = p
            break
    
    if unet_path is None:
        print(f"[Error] UNet weights not found!")
        print(f"  Tried: {UNET_WEIGHTS}, {ALT_UNET_WEIGHTS}")
        print(f"  Please ensure one of these files exists.")
        sys.exit(1)
    
    if dce_path is None:
        print(f"[Error] DCE weights not found!")
        print(f"  Tried: {DCE_WEIGHTS}, {ALT_DCE_WEIGHTS}")
        sys.exit(1)
    
    print(f"  UNet weights: {unet_path}")
    print(f"  DCE weights: {dce_path}")
    
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=str(dce_path)).to(device)
    state = torch.load(str(unet_path), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("  -> Model loaded successfully")
    
    # ----------------------------------------
    # 3. 处理每种退化类型
    # ----------------------------------------
    all_metrics = {}
    
    for deg_name, deg_config in DEGRADATIONS.items():
        print(f"\n[Step 3] Processing degradation: {deg_name}")
        print("-" * 50)
        
        samples_data = []
        psnr_deg_list, ssim_deg_list = [], []
        psnr_rest_list, ssim_rest_list = [], []
        
        for img_idx, (img_name, img_path) in enumerate(selected):
            print(f"  [{img_idx + 1}/{len(selected)}] {img_name}...", end=" ")
            
            # (a) 加载原图
            clean_orig = load_image(img_path)
            clean = resize_for_model(clean_orig, MODEL_INPUT_SIZE)
            
            # (b) 应用退化
            seed = get_deterministic_seed(img_name, deg_name, GLOBAL_SEED)
            rng = np.random.default_rng(seed)
            degraded = apply_degradation(clean, deg_name, deg_config, rng)
            
            # (c) UNet 复原
            with torch.no_grad():
                deg_tensor = bgr_to_tensor(degraded, device)
                restored_tensor, _, _ = model(deg_tensor, enable_restoration=True, enable_segmentation=False)
            restored = tensor_to_bgr(restored_tensor)
            
            # (d) 计算指标
            psnr_deg = compute_psnr(degraded, clean)
            ssim_deg = compute_ssim(degraded, clean)
            psnr_rest = compute_psnr(restored, clean)
            ssim_rest = compute_ssim(restored, clean)
            
            psnr_deg_list.append(psnr_deg)
            ssim_deg_list.append(ssim_deg)
            psnr_rest_list.append(psnr_rest)
            ssim_rest_list.append(ssim_rest)
            
            print(f"PSNR: {psnr_deg:.2f} -> {psnr_rest:.2f} dB, SSIM: {ssim_deg:.4f} -> {ssim_rest:.4f}")
            
            # (e) 生成差异图
            diff = create_diff_heatmap(restored, clean)
            
            samples_data.append({
                "img_name": img_name,
                "clean": clean,
                "degraded": degraded,
                "restored": restored,
                "diff": diff,
                "psnr_deg": psnr_deg,
                "ssim_deg": ssim_deg,
                "psnr_rest": psnr_rest,
                "ssim_rest": ssim_rest,
            })
        
        # 统计指标
        metrics = {
            "psnr_deg_mean": np.mean(psnr_deg_list),
            "psnr_deg_std": np.std(psnr_deg_list),
            "ssim_deg_mean": np.mean(ssim_deg_list),
            "ssim_deg_std": np.std(ssim_deg_list),
            "psnr_rest_mean": np.mean(psnr_rest_list),
            "psnr_rest_std": np.std(psnr_rest_list),
            "ssim_rest_mean": np.mean(ssim_rest_list),
            "ssim_rest_std": np.std(ssim_rest_list),
        }
        all_metrics[deg_name] = metrics
        
        print(f"\n  Summary for {deg_name}:")
        print(f"    Degraded  -> PSNR: {metrics['psnr_deg_mean']:.2f} ± {metrics['psnr_deg_std']:.2f} dB, "
              f"SSIM: {metrics['ssim_deg_mean']:.4f} ± {metrics['ssim_deg_std']:.4f}")
        print(f"    Restored  -> PSNR: {metrics['psnr_rest_mean']:.2f} ± {metrics['psnr_rest_std']:.2f} dB, "
              f"SSIM: {metrics['ssim_rest_mean']:.4f} ± {metrics['ssim_rest_std']:.4f}")
        print(f"    Gain      -> ΔPSNR: +{metrics['psnr_rest_mean'] - metrics['psnr_deg_mean']:.2f} dB, "
              f"ΔSSIM: +{metrics['ssim_rest_mean'] - metrics['ssim_deg_mean']:.4f}")
        
        # 生成拼图
        fig_path = OUTPUT_DIR / f"fig_restoration_{deg_name}.png"
        generate_comparison_figure(samples_data, deg_name, fig_path)
    
    # ----------------------------------------
    # 4. 输出汇总文件
    # ----------------------------------------
    print(f"\n[Step 4] Writing summary file...")
    summary_path = OUTPUT_DIR / "restoration_metrics_summary.txt"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Chapter 4: UNet Restoration Branch Evaluation - Metrics Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: 2026-02-07\n")
        f.write(f"Seed: {GLOBAL_SEED}\n")
        f.write(f"Number of Samples: {len(selected)}\n")
        f.write(f"Model Input Size: {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("METRICS TABLE\n")
        f.write("-" * 70 + "\n\n")
        
        # Markdown 表格格式
        f.write("| Degradation | Degraded PSNR (dB) | Degraded SSIM | Restored PSNR (dB) | Restored SSIM | ΔPSNR | ΔSSIM |\n")
        f.write("|-------------|-------------------|---------------|-------------------|---------------|-------|-------|\n")
        
        for deg_name, m in all_metrics.items():
            d_psnr = f"{m['psnr_deg_mean']:.2f} ± {m['psnr_deg_std']:.2f}"
            d_ssim = f"{m['ssim_deg_mean']:.4f} ± {m['ssim_deg_std']:.4f}"
            r_psnr = f"{m['psnr_rest_mean']:.2f} ± {m['psnr_rest_std']:.2f}"
            r_ssim = f"{m['ssim_rest_mean']:.4f} ± {m['ssim_rest_std']:.4f}"
            delta_psnr = f"+{m['psnr_rest_mean'] - m['psnr_deg_mean']:.2f}"
            delta_ssim = f"+{m['ssim_rest_mean'] - m['ssim_deg_mean']:.4f}"
            
            f.write(f"| {deg_name:20s} | {d_psnr:17s} | {d_ssim:13s} | {r_psnr:17s} | {r_ssim:13s} | {delta_psnr:5s} | {delta_ssim:7s} |\n")
        
        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 70 + "\n\n")
        
        for deg_name, m in all_metrics.items():
            f.write(f"### {deg_name}\n")
            f.write(f"  Degraded:  PSNR = {m['psnr_deg_mean']:.2f} ± {m['psnr_deg_std']:.2f} dB, "
                    f"SSIM = {m['ssim_deg_mean']:.4f} ± {m['ssim_deg_std']:.4f}\n")
            f.write(f"  Restored:  PSNR = {m['psnr_rest_mean']:.2f} ± {m['psnr_rest_std']:.2f} dB, "
                    f"SSIM = {m['ssim_rest_mean']:.4f} ± {m['ssim_rest_std']:.4f}\n")
            f.write(f"  Gain:      ΔPSNR = +{m['psnr_rest_mean'] - m['psnr_deg_mean']:.2f} dB, "
                    f"ΔSSIM = +{m['ssim_rest_mean'] - m['ssim_deg_mean']:.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("SELECTED SAMPLES\n")
        f.write("-" * 70 + "\n\n")
        for i, (name, path) in enumerate(selected):
            f.write(f"  [{i}] {name}\n")
    
    print(f"  [Saved] {summary_path}")
    
    # ----------------------------------------
    # 完成
    # ----------------------------------------
    print("\n" + "=" * 70)
    print("[Done] All figures and metrics saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)
    print("\nOutput files:")
    for deg_name in DEGRADATIONS.keys():
        print(f"  - fig_restoration_{deg_name}.png")
    print(f"  - restoration_metrics_summary.txt")


if __name__ == "__main__":
    main()
