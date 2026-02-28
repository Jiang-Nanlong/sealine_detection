#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-7  Fusion CNN 输出可视化
  (a) conv_end 输出的原始热力图（logits）
  (b) 经过 softmax(logits / T) 处理后的概率分布图
温度参数 T 使用模型训练后学到的值（nn.Parameter）。
"""
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================================
# 配置
# ============================================================================
DATASET_ROOT = PROJECT_ROOT / "Hashmani's Dataset"
CACHE_ROOT   = DATASET_ROOT / "FusionCache_new_1024x576"
CACHE_SPLIT  = "test"

FUSION_CKPT = PROJECT_ROOT / "weights_new" / "best_fusion_cnn_1024x576.pth"
RESIZE_H = 2240   # sinogram rho axis
RESIZE_W = 180    # sinogram theta axis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 工具函数
# ============================================================================

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fusion_model():
    """加载 Fusion CNN 模型"""
    from cnn_model import HorizonResNet

    try:
        model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=RESIZE_W).to(DEVICE)
    except TypeError:
        model = HorizonResNet().to(DEVICE)

    ckpt = safe_torch_load(str(FUSION_CKPT), DEVICE)
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def pick_one_cache_file():
    """随机选取一个测试集cache文件"""
    cache_dir = Path(CACHE_ROOT) / CACHE_SPLIT
    files = sorted(cache_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No cache npy found: {cache_dir}/*.npy")
    rs = np.random.RandomState(RANDOM_SEED)
    idx = rs.randint(0, len(files))
    return files[idx]


def load_cache_item(npy_path):
    d = np.load(str(npy_path), allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    if not isinstance(d, dict):
        raise ValueError(f"Cache file is not a dict: {npy_path}")
    return d


@torch.no_grad()
def extract_heatmap_and_prob(model, x):
    """
    手动走 forward 提取:
      logits  : conv_end 输出的原始热力图  [Hf, Wf]
      prob    : softmax(logits / T) 概率图  [Hf, Wf]
      T       : 模型学到的温度参数
    """
    h = x
    h = model.conv1(h)
    if hasattr(model, "bn1"):
        h = model.bn1(h)
    h = F.relu(h)
    if hasattr(model, "maxpool"):
        h = model.maxpool(h)
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        if hasattr(model, lname):
            h = getattr(model, lname)(h)
    if hasattr(model, "cbam"):
        h = model.cbam(h)

    logits = model.conv_end(h)  # [B, 1, Hf, Wf]

    b, c, hf, wf = logits.shape
    x_flat = logits.view(b, -1)

    # 使用模型学到的温度参数
    T = float(model.temperature.data.item())
    prob_flat = F.softmax(x_flat / max(T, 1e-3), dim=1)
    prob = prob_flat.view(b, hf, wf)

    logits_np = logits[0, 0].cpu().numpy()
    prob_np = prob[0].cpu().numpy()

    return logits_np, prob_np, T


def generate_heatmap_figure(logits, prob, T, output_dir):
    """生成两张独立的图"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11

    hf, wf = logits.shape

    # ---- (a) CNN 原始热力图 (logits) ----
    fig_a, ax_a = plt.subplots(1, 1, figsize=(5, 8))
    im_a = ax_a.imshow(logits, aspect='auto', cmap='hot', interpolation='bilinear')
    ax_a.set_title('(a) CNN output heatmap (logits)', fontsize=12, pad=8)
    ax_a.set_xlabel('θ (feature bins)', fontsize=10)
    ax_a.set_ylabel('ρ (feature bins)', fontsize=10)
    fig_a.colorbar(im_a, ax=ax_a, fraction=0.025, pad=0.04)
    fig_a.tight_layout()
    path_a = output_dir / "fig5_7a_cnn_heatmap.png"
    fig_a.savefig(str(path_a), dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_a)
    print(f"  ✓ 已保存: {path_a}")

    # ---- (b) Softmax 概率图 ----
    fig_b, ax_b = plt.subplots(1, 1, figsize=(5, 8))
    im_b = ax_b.imshow(prob, aspect='auto', cmap='hot', interpolation='bilinear')
    ax_b.set_title(f'(b) Probability map  softmax(logits / T)\nT = {T:.4f}',
                   fontsize=12, pad=8)
    ax_b.set_xlabel('θ (feature bins)', fontsize=10)
    ax_b.set_ylabel('ρ (feature bins)', fontsize=10)

    # 标注概率最大点
    r_max, c_max = np.unravel_index(np.argmax(prob), prob.shape)
    ax_b.scatter([c_max], [r_max], s=80, marker='x', color='cyan', linewidths=2,
                 label=f'peak ({r_max}, {c_max})')
    ax_b.legend(loc='upper right', fontsize=9)

    fig_b.colorbar(im_b, ax=ax_b, fraction=0.025, pad=0.04)
    fig_b.tight_layout()
    path_b = output_dir / "fig5_7b_softmax_prob.png"
    fig_b.savefig(str(path_b), dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_b)
    print(f"  ✓ 已保存: {path_b}")


def main():
    print("=" * 60)
    print("图5-7  Fusion CNN 热力图 & 概率分布图")
    print("=" * 60)

    # 加载模型
    print("\n[1] 加载 Fusion CNN 模型")
    model = load_fusion_model()
    T = float(model.temperature.data.item())
    print(f"    学到的温度参数 T = {T:.6f}")

    # 选取样本
    print("\n[2] 加载测试样本")
    npy_path = pick_one_cache_file()
    print(f"    cache: {npy_path.name}")
    cache = load_cache_item(npy_path)

    # 构造 CNN 输入（4 通道 sinogram stack）
    sinogram = cache.get("sinogram", None)            # 可能直接存了 4-ch
    if sinogram is None:
        sinogram = cache.get("input", None)
    if sinogram is None:
        # 逐通道拼
        keys = [k for k in cache.keys() if "sino" in k.lower() or "channel" in k.lower()]
        if keys:
            sinogram = np.stack([cache[k] for k in sorted(keys)], axis=0)

    if sinogram is None:
        raise RuntimeError(f"cache 中没有找到 sinogram 数据, keys={list(cache.keys())}")

    # 保证 [4, H, W]
    if sinogram.ndim == 2:
        sinogram = np.stack([sinogram] * 4, axis=0)
    elif sinogram.ndim == 3 and sinogram.shape[0] != 4:
        # 可能是 HWC
        if sinogram.shape[-1] == 4:
            sinogram = np.transpose(sinogram, (2, 0, 1))

    x = torch.from_numpy(sinogram.astype(np.float32)).unsqueeze(0).to(DEVICE)
    print(f"    CNN 输入 shape: {x.shape}")

    # 前向传播
    print("\n[3] 提取热力图 & 概率图")
    logits, prob, T = extract_heatmap_and_prob(model, x)
    print(f"    logits shape: {logits.shape}  range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"    prob   shape: {prob.shape}    range: [{prob.min():.6f}, {prob.max():.6f}]")
    print(f"    温度 T = {T:.6f}")

    # 生成图片
    print("\n[4] 生成图片")
    generate_heatmap_figure(logits, prob, T, output_dir)

    print("\n" + "=" * 60)
    print("[完成]")
    print("=" * 60)


if __name__ == "__main__":
    main()
