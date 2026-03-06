#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-5 双流互补性可视化（2行×4列）
第一行：(1)原图 (2)梯度流输入 (3)语义流输入 (4)融合输入
第二行：(5)留空 (6)梯度流正弦图 (7)语义流正弦图 (8)融合正弦图
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import letterbox_rgb_u8
from gradient_radon import TextureSuppressedMuSCoWERT

# ============================================================================
# 配置
# ============================================================================
DATASET_ROOT = PROJECT_ROOT / "Hashmani's Dataset" / "MU-SID"
GT_CSV = PROJECT_ROOT / "splits_musid" / "GroundTruth_test.csv"
UNET_WEIGHTS = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"
DCE_WEIGHTS = PROJECT_ROOT / "weights" / "Epoch99.pth"

IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# 语义边缘参数
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)
output_fig = output_dir / "fig5_5_dual_stream_comparison.png"


def load_all_test_images():
    """加载所有MU-SID测试集图像路径"""
    if not GT_CSV.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {GT_CSV}")
    
    image_stems = []
    try:
        with open(GT_CSV, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    image_stems.append(parts[0])
    except Exception as e:
        raise RuntimeError(f"读取CSV失败: {e}")
    
    image_stems = sorted(set(image_stems))
    
    image_paths = []
    for stem in image_stems:
        img_path = DATASET_ROOT / f"{stem}.JPG"
        if img_path.exists():
            image_paths.append(img_path)
    
    return sorted(image_paths)


def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组 (H×W×C)"""
    img = tensor.squeeze().detach().cpu().float().clamp(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def load_stage1_model():
    """加载Stage-1模型"""
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=str(DCE_WEIGHTS))
    model = model.to(DEVICE)
    
    try:
        state = torch.load(str(UNET_WEIGHTS), map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    model.eval()
    return model


@torch.no_grad()
def run_stage1(model, img_bgr):
    """运行Stage-1获取复原图和分割掩码"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_resized, meta = letterbox_rgb_u8(rgb, IMG_SIZE, pad_value=0)
    
    inp_tensor = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0)
    inp_tensor = inp_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    restored_t, seg_logits, _, _, _ = model(inp_tensor, None, True, True)
    
    restored_np = restored_t[0].permute(1, 2, 0).cpu().numpy()
    restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
    
    mask = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    
    h_orig, w_orig = img_bgr.shape[:2]
    pad_top = int(meta['pad_top'])
    pad_left = int(meta['pad_left'])
    new_h = int(meta['new_h'])
    new_w = int(meta['new_w'])
    
    restored_roi = restored_np[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    mask_roi = mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    
    restored_resized = cv2.resize(restored_roi, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_roi, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    return restored_resized, mask_resized


def post_process_mask(mask):
    """后处理掩码：取顶部最大连通域"""
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
    if num <= 1:
        return m
    
    best_lab = None
    best_area = -1
    for lab in range(1, num):
        y = stats[lab, cv2.CC_STAT_TOP]
        area = stats[lab, cv2.CC_STAT_AREA]
        if y <= 10:
            if area > best_area:
                best_area = area
                best_lab = lab
    
    if best_lab is None:
        for lab in range(1, num):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_lab = lab
    
    out = (labels == best_lab).astype(np.uint8) if best_lab is not None else m
    return out


def extract_gradient_features(restored_rgb):
    """提取梯度流特征（多尺度梯度加权图）"""
    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    
    try:
        _, _, debug_info, _ = detector.detect(restored_bgr)
    except:
        debug_info = {}
    
    # 提取各尺度的加权图
    grad_maps = []
    for s in [1, 2, 3]:
        if s in debug_info and 'map' in debug_info[s]:
            grad_maps.append(debug_info[s]['map'])
        else:
            grad_maps.append(np.zeros_like(restored_rgb[:,:,0], dtype=np.float32))
    
    # 合成梯度流输入特征（取最大值融合）
    grad_feature = np.maximum.reduce(grad_maps)
    
    return grad_feature, grad_maps


def extract_semantic_edges(mask):
    """提取语义边缘"""
    edges = cv2.Canny((mask * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
    if EDGE_DILATE > 0:
        k = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, k, iterations=EDGE_DILATE)
    return edges


def compute_sinogram(feature, theta_range=180):
    """计算正弦图"""
    detector = TextureSuppressedMuSCoWERT(scales=[1], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, theta_range, endpoint=False)
    sino = detector._radon_gpu(feature.astype(np.uint8), theta_scan)
    return sino


def create_fusion_input_visualization(grad_maps, sem_edges):
    """创建融合输入可视化（2×2小格拼图）"""
    h, w = grad_maps[0].shape
    
    # 归一化到0-255
    def norm(x):
        x = x.astype(np.float32)
        mi, ma = x.min(), x.max()
        if ma - mi < 1e-6:
            return np.zeros_like(x, dtype=np.uint8)
        return ((x - mi) / (ma - mi) * 255).astype(np.uint8)
    
    # 创建2×2拼图
    top_left = norm(grad_maps[0])
    top_right = norm(grad_maps[1])
    bottom_left = norm(grad_maps[2])
    bottom_right = sem_edges
    
    # 缩放到一半大小
    half_h, half_w = h // 2, w // 2
    top_left = cv2.resize(top_left, (half_w, half_h))
    top_right = cv2.resize(top_right, (half_w, half_h))
    bottom_left = cv2.resize(bottom_left, (half_w, half_h))
    bottom_right = cv2.resize(bottom_right, (half_w, half_h))
    
    # 拼接
    top_row = np.hstack([top_left, top_right])
    bottom_row = np.hstack([bottom_left, bottom_right])
    fusion_vis = np.vstack([top_row, bottom_row])
    
    return fusion_vis


def add_sinogram_axes_labels(ax, title):
    """为正弦图添加坐标轴标注"""
    ax.set_xlabel('θ (degrees)', fontsize=9)
    ax.set_ylabel('ρ (pixels)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=5)


def generate_figure(original_rgb, grad_feature, grad_maps, sem_edges, grad_sino, sem_sino, fusion_sino, output_path):
    """生成2×4对比图"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.15, hspace=0.25, 
                        left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 第一行：原图 + 三种输入对比
    # (1) 原图
    ax = axes[0, 0]
    ax.imshow(original_rgb)
    ax.set_title('(1) Original image', fontsize=10, pad=5)
    ax.axis('off')
    
    # (2) 梯度流输入
    ax = axes[0, 1]
    ax.imshow(grad_feature, cmap='gray')
    ax.set_title('(2) Gradient stream input', fontsize=10, pad=5)
    ax.axis('off')
    
    # (3) 语义流输入
    ax = axes[0, 2]
    ax.imshow(sem_edges, cmap='gray')
    ax.set_title('(3) Semantic stream input', fontsize=10, pad=5)
    ax.axis('off')
    
    # (4) 融合输入（4通道示意）
    ax = axes[0, 3]
    fusion_vis = create_fusion_input_visualization(grad_maps, sem_edges)
    ax.imshow(fusion_vis, cmap='gray')
    ax.set_title('(4) Fusion input (4-ch)', fontsize=10, pad=5)
    ax.axis('off')
    
    # 第二行：正弦图域对比（使用热力图）
    # (5) 留空（对应原图）
    ax = axes[1, 0]
    ax.axis('off')
    
    # (6) 梯度流正弦图（对应梯度流输入）
    ax = axes[1, 1]
    im = ax.imshow(grad_sino, aspect='auto', cmap='hot', interpolation='bilinear')
    add_sinogram_axes_labels(ax, '(6) Gradient sinogram Sg(ρ,θ)')
    
    # (7) 语义流正弦图（对应语义流输入）
    ax = axes[1, 2]
    im = ax.imshow(sem_sino, aspect='auto', cmap='hot', interpolation='bilinear')
    add_sinogram_axes_labels(ax, '(7) Semantic sinogram Ss(ρ,θ)')
    
    # (8) 融合正弦图（对应融合输入）
    ax = axes[1, 3]
    im = ax.imshow(fusion_sino, aspect='auto', cmap='hot', interpolation='bilinear')
    add_sinogram_axes_labels(ax, '(8) Fused sinogram Sf(ρ,θ)')
    
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    print("=" * 70)
    print("图5-5 双流互补性可视化（2行×4列）")
    print("=" * 70)
    
    # 加载模型
    print("\n[步骤1] 加载Stage-1模型")
    model = load_stage1_model()
    if model is None:
        return
    
    # 加载测试集并随机选择1张
    print("\n[步骤2] 加载测试集并随机选择图像")
    try:
        image_paths = load_all_test_images()
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    random.seed(RANDOM_SEED)
    img_path = random.choice(image_paths)
    print(f"  选择图像: {img_path.name}")
    
    # 读取图像
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 保存原图RGB（用于显示）
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Stage-1处理
    print("\n[步骤3] Stage-1处理（复原+分割）")
    restored_rgb, mask = run_stage1(model, img_bgr)
    mask_pp = post_process_mask(mask)
    
    # 提取梯度流特征
    print("\n[步骤4] 提取梯度流特征")
    grad_feature, grad_maps = extract_gradient_features(restored_rgb)
    
    # 提取语义边缘
    print("\n[步骤5] 提取语义边缘")
    sem_edges = extract_semantic_edges(mask_pp)
    
    # 计算正弦图
    print("\n[步骤6] 计算正弦图")
    grad_sino = compute_sinogram(grad_feature)
    sem_sino = compute_sinogram(sem_edges)
    
    # 融合正弦图（简单加权融合示意）
    fusion_sino = 0.7 * grad_sino + 0.3 * sem_sino
    
    # 生成图5-5
    print("\n[步骤7] 生成图5-5")
    generate_figure(original_rgb, grad_feature, grad_maps, sem_edges, 
                   grad_sino, sem_sino, fusion_sino, output_fig)
    print(f"  ✓ 已保存: {output_fig}")
    
    print("\n" + "=" * 70)
    print("[完成]")
    print("=" * 70)


if __name__ == "__main__":
    main()
