#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-2 语义流边缘提取流程图 - 批处理MU-SID测试集
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import letterbox_rgb_u8

# ============================================================================
# 路径配置
# ============================================================================
DATASET_ROOT = PROJECT_ROOT / "Hashmani's Dataset" / "MU-SID"
GT_CSV = PROJECT_ROOT / "splits_musid" / "GroundTruth_test.csv"
UNET_WEIGHTS = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"  # 请根据实际修改
DCE_WEIGHTS = PROJECT_ROOT / "weights" / "Epoch99.pth"

# ============================================================================
# 算法参数
# ============================================================================
threshold = 0.5        # 二值化阈值
ksize = 7              # 形态学结构元素大小（奇数）
edge_dilate_iter = 2   # 语义边缘膨胀次数
prob_channel = 0       # 使用第0通道（天空）或第1通道（海面）

IMG_SIZE = 1024        # UNet输入尺寸
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 输出配置
# ============================================================================
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)


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
    
    # 去重
    image_stems = sorted(set(image_stems))
    
    image_paths = []
    for stem in image_stems:
        img_path = DATASET_ROOT / f"{stem}.JPG"
        if img_path.exists():
            image_paths.append(img_path)
    
    print(f"[INFO] 找到 {len(image_paths)} 张测试集图像")
    return sorted(image_paths)


def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组 (H×W×C)"""
    img = tensor.squeeze().detach().cpu().float().clamp(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def get_prob_map_from_unet(model, img_bgr):
    """
    使用UNet模型推理得到概率图
    
    Args:
        model: UNet模型
        img_bgr: BGR图像
    
    Returns:
        prob: float32数组，范围[0,1]，形状H×W×2（通道0=天空，通道1=海面）
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_resized, meta = letterbox_rgb_u8(rgb, IMG_SIZE, pad_value=0)
    
    # 转换为tensor
    inp_tensor = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0)
    inp_tensor = inp_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # UNet推理
    with torch.no_grad():
        _, seg_logits, _, _, _ = model(inp_tensor, enable_restoration=False, enable_segmentation=True)
        prob_tensor = torch.softmax(seg_logits, dim=1)  # (1, 2, H, W)
    
    # 转换为numpy
    prob = tensor_to_numpy(prob_tensor)  # (H, W, 2)
    
    # 裁剪到原始ROI（去掉padding）
    h_orig, w_orig = img_bgr.shape[:2]
    pad_top = int(meta['pad_top'])
    pad_left = int(meta['pad_left'])
    new_h = int(meta['new_h'])
    new_w = int(meta['new_w'])
    
    prob_roi = prob[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    
    # 调整回原始尺寸
    prob_resized = cv2.resize(prob_roi, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    return prob_resized.astype(np.float32)


def semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter):
    """
    语义边缘提取流程
    
    Args:
        prob: 概率图 H×W，float32，范围[0,1]
        threshold: 二值化阈值
        ksize: 形态学核大小
        edge_dilate_iter: 语义边缘膨胀次数
    
    Returns:
        M_bin: 二值化掩码（uint8，0/1）
        G_morph: 形态学梯度（uint8，0/1）
        M_sem: 语义边缘（uint8，0/1）
    """
    # 1) 二值化
    M_bin = (prob >= threshold).astype(np.uint8)
    
    # 2) 形态学梯度
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil = cv2.dilate(M_bin, kernel, iterations=1)
    ero = cv2.erode(M_bin, kernel, iterations=1)
    G_morph = (dil - ero).astype(np.uint8)
    
    # 3) 语义边缘（轻微膨胀）
    M_sem = cv2.dilate(G_morph, kernel, iterations=edge_dilate_iter)
    
    return M_bin, G_morph, M_sem


def get_label_color(img_patch):
    """
    根据图像块亮度确定标注文字和描边颜色
    
    Args:
        img_patch: 小图像块（用于判断亮度）
    
    Returns:
        (text_color, stroke_color): 文字颜色和描边颜色
    """
    mean_val = np.mean(img_patch)
    if mean_val > 0.5:
        return 'black', 'white'
    else:
        return 'white', 'black'


def generate_figure(prob, M_bin, G_morph, M_sem, output_path: Path):
    """
    生成1行4列的流程图
    
    Args:
        prob: 概率图
        M_bin: 二值化掩码
        G_morph: 形态学梯度
        M_sem: 语义边缘
        output_path: 输出路径
    """
    # 字体设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.weight'] = 'normal'
    
    # 创建子图
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.03, hspace=0.0, 
                        left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # 子图数据和标签
    images = [prob, M_bin, G_morph, M_sem]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    
    h, w = prob.shape[:2]
    patch_h = int(h * 0.1)
    patch_w = int(w * 0.1)
    
    for ax, img, label in zip(axes, images, labels):
        # 显示图像（灰度）
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_axis_off()
        
        # 计算左下角区域亮度
        patch = img[-patch_h:, :patch_w]
        text_color, stroke_color = get_label_color(patch)
        
        # 添加标注
        txt = ax.text(
            0.02, 0.02, label,
            transform=ax.transAxes,
            fontsize=11,
            color=text_color,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=1, foreground=stroke_color),
            pe.Normal()
        ])
    
    # 保存
    plt.savefig(str(output_path), dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def process_single_image(model, img_path: Path, output_path: Path):
    """
    处理单张图像并生成语义边缘提取流程图
    
    Args:
        model: UNet模型
        img_path: 输入图像路径
        output_path: 输出图像路径
    """
    # 1) 读取图像
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[错误] 无法读取图像: {img_path}")
        return False
    
    # 2) UNet推理得到概率图
    prob_map = get_prob_map_from_unet(model, img_bgr)  # (H, W, 2)
    prob = prob_map[:, :, prob_channel]  # 取指定通道
    prob = np.clip(prob, 0, 1)
    
    # 3) 执行语义边缘提取流程
    M_bin, G_morph, M_sem = semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter)
    
    # 4) 生成流程图
    generate_figure(prob, M_bin, G_morph, M_sem, output_path)
    
    return True


def main():
    print("=" * 70)
    print("图5-2 语义流边缘提取流程图 - 批处理MU-SID测试集")
    print("=" * 70)
    
    # 1) 检查UNet权重
    if not UNET_WEIGHTS.exists():
        print(f"[错误] UNet权重文件不存在: {UNET_WEIGHTS}")
        print("请修改脚本中的 UNET_WEIGHTS 路径")
        return
    
    # 2) 加载UNet模型
    print(f"\n[步骤1] 加载UNet模型")
    print(f"  权重: {UNET_WEIGHTS}")
    print(f"  设备: {DEVICE}")
    
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=str(DCE_WEIGHTS))
    model = model.to(DEVICE)
    
    try:
        state = torch.load(str(UNET_WEIGHTS), map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return
    
    model.eval()
    
    # 3) 加载测试集图像列表
    print(f"\n[步骤2] 加载测试集图像列表")
    try:
        image_paths = load_all_test_images()
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    if len(image_paths) == 0:
        print("  ✗ 没有找到测试集图像")
        return
    
    # 4) 批处理所有图像
    print(f"\n[步骤3] 批处理 {len(image_paths)} 张图像")
    print(f"  参数: threshold={threshold}, ksize={ksize}, edge_dilate_iter={edge_dilate_iter}, channel={prob_channel}")
    print(f"  输出目录: {output_dir}")
    
    success_count = 0
    for idx, img_path in enumerate(image_paths, 1):
        img_stem = img_path.stem
        output_path = output_dir / f"{img_stem}_sem_edge.png"
        
        print(f"  [{idx}/{len(image_paths)}] {img_stem} ... ", end="", flush=True)
        
        try:
            if process_single_image(model, img_path, output_path):
                success_count += 1
                print("✓")
            else:
                print("✗")
        except Exception as e:
            print(f"✗ {e}")
    
    # 5) 完成
    print("\n" + "=" * 70)
    print(f"[完成] 成功处理 {success_count}/{len(image_paths)} 张图像")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
