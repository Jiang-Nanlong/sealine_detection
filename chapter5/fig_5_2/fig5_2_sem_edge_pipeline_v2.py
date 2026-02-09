#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-2 语义流边缘提取流程图（方案B，1×5）- 批处理MU-SID测试集
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
UNET_WEIGHTS = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"
DCE_WEIGHTS = PROJECT_ROOT / "weights" / "Epoch99.pth"

# ============================================================================
# 算法参数
# ============================================================================
threshold = 0.5
ksize = 7
edge_dilate_iter = 2
prob_channel = 0
overlay_color = (255, 0, 0)  # BGR: 红色
overlay_alpha = 0.85

IMG_SIZE = 1024
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
    """使用UNet模型推理得到概率图"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_resized, meta = letterbox_rgb_u8(rgb, IMG_SIZE, pad_value=0)
    
    inp_tensor = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0)
    inp_tensor = inp_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, seg_logits, _ = model(inp_tensor, enable_restoration=False, enable_segmentation=True)
        prob_tensor = torch.softmax(seg_logits, dim=1)
    
    prob = tensor_to_numpy(prob_tensor)
    
    h_orig, w_orig = img_bgr.shape[:2]
    pad_top = int(meta['pad_top'])
    pad_left = int(meta['pad_left'])
    new_h = int(meta['new_h'])
    new_w = int(meta['new_w'])
    
    prob_roi = prob[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    prob_resized = cv2.resize(prob_roi, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    return prob_resized.astype(np.float32)


def robust_norm(x, lo=1, hi=99):
    """稳健归一化：基于百分位数"""
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b <= a:
        b = a + 1e-6
    y = np.clip((x - a) / (b - a), 0, 1)
    return y


def semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter):
    """语义边缘提取流程"""
    prob = np.nan_to_num(prob, nan=0.0)
    prob = np.clip(prob, 0, 1)
    
    M_bin = (prob >= threshold).astype(np.uint8)
    
    if ksize % 2 == 0:
        ksize += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil = cv2.dilate(M_bin, kernel, iterations=1)
    ero = cv2.erode(M_bin, kernel, iterations=1)
    G_morph = (dil - ero).astype(np.uint8)
    
    M_sem = cv2.dilate(G_morph, kernel, iterations=edge_dilate_iter).astype(np.uint8)
    
    return M_bin, G_morph, M_sem


def create_overlay(img_rgb, M_sem, color=(255, 0, 0), alpha=0.85):
    """在原图上叠加语义边缘"""
    overlay = img_rgb.copy()
    mask = M_sem > 0
    overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay


def get_label_color(img_patch):
    """根据图像块亮度确定标注文字和描边颜色"""
    mean_val = np.mean(img_patch)
    if mean_val < 0.4:
        return 'white', 'black'
    else:
        return 'black', 'white'


def generate_figure(img_rgb, prob, M_bin, G_morph, M_sem, output_path: Path):
    """生成1行5列的流程图"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.weight'] = 'normal'
    
    prob_vis = robust_norm(prob)
    overlay = create_overlay(img_rgb, M_sem, color=overlay_color, alpha=overlay_alpha)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    plt.subplots_adjust(wspace=0.025, hspace=0.0, 
                        left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    images = [img_rgb / 255.0 if img_rgb.dtype == np.uint8 else img_rgb,
              prob_vis, M_bin, G_morph, 
              overlay / 255.0 if overlay.dtype == np.uint8 else overlay]
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    cmaps = [None, "gray", "gray", "gray", None]
    
    h, w = prob.shape[:2]
    patch_h = int(h * 0.1)
    patch_w = int(w * 0.1)
    
    for ax, img, label, cmap in zip(axes, images, labels, cmaps):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        
        if len(img.shape) == 3:
            patch = np.mean(img[-patch_h:, :patch_w, :])
        else:
            patch = np.mean(img[-patch_h:, :patch_w])
        
        text_color, stroke_color = get_label_color(np.array([[patch]]))
        
        txt = ax.text(
            0.02, 0.02, label,
            transform=ax.transAxes,
            fontsize=11,
            color=text_color,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=1.5, foreground=stroke_color),
            pe.Normal()
        ])
    
    plt.savefig(str(output_path), dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def process_single_image(model, img_path: Path, output_path: Path):
    """处理单张图像并生成语义边缘提取流程图"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[错误] 无法读取图像: {img_path}")
        return False
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    prob_map = get_prob_map_from_unet(model, img_bgr)
    prob = prob_map[:, :, prob_channel]
    
    h_orig, w_orig = img_rgb.shape[:2]
    if prob.shape[:2] != (h_orig, w_orig):
        prob = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    M_bin, G_morph, M_sem = semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter)
    
    generate_figure(img_rgb, prob, M_bin, G_morph, M_sem, output_path)
    
    return True


def main():
    print("=" * 70)
    print("图5-2 语义流边缘提取流程图（方案B，1×5）- 批处理MU-SID测试集")
    print("=" * 70)
    
    if not UNET_WEIGHTS.exists():
        print(f"[错误] UNet权重文件不存在: {UNET_WEIGHTS}")
        return
    
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
    
    print(f"\n[步骤2] 加载测试集图像列表")
    try:
        image_paths = load_all_test_images()
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    if len(image_paths) == 0:
        print("  ✗ 没有找到测试集图像")
        return
    
    print(f"\n[步骤3] 批处理 {len(image_paths)} 张图像")
    print(f"  参数: threshold={threshold}, ksize={ksize}, edge_dilate_iter={edge_dilate_iter}, channel={prob_channel}")
    print(f"  输出目录: {output_dir}")
    
    success_count = 0
    for idx, img_path in enumerate(image_paths, 1):
        img_stem = img_path.stem
        output_path = output_dir / f"{img_stem}_sem_edge_1x5.png"
        
        print(f"  [{idx}/{len(image_paths)}] {img_stem} ... ", end="", flush=True)
        
        try:
            if process_single_image(model, img_path, output_path):
                success_count += 1
                print("✓")
            else:
                print("✗")
        except Exception as e:
            print(f"✗ {e}")
    
    print("\n" + "=" * 70)
    print(f"[完成] 成功处理 {success_count}/{len(image_paths)} 张图像")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
