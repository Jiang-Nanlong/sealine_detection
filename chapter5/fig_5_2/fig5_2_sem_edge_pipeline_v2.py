#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-2 语义流边缘提取流程图（2行×5列，两组样例）
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
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import letterbox_rgb_u8

DATASET_ROOT = PROJECT_ROOT / "Hashmani's Dataset" / "MU-SID"
GT_CSV = PROJECT_ROOT / "splits_musid" / "GroundTruth_test.csv"
UNET_WEIGHTS = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"
DCE_WEIGHTS = PROJECT_ROOT / "weights" / "Epoch99.pth"

threshold = 0.5
ksize = 7
edge_dilate_iter = 2
prob_channel = 0
border_margin = 6
contour_color = (255, 0, 0)
contour_thickness = 2

IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)
output_fig = output_dir / "fig5_2_sem_edge_pipeline_2x5.png"


def load_all_test_images():
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
    img = tensor.squeeze().detach().cpu().float().clamp(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def get_prob_map_from_unet(model, img_bgr):
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


def remove_border_connected_components(M_sem):
    """删除与图像边界相连的连通域"""
    H, W = M_sem.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(M_sem, connectivity=8)
    
    M_clean = np.zeros_like(M_sem)
    
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        
        touches_border = np.any(xs == 0) or np.any(xs == W-1) or np.any(ys == 0) or np.any(ys == H-1)
        
        if not touches_border:
            M_clean[mask > 0] = 1
    
    return M_clean


def semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter, border_margin):
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
    
    H, W = M_sem.shape
    M_sem[:border_margin, :] = 0
    M_sem[-border_margin:, :] = 0
    M_sem[:, :border_margin] = 0
    M_sem[:, -border_margin:] = 0
    
    M_sem_clean = remove_border_connected_components(M_sem)
    
    return M_bin, G_morph, M_sem_clean


def create_contour_overlay(img_rgb, M_sem_clean, color=(255, 0, 0), thickness=2):
    """在原图上叠加语义边缘轮廓"""
    overlay = img_rgb.copy()
    contours, _ = cv2.findContours(M_sem_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def process_image_data(model, img_path):
    """处理单张图像"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    prob_map = get_prob_map_from_unet(model, img_bgr)
    prob = prob_map[:, :, prob_channel]
    
    h_orig, w_orig = img_rgb.shape[:2]
    if prob.shape[:2] != (h_orig, w_orig):
        prob = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    M_bin, G_morph, M_sem_clean = semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter, border_margin)
    
    prob_vis = robust_norm(prob)
    overlay = create_contour_overlay(img_rgb, M_sem_clean, color=contour_color, thickness=contour_thickness)
    
    return img_rgb, prob_vis, M_bin, G_morph, overlay


def generate_figure(data_list, output_path):
    """生成2行5列的流程图（无标注）"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.weight'] = 'normal'
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.025, hspace=0.04, 
                        left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    for row_idx, (img_rgb, prob_vis, M_bin, G_morph, overlay) in enumerate(data_list):
        images = [img_rgb / 255.0 if img_rgb.dtype == np.uint8 else img_rgb,
                  prob_vis,
                  M_bin,
                  G_morph,
                  overlay / 255.0 if overlay.dtype == np.uint8 else overlay]
        cmaps = [None, "gray", "gray", "gray", None]
        
        for col_idx, (img, cmap) in enumerate(zip(images, cmaps)):
            ax = axes[row_idx, col_idx]
            
            if cmap is None:
                ax.imshow(img)
            elif cmap == "gray" and img.dtype == np.uint8 and np.max(img) <= 1:
                ax.imshow(img * 255, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            
            ax.set_axis_off()
    
    plt.savefig(str(output_path), dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    print("=" * 70)
    print("图5-2 语义流边缘提取流程图（2行×5列，无标注）")
    print("=" * 70)
    
    if not UNET_WEIGHTS.exists():
        print(f"[错误] UNet权重文件不存在: {UNET_WEIGHTS}")
        return
    
    print(f"\n[步骤1] 加载UNet模型")
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
    
    print(f"\n[步骤2] 加载测试集并随机选择2张图像")
    try:
        image_paths = load_all_test_images()
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    if len(image_paths) < 2:
        print("  ✗ 测试集图像数量不足")
        return
    
    random.seed(RANDOM_SEED)
    selected_paths = random.sample(image_paths, 2)
    
    print(f"  样例1: {selected_paths[0].name}")
    print(f"  样例2: {selected_paths[1].name}")
    
    print(f"\n[步骤3] 处理图像并生成流程图")
    data_list = []
    for idx, img_path in enumerate(selected_paths, 1):
        print(f"  处理样例{idx} ... ", end="", flush=True)
        try:
            data = process_image_data(model, img_path)
            data_list.append(data)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            return
    
    print(f"\n[步骤4] 生成2×5布局图")
    generate_figure(data_list, output_fig)
    print(f"  ✓ 已保存: {output_fig}")
    
    print("\n" + "=" * 70)
    print("[完成]")
    print("=" * 70)


if __name__ == "__main__":
    main()