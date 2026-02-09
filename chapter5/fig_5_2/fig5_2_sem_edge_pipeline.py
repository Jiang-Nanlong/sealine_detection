#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-2 语义流边缘提取流程图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

# ============================================================================
# 输入配置（请修改路径）
# ============================================================================
prob_path = "PATH/TO/prob.png"  # 或 .npy
prob_channel = 0  # 如果输入是H×W×2，则取第几通道（0或1）；单通道时忽略

# ============================================================================
# 算法参数
# ============================================================================
threshold = 0.5        # 二值化阈值
ksize = 7              # 形态学结构元素大小（奇数）
edge_dilate_iter = 2   # 语义边缘膨胀次数

# ============================================================================
# 输出配置
# ============================================================================
output_dir = Path(__file__).parent
output_fig = output_dir / "fig5_2_sem_edge_pipeline.png"

# 可选：保存中间结果用于调试
SAVE_DEBUG = True
debug_prob = output_dir / "prob_vis.png"
debug_bin = output_dir / "mask_bin.png"
debug_morph = output_dir / "morph_grad.png"
debug_sem = output_dir / "mask_sem.png"


def load_prob_map(path: Path, channel: int = 0):
    """
    加载概率图
    
    Args:
        path: 概率图路径（.npy 或图像文件）
        channel: 如果是多通道，取第几通道
    
    Returns:
        prob: float32数组，范围[0,1]，形状H×W
    """
    if not path.exists():
        raise FileNotFoundError(f"概率图不存在: {path}")
    
    if path.suffix == '.npy':
        prob = np.load(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        if img.dtype == np.uint8:
            prob = img.astype(np.float32) / 255.0
        else:
            prob = img.astype(np.float32)
    
    # 处理维度
    if prob.ndim == 3:
        if prob.shape[2] == 2:
            prob = prob[:, :, channel]
        elif prob.shape[2] == 1:
            prob = prob[:, :, 0]
        else:
            raise ValueError(f"不支持的通道数: {prob.shape[2]}，期望1或2")
    elif prob.ndim == 2:
        pass
    else:
        raise ValueError(f"不支持的维度: {prob.ndim}，期望2或3")
    
    # 确保范围[0,1]
    prob = np.clip(prob, 0, 1)
    return prob.astype(np.float32)


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
    print(f"[完成] 已保存图5-2: {output_path}")


def main():
    # 检查输入路径
    prob_path_obj = Path(prob_path)
    if not prob_path_obj.exists() or str(prob_path_obj) == "PATH/TO/prob.png":
        print("[警告] 请先修改 prob_path 为实际的概率图路径")
        print(f"当前路径: {prob_path}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) 加载概率图
    print(f"[步骤1] 加载概率图: {prob_path_obj}")
    prob = load_prob_map(prob_path_obj, prob_channel)
    print(f"  形状: {prob.shape}, 范围: [{prob.min():.3f}, {prob.max():.3f}]")
    
    # 2) 执行语义边缘提取流程
    print(f"[步骤2] 执行语义边缘提取")
    print(f"  参数: threshold={threshold}, ksize={ksize}, edge_dilate_iter={edge_dilate_iter}")
    M_bin, G_morph, M_sem = semantic_edge_pipeline(prob, threshold, ksize, edge_dilate_iter)
    
    # 3) 生成图5-2
    print(f"[步骤3] 生成图5-2流程图")
    generate_figure(prob, M_bin, G_morph, M_sem, output_fig)
    
    # 4) 可选：保存调试图像
    if SAVE_DEBUG:
        print(f"[步骤4] 保存调试图像")
        cv2.imwrite(str(debug_prob), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(debug_bin), M_bin * 255)
        cv2.imwrite(str(debug_morph), G_morph * 255)
        cv2.imwrite(str(debug_sem), M_sem * 255)
        print(f"  prob_vis.png: {debug_prob}")
        print(f"  mask_bin.png: {debug_bin}")
        print(f"  morph_grad.png: {debug_morph}")
        print(f"  mask_sem.png: {debug_sem}")
    
    print("\n[全部完成]")


if __name__ == "__main__":
    main()
