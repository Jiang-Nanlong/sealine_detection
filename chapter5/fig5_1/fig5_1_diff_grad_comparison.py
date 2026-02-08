#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图5-1 差分梯度纹理抑制效果对比
批量处理 MU-SID 测试集所有图片
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ========================
# 路径配置（自动获取项目根目录）
# ========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # chapter5/fig5_1 -> sealine_detection

# ========================
# A. 输入与输出配置
# ========================
# MU-SID 数据集路径
IMG_DIR = PROJECT_ROOT / "Hashmani's Dataset" / "MU-SID"
GT_CSV = PROJECT_ROOT / "splits_musid" / "GroundTruth_test.csv"

# 输出目录
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 是否显示真值线
SHOW_GT_LINE = True


def load_all_test_images(csv_path: Path):
    """
    从 GroundTruth CSV 文件加载所有测试集图片名和真值。
    CSV格式: image_stem,x1,y1,x2,y2,xm,ym,angle
    返回: [(image_stem, ((x1, y1), (x2, y2))), ...]
    """
    results = []
    if not csv_path.exists():
        print(f"[错误] 真值文件不存在: {csv_path}")
        return results
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    image_stem = parts[0]
                    try:
                        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        gt_line = ((x1, y1), (x2, y2))
                    except ValueError:
                        gt_line = None
                    results.append((image_stem, gt_line))
    except Exception as e:
        print(f"[错误] 读取真值文件失败: {e}")
    
    return results


def load_gt_line(csv_path: Path, image_stem: str):
    """
    从 GroundTruth CSV 文件加载海天线真值。
    CSV格式: image_stem,x1,y1,x2,y2,xm,ym,angle
    返回: ((x1, y1), (x2, y2)) 或 None
    """
    if not csv_path.exists():
        print(f"[警告] 真值文件不存在: {csv_path}")
        return None
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5 and parts[0] == image_stem:
                    x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    return ((x1, y1), (x2, y2))
    except Exception as e:
        print(f"[警告] 读取真值文件失败: {e}")
    
    return None

# ========================
# B. 处理函数
# ========================

def robust_norm(x, lo=1, hi=99):
    """稳健归一化：基于百分位数"""
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b <= a:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - a) / (b - a), 0, 1)
    return y.astype(np.float32)


def get_label_color(img_patch):
    """根据图像区域亮度自动选择标注颜色"""
    mean_brightness = np.mean(img_patch)
    if mean_brightness < 0.4:
        return "white", "black"
    else:
        return "black", "white"


def process_single_image(image_stem: str, gt_line, img_dir: Path, output_dir: Path):
    """
    处理单张图片，生成差分梯度对比图。
    
    Args:
        image_stem: 图片名称（不含扩展名）
        gt_line: 真值线 ((x1, y1), (x2, y2)) 或 None
        img_dir: 图片目录
        output_dir: 输出目录
    
    Returns:
        True 成功, False 失败
    """
    # 查找图片文件
    img_path = None
    for ext in ['.JPG', '.jpg', '.jpeg', '.png']:
        candidate = img_dir / f"{image_stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    
    if img_path is None:
        print(f"  [跳过] 图片不存在: {image_stem}")
        return False
    
    # 读取图像
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  [跳过] 无法读取图像: {img_path}")
        return False
    
    # 灰度化与归一化
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    
    # Sobel 梯度
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # 普通梯度幅值图
    M_sobel = np.sqrt(Gx * Gx + Gy * Gy)
    
    # 差分梯度纹理抑制
    lambda_ratio = 0.6
    M_grad = np.maximum(np.abs(Gy) - lambda_ratio * np.abs(Gx), 0)
    tau = np.percentile(M_grad, 99)
    M_grad = np.clip(M_grad, 0, tau)
    
    # 显示归一化
    M_sobel_norm = robust_norm(M_sobel, lo=1, hi=99)
    M_grad_norm = robust_norm(M_grad, lo=1, hi=99)
    
    # ========================
    # 排版与绘图
    # ========================
    
    # 字体设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'normal'
    
    # 画布尺寸
    width_in = 180 / 25.4  # 180 mm -> inches
    height_in = 65 / 25.4  # 约 65 mm -> inches
    
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in))
    plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # 转换为 RGB 用于显示
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_norm = img_rgb.astype(np.float32) / 255.0
    
    # 准备显示数据
    display_imgs = [
        img_rgb_norm,      # (a) RGB 原图
        M_sobel_norm,      # (b) Sobel 梯度幅值
        M_grad_norm        # (c) 差分梯度
    ]
    labels = ["(a)", "(b)", "(c)"]
    cmaps = [None, "gray", "gray"]  # (a) 用 RGB，(b)(c) 用灰度
    
    h, w = gray.shape[:2]
    patch_h = int(h * 0.1)
    patch_w = int(w * 0.1)
    
    for ax, img_disp, label, cmap in zip(axes, display_imgs, labels, cmaps):
        if cmap is None:
            ax.imshow(img_disp)
        else:
            ax.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        
        # 计算左下角区域亮度以确定标注颜色
        if len(img_disp.shape) == 3:
            patch = np.mean(img_disp[-patch_h:, :patch_w, :])
        else:
            patch = np.mean(img_disp[-patch_h:, :patch_w])
        
        text_color, stroke_color = get_label_color(np.array([[patch]]))
        
        # 添加标注
        txt = ax.text(
            0.02, 0.02, label,
            transform=ax.transAxes,
            fontsize=10,
            color=text_color,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=2, foreground=stroke_color),
            pe.Normal()
        ])
    
    # 叠加真值线（可选）
    if SHOW_GT_LINE and gt_line is not None:
        (x1, y1), (x2, y2) = gt_line
        # 在 (a) 和 (c) 上叠加
        for idx in [0, 2]:
            axes[idx].plot(
                [x1, x2], [y1, y2],
                color="red",
                linewidth=2,
                alpha=0.9
            )
    
    # 保存
    output_path = output_dir / f"{image_stem}_diff_grad.png"
    plt.savefig(str(output_path), dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return True


def main():
    """批量处理 MU-SID 测试集所有图片"""
    print("=" * 60)
    print("图5-1 差分梯度纹理抑制效果对比 - 批量生成")
    print("=" * 60)
    
    # 加载测试集所有图片
    test_images = load_all_test_images(GT_CSV)
    if not test_images:
        print("[错误] 未找到测试集图片列表")
        return
    
    print(f"[信息] 共找到 {len(test_images)} 张测试集图片")
    print(f"[信息] 输出目录: {OUTPUT_DIR}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for i, (image_stem, gt_line) in enumerate(test_images, 1):
        print(f"[{i:3d}/{len(test_images)}] 处理: {image_stem}", end=" ... ")
        
        try:
            if process_single_image(image_stem, gt_line, IMG_DIR, OUTPUT_DIR):
                print("完成")
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"失败: {e}")
            fail_count += 1
    
    print("-" * 60)
    print(f"[完成] 成功: {success_count}, 失败: {fail_count}")
    print(f"[信息] 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
