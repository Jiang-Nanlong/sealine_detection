#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图5-1 差分梯度纹理抑制效果对比
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ========================
# A. 输入与输出配置
# ========================
restored_img_path = "PATH/TO/restored.png"

# 可选：真值海天线端点（像素坐标），若没有则设为 None
gt_line = None  # 例如: ((x1, y1), (x2, y2))

# 输出文件
output_path = "fig5_1_diff_grad_comparison.png"

# 调试输出（可选）
debug_sobel_path = "sobel_mag.png"
debug_diffgrad_path = "diff_grad.png"

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


def main():
    # ========================
    # 1) 读取复原图
    # ========================
    if not os.path.exists(restored_img_path):
        raise FileNotFoundError(f"输入图像不存在: {restored_img_path}")
    
    img_bgr = cv2.imread(restored_img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {restored_img_path}")
    
    # ========================
    # 2) 灰度化与归一化
    # ========================
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    
    # ========================
    # 3) Sobel 梯度
    # ========================
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # ========================
    # 4) 普通梯度幅值图
    # ========================
    M_sobel = np.sqrt(Gx * Gx + Gy * Gy)
    
    # ========================
    # 5) 差分梯度纹理抑制
    # ========================
    lambda_ratio = 0.6
    M_grad = np.maximum(np.abs(Gy) - lambda_ratio * np.abs(Gx), 0)
    tau = np.percentile(M_grad, 99)
    M_grad = np.clip(M_grad, 0, tau)
    
    # ========================
    # 6) 显示归一化
    # ========================
    M_sobel_norm = robust_norm(M_sobel, lo=1, hi=99)
    M_grad_norm = robust_norm(M_grad, lo=1, hi=99)
    
    # ========================
    # 调试输出（可选）
    # ========================
    cv2.imwrite(debug_sobel_path, (M_sobel_norm * 255).astype(np.uint8))
    cv2.imwrite(debug_diffgrad_path, (M_grad_norm * 255).astype(np.uint8))
    
    # ========================
    # C. 排版与绘图
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
    
    # ========================
    # 叠加真值线（可选）
    # ========================
    if gt_line is not None:
        (x1, y1), (x2, y2) = gt_line
        # 在 (a) 和 (c) 上叠加
        for idx in [0, 2]:
            axes[idx].plot(
                [x1, x2], [y1, y2],
                color="red",
                linewidth=2,
                alpha=0.9
            )
    
    # ========================
    # D. 保存
    # ========================
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"已保存: {output_path}")
    print(f"调试输出: {debug_sobel_path}, {debug_diffgrad_path}")


if __name__ == "__main__":
    main()
