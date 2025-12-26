import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset


def check_data_alignment():
    # 配置你的路径
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"
    RESIZE_H = 362
    RESIZE_W = 180

    print("正在加载数据集 (只读几张)...")
    dataset = HorizonFusionDataset(CSV_PATH, IMG_DIR, resize_h=RESIZE_H, resize_w=RESIZE_W)

    # 随机看 5 张图
    indices = np.random.choice(len(dataset), 5, replace=False)
    # 或者手动指定几张比较清晰的图的索引
    # indices = [0, 10, 20, 30]

    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(indices):
        # 获取输入 (3, H, W) 和 标签 (2,)
        input_tensor, label = dataset[idx]

        # 取出其中一个尺度的 Sinogram (比如 Scale 2)
        # 形状是 [362, 180]
        sino_img = input_tensor[1].numpy()

        # 获取标签坐标 (归一化的 0-1)
        gt_rho_norm = label[0].item()
        gt_theta_norm = label[1].item()

        # 还原回 Sinogram 上的像素坐标
        # x轴是 Theta (0-180)
        # y轴是 Rho (0-362)
        target_x = gt_theta_norm * RESIZE_W
        # 注意：Rho 的还原要看你的 Dataset 是怎么归一化的
        # 假设是 (rho / max_rho + 1) / 2 这种逻辑
        # 这里直接映射到 0~Height
        target_y = gt_rho_norm * RESIZE_H

        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(sino_img, cmap='jet', aspect='auto')

        # 画出红点 (Ground Truth)
        ax.plot(target_x, target_y, 'rx', markersize=12, markeredgewidth=2, label='GT Label')

        # 标出最亮点 (Max Value) - 这是 Radon 算出来的“事实”
        # 简单的找最大值位置
        max_y, max_x = np.unravel_index(np.argmax(sino_img), sino_img.shape)
        ax.plot(max_x, max_y, 'go', markersize=8, markerfacecolor='none', label='Brightest')

        ax.set_title(f"Sample {idx}")
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_data_alignment()