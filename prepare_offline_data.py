import os
import torch
import numpy as np
from tqdm import tqdm
# 确保 dataset_loader_gradient_radon_cnn.py 在同一目录下
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset


def cache_dataset():
    # === 配置 ===
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"
    SAVE_DIR = r"Hashmani's Dataset/OfflineCache"  # 缓存保存路径

    RESIZE_H = 362
    RESIZE_W = 180
    # ===========

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("正在初始化原始数据集 (包含 GPU Radon 变换)...")
    # 这里会用到你的 GPU 代码，所以速度会比纯 CPU 快很多，但 IO 依然是瓶颈
    dataset = HorizonFusionDataset(CSV_PATH, IMG_DIR, resize_h=RESIZE_H, resize_w=RESIZE_W)

    print(f"开始转换 {len(dataset)} 张图片...")
    print("请耐心等待，这可能需要 30-90 分钟，但只需要跑一次！")

    # 遍历所有数据
    for i in tqdm(range(len(dataset))):
        try:
            # 这一步会执行读取图片、滤波、GPU Radon 等所有耗时操作
            input_tensor, label_tensor = dataset[i]

            # 保存为 .npy (字典格式)
            data_dict = {
                'input': input_tensor.numpy(),  # [3, 362, 180]
                'label': label_tensor.numpy()  # [2]
            }
            # 文件名为索引：0.npy, 1.npy ...
            np.save(os.path.join(SAVE_DIR, f"{i}.npy"), data_dict)

        except Exception as e:
            print(f"Skipping index {i}: {e}")

    print(f"缓存完成！数据已保存至: {SAVE_DIR}")


if __name__ == "__main__":
    cache_dataset()