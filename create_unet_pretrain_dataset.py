# create_unet_pretrain_dataset.py
import cv2
import numpy as np
import os
import random
from tqdm import tqdm


def add_rain(image, rain_amount=0.3):
    """在图像上添加合成雨痕。"""
    img_copy = image.copy()
    h, w, _ = img_copy.shape
    num_drops = int(h * w * 0.001 * rain_amount)
    min_len, max_len = int(h * 0.02), int(h * 0.08)
    min_width, max_width = 1, int(2 * rain_amount) + 1

    for _ in range(num_drops):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        length = random.randint(min_len, max_len)
        angle = random.randint(-20, 20)
        x2 = x1 + int(length * np.sin(np.pi * angle / 180.0))
        y2 = y1 + int(length * np.cos(np.pi * angle / 180.0))
        width = random.randint(min_width, max_width)
        brightness = random.randint(200, 255)
        cv2.line(img_copy, (x1, y1), (x2, y2), (brightness, brightness, brightness), width)

    rainy_image = cv2.blur(img_copy, (3, 3))
    return cv2.addWeighted(image, 0.9, rainy_image, 0.1, 0)


def add_haze(image, haze_intensity=0.6):
    """在图像上添加合成雾。"""
    image_f = image.astype(np.float32) / 255.0
    h, w, _ = image.shape
    noise = np.random.rand(h, w)
    t = 1 - (haze_intensity * cv2.GaussianBlur(noise, (101, 101), 50))
    t = np.clip(t, 0.1, 1.0)
    t = t[:, :, np.newaxis]
    A = 1 - 0.3 * random.random()
    hazy_image_f = image_f * t + A * (1 - t)
    return (hazy_image_f * 255).astype(np.uint8)


def generate_unet_pretrain_dataset(source_dir, degraded_output_dir, clean_output_dir):
    """
    生成用于U-Net预训练的数据集。
    输入是随机加了雨/雾的图，目标是干净的图。
    """
    print(f"--- Starting U-Net Pre-training Dataset Generation ---")
    print(f"Source clean images from: '{source_dir}'")
    os.makedirs(degraded_output_dir, exist_ok=True)
    os.makedirs(clean_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Generating U-Net data"):
        source_path = os.path.join(source_dir, filename)
        clean_image = cv2.imread(source_path)
        if clean_image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        degraded_image = clean_image.copy()

        # 随机决定是否加雾
        if random.random() < 0.6:  # 60%概率加雾
            degraded_image = add_haze(degraded_image, haze_intensity=random.uniform(0.4, 0.8))

        # 随机决定是否加雨
        if random.random() < 0.6:  # 60%概率加雨
            degraded_image = add_rain(degraded_image, rain_amount=random.uniform(0.2, 0.6))

        # 如果一张图片碰巧什么都没加，确保它至少有一种退化（默认加雾）
        if np.array_equal(degraded_image, clean_image):
            degraded_image = add_haze(degraded_image)

        # 保存 degraded_image (作为U-Net的输入) 和 clean_image (作为U-Net的目标)
        cv2.imwrite(os.path.join(degraded_output_dir, filename), degraded_image)
        cv2.imwrite(os.path.join(clean_output_dir, filename), clean_image)

    print("\nU-Net pre-training dataset generation complete!")
    print(f"Degraded input images saved to: '{degraded_output_dir}'")
    print(f"Clean target images saved to:   '{clean_output_dir}'")


if __name__ == '__main__':
    # --- 配置区域 ---
    # !!! 1. 修改你的“原材料”数据集路径 !!!
    SOURCE_CLEAN_IMAGES_DIR = r"D:\dataset\clean_normal_light"

    # !!! 2. 定义U-Net预训练数据集的输出路径 !!!
    OUTPUT_DEGRADED_DIR = r"D:\dataset_generated\unet_pretrain\degraded_input"
    OUTPUT_CLEAN_DIR = r"D:\dataset_generated\unet_pretrain\clean_target"

    # --- 执行 ---
    if not os.path.isdir(SOURCE_CLEAN_IMAGES_DIR):
        print(f"Error: Source directory not found at '{SOURCE_CLEAN_IMAGES_DIR}'")
    else:
        generate_unet_pretrain_dataset(SOURCE_CLEAN_IMAGES_DIR, OUTPUT_DEGRADED_DIR, OUTPUT_CLEAN_DIR)