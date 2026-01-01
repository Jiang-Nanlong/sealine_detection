# create_finetune_dataset.py
import cv2
import numpy as np
import os
import random
from tqdm import tqdm


# --- 辅助函数 (与脚本一中的函数完全相同) ---
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
    t = np.clip(t, 0.1, 1.0)[:, :, np.newaxis]
    A = 1 - 0.3 * random.random()
    hazy_image_f = image_f * t + A * (1 - t)
    return (hazy_image_f * 255).astype(np.uint8)


def darken_image(image, gamma_range=(1.5, 2.5)):
    """使用Gamma校正来模拟低光照效果。"""
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def generate_finetune_dataset(source_dir, input_output_dir, target_output_dir):
    """
    生成用于联合微调的数据集。
    输入是低光照+雨/雾，目标是干净的图。
    """
    print(f"--- Starting Finetune Dataset Generation ---")
    print(f"Source clean images from: '{source_dir}'")
    os.makedirs(input_output_dir, exist_ok=True)
    os.makedirs(target_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Generating Finetune data"):
        source_path = os.path.join(source_dir, filename)
        clean_image = cv2.imread(source_path)
        if clean_image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # 1. 随机添加雨或雾
        degraded_image = clean_image.copy()
        if random.random() < 0.6:
            degraded_image = add_haze(degraded_image, haze_intensity=random.uniform(0.4, 0.8))
        if random.random() < 0.6:
            degraded_image = add_rain(degraded_image, rain_amount=random.uniform(0.2, 0.6))
        if np.array_equal(degraded_image, clean_image):
            degraded_image = add_haze(degraded_image)

        # 2. 在添加了退化的基础上，模拟低光照
        low_light_degraded_image = darken_image(degraded_image)

        # 3. 保存图像对
        cv2.imwrite(os.path.join(input_output_dir, filename), low_light_degraded_image)
        cv2.imwrite(os.path.join(target_output_dir, filename), clean_image)

    print("\nFinetune dataset generation complete!")
    print(f"Input images (low-light+degraded) saved to: '{input_output_dir}'")
    print(f"Target images (normal-light+clean) saved to: '{target_output_dir}'")


if __name__ == '__main__':
    # --- 配置区域 ---
    # !!! 1. 修改你的“原材料”数据集路径 (应与脚本一相同) !!!
    SOURCE_CLEAN_IMAGES_DIR = r"D:\dataset\clean_normal_light"

    # !!! 2. 定义微调数据集的输出路径 !!!
    OUTPUT_FINETUNE_INPUT_DIR = r"D:\dataset_generated\finetune\input_low_light_degraded"
    OUTPUT_FINETUNE_TARGET_DIR = r"D:\dataset_generated\finetune\target_normal_light_clean"

    # --- 执行 ---
    if not os.path.isdir(SOURCE_CLEAN_IMAGES_DIR):
        print(f"Error: Source directory not found at '{SOURCE_CLEAN_IMAGES_DIR}'")
    else:
        generate_finetune_dataset(SOURCE_CLEAN_IMAGES_DIR, OUTPUT_FINETUNE_INPUT_DIR, OUTPUT_FINETUNE_TARGET_DIR)