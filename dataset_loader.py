import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import glob


def synthesize_rain_fog(image_rgb_u8):
    """
    更真实的物理感知退化函数
    """
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # 允许混合退化 (比如又下雨又起雾)
    is_rain = random.random() < 0.4
    is_fog = random.random() < 0.4
    is_dark = random.random() < 0.3

    # 至少要有一种退化，否则就是原图 (保留一小部分原图作为Identity)
    if not (is_rain or is_fog or is_dark) and random.random() < 0.8:
        is_fog = True  # 强行加雾

    if is_rain:
        severity = random.uniform(0.3, 0.9)
        # 雨丝
        streak = np.zeros((h, w, 3), dtype=np.float32)
        n_lines = int(800 * severity)
        angle = random.uniform(-15, 15)
        tan_a = np.tan(np.deg2rad(angle))
        for _ in range(n_lines):
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            length = random.randint(15, 45)
            x1 = int(x0 + length * tan_a)
            y1 = int(y0 + length)
            cv2.line(streak, (x0, y0), (int(x1), int(y1)), (0.8, 0.8, 0.8), 1)
        streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=0.5, sigmaY=0.5)
        img = np.clip(img + 0.4 * streak, 0, 1)

    if is_fog:
        # 修复: Airlight 和 浓度 更加随机
        severity = random.uniform(0.2, 0.9)
        airlight = random.uniform(0.6, 0.95)  # 雾不一定是死白，也可能是灰

        ys = np.linspace(0, 1, h)[:, None]
        # 随机雾梯度方向 (上->下 或 下->上)
        if random.random() > 0.5:
            base_map = np.tile(ys, (1, w))
        else:
            base_map = np.tile(1 - ys, (1, w))

        # 柏林噪声太慢，用高斯模糊模拟不均匀分布
        noise_map = np.random.rand(h, w)
        noise_map = cv2.GaussianBlur(noise_map, (0, 0), sigmaX=30, sigmaY=30)

        fog_map = base_map * 0.7 + noise_map * 0.3
        fog_map = fog_map * severity * random.uniform(0.8, 1.2)
        fog_map = np.clip(fog_map[:, :, None], 0, 1)

        img = img * (1 - fog_map) + airlight * fog_map

    if is_dark:
        factor = random.uniform(0.25, 0.6)
        img = img * factor

    # 传感器噪声
    noise = np.random.randn(h, w, 3) * 0.01
    img = np.clip(img + noise, 0, 1)

    return img.astype(np.float32)


# ... (SimpleFolderDataset 和 HorizonImageDataset 的类定义与之前相同，直接复制即可) ...
# 注意：务必确保 getitem 返回的是 input_tensor 和 target_clean (0-1 Float)
# 这里为了节省篇幅，假设你已经保留了上一次我给你的 Dataset 类结构，
# 只需要替换上面的 synthesize_rain_fog 函数即可。
# 下面我还是完整给出 HorizonImageDataset 以防万一

class HorizonImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=384, mode='joint', ignore_band=10):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        self.ignore_band = ignore_band

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])
        try:
            x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        except:
            x1, y1, x2, y2 = 0, 0, 0, 0

        path = os.path.join(self.img_dir, img_name)
        possible_exts = ['', '.JPG', '.jpg', '.png', '.jpeg']
        image = None
        for ext in possible_exts:
            if os.path.exists(path + ext):
                image = cv2.imread(path + ext)
                break

        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.img_size, self.img_size))

        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        sx, sy = self.img_size / orig_w, self.img_size / orig_h
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        pts = np.array([[0, 0], [self.img_size, 0], [self.img_size, p2[1]], [0, p1[1]]], np.int32)
        cv2.fillPoly(mask, [pts], 1)
        if self.ignore_band > 0:
            cv2.line(mask, p1, p2, 255, self.ignore_band)

        mask_tensor = torch.from_numpy(mask).long()
        clean_np = image_resized.copy()
        target_clean = torch.from_numpy(clean_np.astype(np.float32) / 255.0).permute(2, 0, 1)

        if self.mode == 'segmentation':
            return target_clean, mask_tensor

        degraded_np = synthesize_rain_fog(clean_np)
        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1)

        return input_tensor, target_clean, mask_tensor


class SimpleFolderDataset(Dataset):
    def __init__(self, img_dir, img_size=384):
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_paths = glob.glob(os.path.join(img_dir, "*.[jJ][pP]*[gG]")) + \
                         glob.glob(os.path.join(img_dir, "*.png"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        target_clean = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        degraded_np = synthesize_rain_fog(image_resized)
        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1)
        return input_tensor, target_clean