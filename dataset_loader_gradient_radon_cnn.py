import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from skimage.transform import radon
from gradient_radon import TextureSuppressedMuSCoWERT


class HorizonFusionDataset(Dataset):
    def __init__(self, csv_file, img_dir, resize_h=362, resize_w=180):
        # 修正1: header=None 防止第一行被吃掉
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.resize_h = resize_h
        self.resize_w = resize_w

        print(f"Dataset loaded. Total images: {len(self.data)}")  # 现在应该是 2673

        self.detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
        self.theta_scan = np.linspace(0., 180., resize_w, endpoint=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 解析行
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])

        try:
            # 这里的列索引根据你的CSV结构：0是文件名, 1,2是x1,y1, 3,4是x2,y2
            x1, y1 = float(row.iloc[1]), float(row.iloc[2])
            x2, y2 = float(row.iloc[3]), float(row.iloc[4])
        except ValueError:
            return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        # 读取图片
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            for suf in ['.JPG', '.jpg', '.png']:
                if os.path.exists(img_path + suf):
                    img_path += suf
                    break

        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        h_img, w_img = image.shape[:2]

        # 1. 传统特征提取
        _, _, _, trad_sinograms = self.detector.detect(image)

        processed_stack = []
        for sino in trad_sinograms[:3]:
            sino_resized = cv2.resize(sino, (self.resize_w, self.resize_h))
            mi, ma = sino_resized.min(), sino_resized.max()
            if ma - mi > 1e-6:
                sino_norm = (sino_resized - mi) / (ma - mi)
            else:
                sino_norm = np.zeros_like(sino_resized)
            processed_stack.append(sino_norm)

        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((self.resize_h, self.resize_w)))

        input_tensor = torch.from_numpy(np.array(processed_stack)).float()

        # 2. 标签生成 (修正版)
        # 不要在原图画线再缩放(线会消失)，而是缩放坐标后直接在小图画线
        scale = 256.0 / max(h_img, w_img)
        h_small, w_small = int(h_img * scale), int(w_img * scale)

        mask_small = np.zeros((h_small, w_small), dtype=np.uint8)

        # 缩放坐标
        sx1, sy1 = x1 * scale, y1 * scale
        sx2, sy2 = x2 * scale, y2 * scale

        # 画线
        cv2.line(mask_small, (int(sx1), int(sy1)), (int(sx2), int(sy2)), 255, 1)

        # Radon变换
        gt_sinogram = radon(mask_small, theta=self.theta_scan, circle=False)

        # 找最大值
        if gt_sinogram.max() > 0:
            max_idx = np.unravel_index(np.argmax(gt_sinogram), gt_sinogram.shape)
            rho_idx, theta_idx = max_idx

            # 归一化
            label_rho = rho_idx / (gt_sinogram.shape[0] - 1)
            label_theta = theta_idx / (gt_sinogram.shape[1] - 1)
        else:
            # 极少数情况线在图外，给个默认值
            label_rho, label_theta = 0.5, 0.5

        target = torch.tensor([label_rho, label_theta]).float()

        return input_tensor, target