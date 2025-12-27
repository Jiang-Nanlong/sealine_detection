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
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.resize_h = resize_h  # Rho 轴分辨率
        self.resize_w = resize_w  # Theta 轴分辨率

        # 初始化传统算法提取器
        print("初始化 Radon 变换器 (4090 Ready)...")
        self.detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

        # 预计算用于 Label 生成的角度列表
        self.theta_scan = np.linspace(0., 180., resize_w, endpoint=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])

        # 读取坐标
        try:
            x1, y1 = float(row.iloc[1]), float(row.iloc[2])
            x2, y2 = float(row.iloc[3]), float(row.iloc[4])
        except ValueError:
            return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        # 读取图片
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            suffixes = ['.JPG', '.jpg', '.png']
            found = False
            for suf in suffixes:
                if os.path.exists(img_path + suf):
                    img_path += suf
                    found = True
                    break
            if not found:
                # print(f"Warning: Image {img_name} not found.")
                return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        h_img, w_img = image.shape[:2]

        # --- 1. 获取输入特征 (Sinograms) ---
        # 这一步比较耗时，但为了准确性是必须的 (离线缓存会解决这个问题)
        _, _, _, trad_sinograms = self.detector.detect(image)

        processed_stack = []
        # 我们只取前3个尺度的特征
        for sino in trad_sinograms[:3]:
            # 统一缩放
            sino_resized = cv2.resize(sino, (self.resize_w, self.resize_h))  # cv2是 (W, H)

            # Robust Min-Max 归一化 (防除零)
            mi, ma = sino_resized.min(), sino_resized.max()
            if ma - mi > 1e-6:
                sino_norm = (sino_resized - mi) / (ma - mi)
            else:
                sino_norm = np.zeros_like(sino_resized)
            processed_stack.append(sino_norm)

        # 补齐通道 (万一不够3个)
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((self.resize_h, self.resize_w)))

        input_tensor = torch.from_numpy(np.array(processed_stack)).float()

        # --- 2. 生成绝对准确的标签 (Projection-based Labeling) ---
        # 核心逻辑：在空图上画 GT 线 -> 做同样的 Radon 变换 -> 找最亮点的坐标
        # 这样网络只需要学习 "找最亮点"，这比学习复杂的几何映射容易得多

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)

        # 缩小 Mask 以加速 Radon 计算 (保持比例即可)
        scale_factor = 256.0 / max(h_img, w_img)
        mask_small = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # 执行 Radon
        gt_sinogram = radon(mask_small, theta=self.theta_scan, circle=False)

        # 找到最大值的位置 (unravel_index 返回的是 (row, col))
        # row -> rho, col -> theta
        max_idx = np.unravel_index(np.argmax(gt_sinogram), gt_sinogram.shape)
        rho_idx, theta_idx = max_idx

        # 归一化标签到 [0, 1]
        # 注意：这里的 gt_sinogram 的形状是 (num_rho, num_theta)
        # num_theta = self.resize_w
        # num_rho 取决于 mask_small 的对角线长度

        label_rho = rho_idx / (gt_sinogram.shape[0] - 1)
        label_theta = theta_idx / (gt_sinogram.shape[1] - 1)

        # 添加极小的噪声防止过拟合 (可选)
        # label_rho += np.random.normal(0, 0.001)

        target = torch.tensor([label_rho, label_theta]).float()

        return input_tensor, target