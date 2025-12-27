import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
# 导入你的 GPU 加速类
from gradient_radon import TextureSuppressedMuSCoWERT


class HorizonFusionDataset(Dataset):
    def __init__(self, csv_file, img_dir, resize_h=2240, resize_w=180):
        """
        全分辨率 + GPU 加速的数据加载器
        resize_h: 设为 2240 (略大于 sqrt(1920^2 + 1080^2) = 2203) 以容纳完整正弦图
        resize_w: 180 (角度分辨率)
        """
        # 1. 读取 CSV (header=None 修复了第一行丢失的问题)
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir

        # 这里的 resize_h 不再是缩放目标，而是 Padding 的容器大小
        self.resize_h = resize_h
        self.resize_w = resize_w

        print(f"Dataset initialized. Total images: {len(self.data)}")

        # 2. 初始化 GPU 处理器
        # 必须开启 full_scan=True 以获取 0-180 度数据
        self.detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

        # 预计算角度 (0-180度)，用于生成标签
        self.theta_scan = np.linspace(0., 180., resize_w, endpoint=False)

        # 检查是否真的在用 4090
        if self.detector.device.type == 'cuda':
            print(f"Using GPU Radon Transform on {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: Running on CPU! Performance will be limited.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # --- 1. 解析数据 ---
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])

        try:
            # 读取坐标 (第2,3列是点1，第4,5列是点2)
            x1, y1 = float(row.iloc[1]), float(row.iloc[2])
            x2, y2 = float(row.iloc[3]), float(row.iloc[4])
        except (ValueError, IndexError):
            # 容错返回
            return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        # --- 2. 读取图片 ---
        img_path = os.path.join(self.img_dir, img_name)
        image = None
        # 自动尝试后缀
        possible_exts = ['', '.JPG', '.jpg', '.png']
        for ext in possible_exts:
            test_path = img_path if img_path.lower().endswith(ext.lower()) else img_path + ext
            if os.path.exists(test_path):
                image = cv2.imread(test_path)
                break

        if image is None:
            # print(f"Warning: Image {img_name} not found.")
            return torch.zeros((3, self.resize_h, self.resize_w)), torch.zeros(2)

        h_img, w_img = image.shape[:2]

        # --- 3. 生成输入特征 (Input) - Padding 模式 ---
        # 使用 detector 提取传统特征 (梯度+Radon)
        # trad_sinograms 是一个列表，包含3个不同尺度的正弦图
        _, _, _, trad_sinograms = self.detector.detect(image)

        processed_stack = []

        # 我们只取前3个尺度
        for sino in trad_sinograms[:3]:
            # A. 归一化 (先归一化保留细节)
            mi, ma = sino.min(), sino.max()
            if ma - mi > 1e-6:
                sino_norm = (sino - mi) / (ma - mi)
            else:
                sino_norm = np.zeros_like(sino)

            # B. 居中填充 (Padding)
            # 原始 sinogram 高度取决于图片对角线 (约 2203)，宽度固定 180
            h_curr, w_curr = sino_norm.shape

            # 创建全黑容器
            container = np.zeros((self.resize_h, self.resize_w), dtype=np.float32)

            # 计算起始位置 (居中)
            start_h = (self.resize_h - h_curr) // 2

            # 放入容器
            if h_curr <= self.resize_h:
                container[start_h: start_h + h_curr, :] = sino_norm
            else:
                # 如果原图巨大超过了容器(极少情况)，进行中心裁剪
                crop_start = (h_curr - self.resize_h) // 2
                container[:, :] = sino_norm[crop_start: crop_start + self.resize_h, :]

            processed_stack.append(container)

        # 补齐通道 (防止提取失败只返回了空列表)
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((self.resize_h, self.resize_w)))

        input_tensor = torch.from_numpy(np.array(processed_stack)).float()

        # --- 4. 生成标签 (Label) - 4090 暴力版 ---
        # A. 创建全尺寸 Mask (不缩放，保持最高精度)
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # B. 直接用原始坐标画线
        # 线宽设为 3，确保在高分辨率下有足够的积分强度
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)

        # C. 调用 GPU Radon 变换 (原图尺寸)
        # 这会返回一个高分辨率的 Sinogram
        gt_sinogram = self.detector._radon_gpu(mask, self.theta_scan)

        # D. 寻找最大值 (Ground Truth Location)
        if gt_sinogram.max() > 0:
            # 找到原始 Sinogram 中的坐标
            max_idx = np.unravel_index(np.argmax(gt_sinogram), gt_sinogram.shape)
            raw_rho_idx, theta_idx = max_idx  # raw_rho_idx 是在非 Padding 图中的位置

            # E. 坐标映射 (关键步骤！)
            # 我们必须把 "原始位置" 映射到 "CNN看到的Padding后的容器位置"
            h_curr = gt_sinogram.shape[0]
            pad_top = (self.resize_h - h_curr) // 2

            # 如果发生了裁剪(Overflow)，逻辑需要反过来
            if h_curr > self.resize_h:
                crop_start = (h_curr - self.resize_h) // 2
                final_rho_idx = raw_rho_idx - crop_start
            else:
                final_rho_idx = raw_rho_idx + pad_top

            # 归一化到 [0, 1]
            # 注意分母是容器的高度 self.resize_h
            label_rho = final_rho_idx / (self.resize_h - 1)
            label_theta = theta_idx / (self.resize_w - 1)

            # 边界截断防止计算误差溢出
            label_rho = np.clip(label_rho, 0.0, 1.0)
            label_theta = np.clip(label_theta, 0.0, 1.0)
        else:
            # 异常兜底
            label_rho, label_theta = 0.5, 0.5

        target = torch.tensor([label_rho, label_theta]).float()

        return input_tensor, target


# 测试块
if __name__ == "__main__":
    # 简单的路径配置用于测试
    csv_path = r"Hashmani's Dataset/GroundTruth.csv"
    img_folder = r"Hashmani's Dataset/MU-SID"

    print("Testing Dataset Loader...")
    ds = HorizonFusionDataset(csv_path, img_folder)

    if len(ds) > 0:
        inp, lab = ds[0]
        print(f"Input Shape: {inp.shape}")  # 应为 [3, 2240, 180]
        print(f"Label: {lab}")