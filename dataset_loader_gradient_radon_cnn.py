# 文件名: dataset_loader.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from skimage.transform import radon
from gradient_radon import TextureSuppressedMuSCoWERT

def calculate_radon_label(x1, y1, x2, y2, img_w, img_h, resize_h):
    """
    将图像坐标系下的直线端点 (x1,y1), (x2,y2) 转换为 Radon 域的标签 (rho_norm, theta_norm)。

    参数:
        x1, y1, x2, y2: 直线端点坐标
        img_w, img_h: 原始图片宽高
        resize_h: 输入给 CNN 的正弦图的高度 (rho轴的分辨率)

    返回:
        label_rho: 归一化到 [0, 1] 的距离，0.5 表示在中心
        label_theta: 归一化到 [0, 1] 的角度 (对应 0-180度)
    """
    # 1. 将原点移到图像中心 (skimage.radon 的旋转中心是图片中心)
    cx, cy = img_w / 2.0, img_h / 2.0

    # 2. 计算直线的斜率角度 (image coordinates, y向下)
    dx = x2 - x1
    dy = y2 - y1

    # 计算线段的角度 [-pi, pi]
    # 注意：opencv/numpy 图像坐标系中 y 是向下的
    line_angle = np.arctan2(dy, dx)

    # 3. 计算法线角度 (Radon变换的角度 theta)
    # 法线垂直于直线，所以 +90度 (pi/2)
    # skimage 定义: projection at angle theta is sum over line x*cos(theta) + y*sin(theta) = rho
    # 这意味着 theta 是法向量的角度。
    theta_rad = line_angle - np.pi / 2

    # 规范化 theta 到 [0, pi) -> [0, 180度)
    # 因为 Radon 变换通常是 0-180 度
    while theta_rad < 0:
        theta_rad += np.pi
    while theta_rad >= np.pi:
        theta_rad -= np.pi

    # 4. 计算 Rho (原点到直线的垂直距离)
    # 使用点到直线距离公式（中心坐标系下）
    # 直线方程: (y1-y2)x + (x2-x1)y + x1y2 - x2y1 = 0 (在原始坐标系下)
    # 转换到中心坐标系: X = x - cx, Y = y - cy
    # 简单方法：投影法
    # rho = x_center * cos(theta) + y_center * sin(theta)
    # 取中点
    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy

    # 计算有符号距离
    # 注意：这里的 theta_rad 是法线角度
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)

    # 5. 归一化标签
    # Theta: [0, 180] -> [0, 1]
    label_theta = np.degrees(theta_rad) / 180.0

    # Rho:
    # 正弦图的高度通常对应图像的对角线长度
    # resize_h 是我们缩放后的正弦图高度
    # 我们假设 resize_h 的中心 (resize_h/2) 对应 rho=0
    # 这里的缩放因子不仅是图像尺寸，还有 resize_h 的归一化
    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)

    # 映射逻辑：
    # 真实 rho 在 [-diag/2, diag/2] 之间
    # 归一化 rho_norm = (rho / (diag/2) + 1) / 2  -> [0, 1]
    # 或者简单点：rho_pixel_pos = rho + (resize_h / scale_factor / 2)

    # 为了简化 CNN 学习，我们直接把 rho 映射到 [-1, 1] 区间然后转到 [0, 1]
    # 假设最大可能的 rho 是对角线的一半
    max_rho = original_diag / 2.0
    label_rho = (rho / max_rho + 1.0) / 2.0

    # 截断防止溢出 (通常不会，除非线在图外面)
    label_rho = np.clip(label_rho, 0.0, 1.0)

    return label_rho, label_theta


class HorizonFusionDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, resize_h=362, resize_w=180):
        """
        参数:
            csv_file (str): 标签 CSV 文件路径
            img_dir (str): 原始图片文件夹路径
            mask_dir (str, optional): 语义分割 Mask 图片文件夹路径。如果不传，则使用 GroundTruth 模拟 Mask。
            resize_h (int): 正弦图统一缩放的高度 (建议 362 或 512)
            resize_w (int): 正弦图统一缩放的宽度 (角度分辨率，通常 180)
        """
        # 读取 CSV，假设没有表头，如果有表头请改为 header=0
        # 根据你的描述，最后一列是角度，前几列是坐标
        # 我们这里假设读取为 DataFrame
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.resize_h = resize_h
        self.resize_w = resize_w

        # 初始化传统算法检测器
        # 必须开启 full_scan=True 以获取完整的 0-180 度正弦图
        print("正在初始化传统算法提取器...")
        self.detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

        # 预计算角度列表，用于 Mask 的 Radon 变换
        self.theta_scan = np.linspace(0., 180., resize_w, endpoint=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 解析 CSV 数据
        # 假设 CSV 格式: [ImageName, x1, y1, x2, y2, xc, yc, angle]
        # 请根据你实际的 CSV 列顺序调整 iloc 的索引
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])

        # 读取坐标 (根据你的描述调整列号)
        # 假设第2,3列是点1，第4,5列是点2
        try:
            x1, y1 = float(row.iloc[1]), float(row.iloc[2])
            x2, y2 = float(row.iloc[3]), float(row.iloc[4])
        except ValueError:
            # 如果坐标解析失败，返回全0数据（防止训练中断）
            return torch.zeros((4, self.resize_h, self.resize_w)), torch.zeros(2)

        # 2. 读取图片
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            if os.path.exists(img_path + '.JPG'):
                img_path += '.JPG'
            elif os.path.exists(img_path + '.jpg'):
                img_path += '.jpg'
            elif os.path.exists(img_path + '.png'):
                img_path += '.png'

        image = cv2.imread(img_path)

        # 容错处理
        if image is None:
            # 尝试加后缀读取 (有的csv不带后缀)
            image = cv2.imread(img_path + '.JPG')
            if image is None:
                print(f"Warning: Image {img_name} not found.")
                return torch.zeros((4, self.resize_h, self.resize_w)), torch.zeros(2)

        h_img, w_img = image.shape[:2]

        # 3. --- 获取传统方法正弦图 ---
        # 调用 detect 方法，获取第4个返回值：collected_sinograms
        _, _, _, trad_sinograms = self.detector.detect(image)

        # 4. --- 获取语义分割正弦图 ---
        # if self.mask_dir:
        #     # 尝试读取预先生成的 Mask
        #     # 假设 mask 文件名与原图一致，只是后缀可能是 png
        #     mask_name = os.path.splitext(img_name)[0] + ".png"
        #     mask_path = os.path.join(self.mask_dir, mask_name)
        #     mask = cv2.imread(mask_path, 0)  # 读取为灰度
        #
        #     if mask is None:
        #         # 如果没读到，创建一个空 mask (或者报错)
        #         mask = np.zeros((h_img, w_img), dtype=np.uint8)
        # else:
        #     # 【训练阶段 - 模拟语义分割】
        #     # 如果没有 mask_dir，使用 Ground Truth 画一条线来模拟完美的语义分割结果
        #     # 注意：这只是为了验证代码流程。真正的训练应该使用你 UNet 的预测结果。
        #     mask = np.zeros((h_img, w_img), dtype=np.uint8)
        #     cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)

        # 对 Mask 做 Radon 变换
        # Resize mask to speed up radon transform (optional but recommended for large images)
        # mask_small = cv2.resize(mask, (256, 256))
        # seg_sinogram = radon(mask_small, theta=self.theta_scan, circle=False)
        # seg_sinogram = radon(mask, theta=self.theta_scan, circle=False)

        # 5. --- 数据融合与归一化 ---
        # 此时我们有：3张传统正弦图 + 1张分割正弦图
        # all_sinos = trad_sinograms + [seg_sinogram]
        all_sinos = trad_sinograms
        processed_stack = []
        for sino in all_sinos:
            # A. 统一尺寸 -> (resize_h, resize_w)
            sino_resized = cv2.resize(sino, (self.resize_w, self.resize_h))

            # B. Min-Max 归一化 (关键步骤)
            # 将数值映射到 [0, 1]
            mi, ma = sino_resized.min(), sino_resized.max()
            if ma - mi > 1e-6:
                sino_norm = (sino_resized - mi) / (ma - mi)
            else:
                sino_norm = np.zeros_like(sino_resized)

            processed_stack.append(sino_norm)

        # 堆叠成 Tensor: shape [4, H, W]
        input_tensor = torch.from_numpy(np.array(processed_stack)).float()

        # 6. --- 标签生成 ---
        label_rho, label_theta = calculate_radon_label(x1, y1, x2, y2, w_img, h_img, self.resize_h)
        target = torch.tensor([label_rho, label_theta]).float()

        return input_tensor, target


# -----------------------------------------------------------------------------
# 简单的测试代码
if __name__ == "__main__":
    # 配置你的路径用于测试
    csv_path = r"D:\dataset\Hashmani's Dataset\GroundTruth.csv"  # 替换你的csv路径
    img_folder = r"D:\dataset\Hashmani's Dataset\MU-SID"  # 替换你的图片路径

    # 实例化数据集
    print("初始化数据集...")
    dataset = HorizonFusionDataset(csv_path, img_folder)

    print(f"数据集长度: {len(dataset)}")

    # 测试获取一个样本
    if len(dataset) > 0:
        data, label = dataset[0]
        print(f"输入 Tensor 形状: {data.shape}")  # 应该是 [4, 362, 180]
        print(f"标签 Tensor: {label}")  # 应该是 [rho_norm, theta_norm]

        # 可视化检查一下
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.imshow(data[i].numpy(), cmap='jet', aspect='auto')
            title = "Seg Mask" if i == 3 else f"Trad Scale {i + 1}"
            plt.title(title)
            plt.axis('off')
        plt.suptitle(f"Label: Rho={label[0]:.4f}, Theta={label[1]:.4f}")
        plt.show()