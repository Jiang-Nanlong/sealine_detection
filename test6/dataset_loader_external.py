# -*- coding: utf-8 -*-
"""
Dataset loader for external datasets (SMD, Buoy) - Experiment 6.

用于在SMD和Buoy数据集上训练UNet和CNN。
与主训练代码的数据加载策略保持一致。

支持两种模式：
  - joint: 返回 (degraded_img, clean_img, seg_mask) 用于联合训练
  - segmentation: 返回 (clean_img, seg_mask) 用于分割训练

注意：由于SMD和Buoy数据集本身就包含各种退化（雾、雨等），
所以这里的"clean"图像实际上就是原图，退化合成比例较低。
"""

import os
import random
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ExternalDataset(Dataset):
    """
    Dataset for SMD/Buoy with on-the-fly degradation synthesis.
    
    CSV格式要求：img_name,x1,y1,x2,y2
    """
    
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        img_size: Tuple[int, int] = (576, 1024),  # (H, W)
        mode: str = "joint",  # "joint" or "segmentation"
        augment: bool = False,
        p_clean: float = 0.35,  # 保持干净的概率
    ):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        self.augment = augment
        self.p_clean = p_clean
        
        # 加载CSV
        self.df = pd.read_csv(csv_path)
        self.n_samples = len(self.df)
        
        # 退化类型（与主代码一致：rain, fog, dark）
        self.degradation_types = ["rain", "fog", "dark"]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = str(row["img_name"])
        
        # 读取图像
        img_path = os.path.join(self.img_dir, img_name)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            # 返回空数据
            h, w = self.img_size
            if self.mode == "segmentation":
                return torch.zeros(3, h, w), torch.zeros(h, w, dtype=torch.long)
            else:
                return torch.zeros(3, h, w), torch.zeros(3, h, w), torch.zeros(h, w, dtype=torch.long)
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Resize到目标尺寸
        h, w = self.img_size
        rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        
        # 获取GT坐标并缩放
        h_orig, w_orig = bgr.shape[:2]
        x1 = float(row["x1"]) * w / w_orig
        y1 = float(row["y1"]) * h / h_orig
        x2 = float(row["x2"]) * w / w_orig
        y2 = float(row["y2"]) * h / h_orig
        
        # 生成分割mask（基于GT线）
        mask = self._generate_mask_from_line(x1, y1, x2, y2, h, w)
        
        # 数据增强
        if self.augment:
            rgb_resized, mask = self._augment(rgb_resized, mask)
        
        # 归一化
        clean_tensor = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        
        if self.mode == "segmentation":
            # 分割模式：返回 (clean, mask)
            return clean_tensor, mask_tensor
        else:
            # Joint模式：返回 (degraded, clean, mask)
            if random.random() < self.p_clean:
                # 保持干净
                input_tensor = clean_tensor.clone()
            else:
                # 应用退化
                input_tensor = self._apply_degradation(clean_tensor)
            
            return input_tensor, clean_tensor, mask_tensor
    
    def _generate_mask_from_line(self, x1: float, y1: float, x2: float, y2: float, h: int, w: int) -> np.ndarray:
        """根据GT水平线生成分割mask：上方为sky(1)，下方为sea(0)"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 计算直线方程：y = k*x + b
        if abs(x2 - x1) < 1e-6:
            # 垂直线（不太可能是水平线）
            return mask
        
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        
        # 对每一列x，计算y值，上方设为1（sky）
        for x in range(w):
            y_line = int(k * x + b)
            y_line = max(0, min(h - 1, y_line))
            mask[:y_line, x] = 1  # 上方是sky
        
        return mask
    
    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """简单的数据增强：水平翻转"""
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        return img, mask
    
    def _apply_degradation(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """应用随机退化（与主代码一致）"""
        deg_type = random.choice(self.degradation_types)
        
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        if deg_type == "rain":
            img_np = self._add_rain(img_np)
        elif deg_type == "fog":
            img_np = self._add_fog(img_np)
        elif deg_type == "dark":
            img_np = self._add_dark(img_np)
        
        return torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
    
    def _add_rain(self, img: np.ndarray) -> np.ndarray:
        """添加雨效果"""
        h, w = img.shape[:2]
        rain = np.zeros((h, w), dtype=np.float32)
        
        # 随机雨滴
        n_drops = random.randint(100, 300)
        for _ in range(n_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(10, 30)
            angle = random.uniform(-0.1, 0.1)  # 近乎垂直
            
            for l in range(length):
                yy = int(y + l)
                xx = int(x + l * angle)
                if 0 <= yy < h and 0 <= xx < w:
                    rain[yy, xx] = random.uniform(0.3, 0.8)
        
        # 模糊雨滴
        rain = cv2.GaussianBlur(rain, (3, 3), 0)
        
        # 叠加
        rain_rgb = np.stack([rain, rain, rain], axis=-1) * 255
        result = np.clip(img.astype(np.float32) + rain_rgb * 0.5, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_fog(self, img: np.ndarray) -> np.ndarray:
        """添加雾效果"""
        h, w = img.shape[:2]
        fog_intensity = random.uniform(0.2, 0.5)
        
        # 渐变雾（从上到下逐渐减弱）
        fog = np.linspace(fog_intensity, fog_intensity * 0.3, h).reshape(-1, 1)
        fog = np.tile(fog, (1, w))
        
        # 添加随机噪声
        noise = np.random.randn(h, w) * 0.05
        fog = np.clip(fog + noise, 0, 1).astype(np.float32)
        
        # 雾色（灰白色）
        fog_color = np.array([200, 200, 200], dtype=np.float32)
        
        result = img.astype(np.float32) * (1 - fog[:, :, None]) + fog_color * fog[:, :, None]
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _add_dark(self, img: np.ndarray) -> np.ndarray:
        """降低亮度"""
        factor = random.uniform(0.3, 0.6)
        result = (img.astype(np.float32) * factor).astype(np.uint8)
        return result


def load_external_split_indices(split_dir: str) -> dict:
    """加载外部数据集的split索引"""
    result = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(split_dir, f"{split}_indices.npy")
        if os.path.exists(path):
            result[split] = np.load(path).astype(np.int64).tolist()
        else:
            result[split] = []
    return result
