# --- START OF FILE dataset_loader.py (FIXED INPUT LOGIC) ---

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


# ==========================================
# 0. 图像缩放（支持正方形或 16:9 等任意尺寸）
# ==========================================
def _parse_img_size(img_size):
    """解析 img_size，支持 int 或 (H, W) 元组"""
    if isinstance(img_size, (list, tuple)):
        return int(img_size[0]), int(img_size[1])  # (H, W)
    else:
        s = int(img_size)
        return s, s  # 正方形

def resize_rgb_u8(image_rgb_u8: np.ndarray, dst_size, pad_value: int = 0):
    """
    直接 resize 到目标尺寸（无 letterbox padding）
    
    Args:
        image_rgb_u8: 输入图像 (H, W, 3) uint8 RGB
        dst_size: 目标尺寸，int (正方形) 或 (H, W) 元组
        pad_value: 未使用，保留兼容性
    
    Returns:
        resized: 缩放后图像 (dst_H, dst_W, 3)
        meta: 元信息字典
    """
    dst_h, dst_w = _parse_img_size(dst_size)
    h, w = image_rgb_u8.shape[:2]
    
    if h <= 0 or w <= 0:
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        meta = dict(scale_x=1.0, scale_y=1.0, pad_left=0, pad_top=0, 
                    new_w=dst_w, new_h=dst_h, orig_w=w, orig_h=h)
        return canvas, meta
    
    scale_x = dst_w / float(w)
    scale_y = dst_h / float(h)
    
    interp = cv2.INTER_AREA if (scale_x < 1.0 or scale_y < 1.0) else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (dst_w, dst_h), interpolation=interp)
    
    meta = dict(scale_x=scale_x, scale_y=scale_y, scale=scale_x,  # scale 保留兼容性
                pad_left=0, pad_top=0, new_w=dst_w, new_h=dst_h, orig_w=w, orig_h=h)
    return resized, meta

# 兼容旧代码的别名
letterbox_rgb_u8 = resize_rgb_u8


# ==========================================
# 1. 简单退化合成（rain + fog + dark，效果经验证更好）
# ==========================================
def synthesize_rain_fog(image_rgb_u8, p_clean: float = 0.45):
    """
    简化版退化合成函数（只有 3 种退化类型）
    
    退化类型：
      - rain: 雨丝效果
      - fog: 雾霾效果  
      - dark: 低光照/黄昏
    
    经测试，简单退化类型训练出的模型效果更好。
    
    输入:  uint8 RGB
    输出:  float32 RGB in [0,1]
    
    p_clean: 直接返回干净图的概率
    """
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # ---- clean pass-through ----
    if random.random() < float(p_clean):
        noise = np.random.randn(h, w, 3).astype(np.float32) * 0.005
        img = np.clip(img + noise, 0, 1)
        return img.astype(np.float32)

    # ---- 随机选择退化类型 ----
    is_rain = random.random() < 0.4
    is_fog = random.random() < 0.4
    is_dark = random.random() < 0.3

    if not (is_rain or is_fog or is_dark) and random.random() < 0.8:
        is_fog = True

    if is_rain:
        severity = random.uniform(0.3, 0.9)
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
        severity = random.uniform(0.2, 0.9)
        airlight = random.uniform(0.6, 0.95)
        ys = np.linspace(0, 1, h)[:, None]
        if random.random() > 0.5: base_map = np.tile(ys, (1, w))
        else: base_map = np.tile(1 - ys, (1, w))
        noise_map = np.random.rand(h, w)
        noise_map = cv2.GaussianBlur(noise_map, (0, 0), sigmaX=30, sigmaY=30)
        fog_map = base_map * 0.7 + noise_map * 0.3
        fog_map = fog_map * severity * random.uniform(0.8, 1.2)
        fog_map = np.clip(fog_map[:, :, None], 0, 1)
        img = img * (1 - fog_map) + airlight * fog_map

    if is_dark:
        factor = random.uniform(0.25, 0.6)
        img = img * factor

    noise = np.random.randn(h, w, 3) * 0.01
    img = np.clip(img + noise, 0, 1)
    return img.astype(np.float32)


def _scaled_ignore_band(ignore_band_at_384: int, img_size: int) -> int:
    if ignore_band_at_384 <= 0: return 0
    return max(1, int(round(ignore_band_at_384 * (img_size / 384.0))))

def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))

def _y_on_line_at_x(p1, p2, x: float) -> float:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-6: return y1
    t = (x - x1) / dx
    return y1 + t * (y2 - y1)


# ===========================================
# 2. Stage B/C 专用: 带 CSV 标签加载器
# ===========================================
class HorizonImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=384, mode='joint', ignore_band=10, augment=True, p_clean=0.45):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.img_h, self.img_w = _parse_img_size(img_size)  # 支持 (H, W) 元组
        self.mode = mode
        self.ignore_band = int(ignore_band)
        self.augment = augment
        self.p_clean = float(p_clean)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])
        try:
            x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        except Exception:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

        path = os.path.join(self.img_dir, img_name)
        possible_exts = ['', '.JPG', '.jpg', '.png', '.jpeg']
        bgr = None
        for ext in possible_exts:
            if os.path.exists(path + ext):
                bgr = cv2.imread(path + ext)
                break

        H, W = self.img_h, self.img_w
        if bgr is None:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
            meta = dict(scale_x=1.0, scale_y=1.0, pad_left=0, pad_top=0, orig_w=W, orig_h=H)
        else:
            rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb, meta = resize_rgb_u8(rgb0, (H, W))

        scale_x = float(meta["scale_x"])
        scale_y = float(meta["scale_y"])
        
        p1 = (_clamp_int(x1 * scale_x, 0, W - 1), _clamp_int(y1 * scale_y, 0, H - 1))
        p2 = (_clamp_int(x2 * scale_x, 0, W - 1), _clamp_int(y2 * scale_y, 0, H - 1))

        # Mask 初始化 (全 255)
        mask = np.full((H, W), 255, dtype=np.uint8)
        mask[:, :] = 0  # 图像区域为海(0)

        y_left = _y_on_line_at_x(p1, p2, 0.0)
        y_right = _y_on_line_at_x(p1, p2, float(W - 1))
        y_left = _clamp_int(y_left, 0, H - 1)
        y_right = _clamp_int(y_right, 0, H - 1)

        pts = np.array([[0, 0], [W - 1, 0], [W - 1, y_right], [0, y_left]], np.int32)
        cv2.fillPoly(mask, [pts], 1) # 天空(1)

        # ignore_band 按高度缩放
        thick = _scaled_ignore_band(self.ignore_band, H)
        if thick > 0:
            cv2.line(mask, p1, p2, 255, thick)

        # 旋转增强
        if self.augment and random.random() > 0.5:
            angle = random.uniform(-45, 45)
            img_pil = Image.fromarray(rgb)
            mask_pil = Image.fromarray(mask)
            img_pil = TF.rotate(img_pil, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
            mask_pil = TF.rotate(mask_pil, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=255)
            rgb = np.array(img_pil)
            mask = np.array(mask_pil)

        clean_np = rgb.copy()
        
        # [核心修正] 无论什么模式，先做退化！
        # 验证集 (augment=False) 固定随机种子
        if not self.augment:
            state_random = random.getstate()
            state_numpy = np.random.get_state()
            seed = idx + 100000
            random.seed(seed)
            np.random.seed(seed)
            degraded_np = synthesize_rain_fog(clean_np, p_clean=self.p_clean)
            random.setstate(state_random)
            np.random.set_state(state_numpy)
        else:
            degraded_np = synthesize_rain_fog(clean_np, p_clean=self.p_clean)

        # 准备 Tensor
        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1) # Degraded
        target_clean = torch.from_numpy(clean_np.astype(np.float32) / 255.0).permute(2, 0, 1) # Clean
        mask_tensor = torch.from_numpy(mask).long() # Mask

        # [关键改动] 分割模式返回 input(退化), mask
        if self.mode == 'segmentation':
            return input_tensor, mask_tensor  # <--- 改为返回退化图！

        return input_tensor, target_clean, mask_tensor


# ===========================================
# 3. Stage A 专用
# ===========================================
class SimpleFolderDataset(Dataset):
    def __init__(self, img_dir, img_size=384, augment=True, p_clean=0.45):
        self.img_dir = img_dir
        self.img_h, self.img_w = _parse_img_size(img_size)  # 支持 (H, W) 元组
        self.augment = augment
        self.p_clean = float(p_clean)
        raw_paths = glob.glob(os.path.join(img_dir, "*.[jJ][pP]*[gG]")) + \
                    glob.glob(os.path.join(img_dir, "*.png")) + \
                    glob.glob(os.path.join(img_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(img_dir, "*.jpg"))
        self.img_paths = sorted(list(set(raw_paths)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        bgr = cv2.imread(path)
        H, W = self.img_h, self.img_w
        if bgr is None:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb, _meta = resize_rgb_u8(rgb0, (H, W))

        if self.augment and random.random() > 0.5:
            angle = random.uniform(-45, 45)
            img_pil = Image.fromarray(rgb)
            img_pil = TF.rotate(img_pil, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
            rgb = np.array(img_pil)

        target_clean = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        if not self.augment:
            state_random = random.getstate()
            state_numpy = np.random.get_state()
            seed = idx + 200000
            random.seed(seed)
            np.random.seed(seed)
            degraded_np = synthesize_rain_fog(rgb, p_clean=self.p_clean)
            random.setstate(state_random)
            np.random.set_state(state_numpy)
        else:
            degraded_np = synthesize_rain_fog(rgb, p_clean=self.p_clean)

        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1)
        return input_tensor, target_clean