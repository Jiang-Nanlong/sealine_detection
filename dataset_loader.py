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
# 0. Letterbox (keep aspect ratio + padding)
# ==========================================
def letterbox_rgb_u8(image_rgb_u8: np.ndarray, dst_size: int, pad_value: int = 0):
    h, w = image_rgb_u8.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.zeros((dst_size, dst_size, 3), dtype=np.uint8)
        meta = dict(scale=1.0, pad_left=0, pad_top=0, new_w=dst_size, new_h=dst_size, orig_w=w, orig_h=h)
        return canvas, meta

    scale = min(dst_size / float(w), dst_size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = max(1, min(dst_size, new_w))
    new_h = max(1, min(dst_size, new_h))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=interp)

    canvas = np.full((dst_size, dst_size, 3), pad_value, dtype=np.uint8)
    pad_left = (dst_size - new_w) // 2
    pad_top = (dst_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = dict(scale=scale, pad_left=pad_left, pad_top=pad_top, new_w=new_w, new_h=new_h, orig_w=w, orig_h=h)
    return canvas, meta


# ==========================================
# 1. 物理感知的合成退化函数
# ==========================================
def synthesize_rain_fog(image_rgb_u8):
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape

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
    def __init__(self, csv_file, img_dir, img_size=384, mode='joint', ignore_band=10, augment=True):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.img_size = int(img_size)
        self.mode = mode
        self.ignore_band = int(ignore_band)
        self.augment = augment

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

        if bgr is None:
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            meta = dict(scale=1.0, pad_left=0, pad_top=0, orig_w=self.img_size, orig_h=self.img_size)
        else:
            rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb, meta = letterbox_rgb_u8(rgb0, self.img_size, pad_value=0)

        S = self.img_size
        scale = float(meta["scale"])
        pl = int(meta["pad_left"]); pt = int(meta["pad_top"])
        nw = int(meta["new_w"]); nh = int(meta["new_h"])
        
        p1 = (_clamp_int(x1 * scale + pl, 0, S - 1), _clamp_int(y1 * scale + pt, 0, S - 1))
        p2 = (_clamp_int(x2 * scale + pl, 0, S - 1), _clamp_int(y2 * scale + pt, 0, S - 1))

        # Mask 初始化 (全 255)
        mask = np.full((S, S), 255, dtype=np.uint8)
        if nw > 0 and nh > 0:
            mask[pt : pt + nh, pl : pl + nw] = 0 # 图像区域为海(0)

        y_left = _y_on_line_at_x(p1, p2, 0.0)
        y_right = _y_on_line_at_x(p1, p2, float(S - 1))
        y_left = _clamp_int(y_left, 0, S - 1)
        y_right = _clamp_int(y_right, 0, S - 1)

        pts = np.array([[0, 0], [S - 1, 0], [S - 1, y_right], [0, y_left]], np.int32)
        cv2.fillPoly(mask, [pts], 1) # 天空(1)

        thick = _scaled_ignore_band(self.ignore_band, S)
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
            degraded_np = synthesize_rain_fog(clean_np)
            random.setstate(state_random)
            np.random.set_state(state_numpy)
        else:
            degraded_np = synthesize_rain_fog(clean_np)

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
    def __init__(self, img_dir, img_size=384, augment=True):
        self.img_dir = img_dir
        self.img_size = int(img_size)
        self.augment = augment
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
        if bgr is None:
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb, _meta = letterbox_rgb_u8(rgb0, self.img_size, pad_value=0)

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
            degraded_np = synthesize_rain_fog(rgb)
            random.setstate(state_random)
            np.random.set_state(state_numpy)
        else:
            degraded_np = synthesize_rain_fog(rgb)

        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1)
        return input_tensor, target_clean