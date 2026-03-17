"""
musid_linea_entropy.py — MU-SID dataset adapted for LINEA with entropy map

Resize 策略：等比例缩放 + 右下角零填充 (letterbox) 到正方形。
  避免直接拉伸 MU-SID 宽屏图像到正方形导致形变。

  1. scale = min(sz / orig_w, sz / orig_h) — 统一缩放因子
  2. 图像等比缩放到 (new_h, new_w)，右下角填 0 到 (sz, sz)
  3. GT 端点: (x, y) → (x * scale, y * scale)，padding 在右下角不引入偏移
  4. entropy map 同样等比缩放 + 右下角零填充
  5. 归一化: lines / sz → [0, 1]

MU-SID 每张图恰好 1 条 GT 海天线（class 0）。

返回格式取决于 include_entropy 开关：
  include_entropy=True  → 4-tuple (image, target, entropy_map, meta)
  include_entropy=False → 2-tuple (image, target)  — baseline 模式

target 格式与 LINEA COCO pipeline 完全一致（lines 归一化至 [0, 1]）。
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# ============================================================
# Global config — 顶部全局变量配置
# ============================================================
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

CSV_FILE = os.path.join(_PROJECT_ROOT, "splits_musid", "GroundTruth_train.csv")
IMG_DIR = os.path.join(_PROJECT_ROOT, "Hashmani's Dataset", "MU-SID")
ENTROPY_DIR = os.path.join(_PROJECT_ROOT, "Hashmani's Dataset", "MU-SID_entropy_blue")
IMG_SIZE = 640
NUM_SAMPLES_TO_PRINT = 3

# LINEA 归一化参数（来自 LINEA/datasets/coco.py）
MEAN = [0.538, 0.494, 0.453]
STD = [0.257, 0.263, 0.273]

IMAGE_EXTS = ('', '.JPG', '.jpg', '.png', '.jpeg', '.JPEG', '.PNG')


def _resolve_image_path(img_dir, name_in_csv):
    """尝试多种后缀定位图像文件。"""
    base = os.path.join(img_dir, str(name_in_csv))
    for suffix in IMAGE_EXTS:
        p = base if suffix == '' else base + suffix
        if os.path.exists(p):
            return p
    return None


def _letterbox_pil(pil_img, sz):
    """
    等比例缩放 PIL Image + 右下角零填充到 (sz, sz)。

    Returns:
        pil_out : PIL Image (sz, sz)
        scale   : float — 统一缩放因子
        new_w   : int — 缩放后实际宽度（不含 padding）
        new_h   : int — 缩放后实际高度（不含 padding）
    """
    orig_w, orig_h = pil_img.size
    scale = min(sz / orig_w, sz / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pil_resized = TF.resize(pil_img, [new_h, new_w])
    pad_right = sz - new_w
    pad_bottom = sz - new_h
    # TF.pad 参数: [left, top, right, bottom]
    pil_out = TF.pad(pil_resized, [0, 0, pad_right, pad_bottom], fill=0)
    return pil_out, scale, new_w, new_h


def _letterbox_entropy(ent_np, sz, new_w, new_h):
    """
    对 entropy map 做相同的 letterbox 变换：缩放 + 右下角零填充。

    Args:
        ent_np : ndarray [H_orig, W_orig], raw entropy (约 [0, 8])
        sz     : 目标正方形尺寸
        new_w  : 缩放后实际宽度
        new_h  : 缩放后实际高度

    Returns:
        ndarray [sz, sz], float32, in [0, 1]
    """
    ent_resized = cv2.resize(ent_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ent_resized = np.clip(ent_resized / 8.0, 0.0, 1.0).astype(np.float32)
    ent_out = np.zeros((sz, sz), dtype=np.float32)
    ent_out[:new_h, :new_w] = ent_resized
    return ent_out


class MUSIDLineaEntropyDataset(Dataset):
    """
    MU-SID → LINEA 格式 dataset，带可选 entropy map。
    使用 letterbox（等比缩放 + 右下角零填充）到正方形。

    Args:
        csv_file            : MU-SID split CSV (stem, x1, y1, x2, y2, ...)
        img_dir             : 原始图像目录
        entropy_dir         : 预计算 entropy .npy 目录
        img_size            : 输出正方形尺寸（默认 640）
        image_set           : 'train' / 'val' / 'test'
        include_entropy     : True → 返回 4-tuple; False → 返回 2-tuple (baseline)
        multiscale_entropy  : True → entropy_map [3, sz, sz] (MSLEP 多尺度);
                              False → entropy_map [1, sz, sz] (默认单通道)
    """

    # MSLEP 多尺度平均池化核大小 —— 从细节到粗粒度
    _MS_KERNELS = [1, 5, 11]

    def __init__(self, csv_file, img_dir, entropy_dir,
                 img_size=640, image_set='train', include_entropy=True,
                 multiscale_entropy=False):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f'CSV not found: {csv_file}')
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'IMG_DIR not found: {img_dir}')
        if include_entropy and not os.path.isdir(entropy_dir):
            raise FileNotFoundError(f'ENTROPY_DIR not found: {entropy_dir}')

        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.entropy_dir = entropy_dir
        self.sz = int(img_size)
        self.image_set = image_set
        self.include_entropy = include_entropy
        self.multiscale_entropy = multiscale_entropy
        self.is_train = 'train' in image_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        stem = str(row.iloc[0])
        x1, y1 = float(row.iloc[1]), float(row.iloc[2])
        x2, y2 = float(row.iloc[3]), float(row.iloc[4])

        # ---- 读取图像 ----
        img_path = _resolve_image_path(self.img_dir, stem)
        if img_path is None:
            raise FileNotFoundError(f'Image not found for stem: {stem}')

        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        # ---- Letterbox: 等比缩放 + 右下角零填充到正方形 ----
        pil_img, scale, new_w, new_h = _letterbox_pil(pil_img, self.sz)

        # ---- GT 线段坐标：统一缩放（padding 在右下角，不引入偏移） ----
        rx1, ry1 = x1 * scale, y1 * scale
        rx2, ry2 = x2 * scale, y2 * scale

        # ---- 读取 + letterbox entropy map ----
        if self.include_entropy:
            ent_path = os.path.join(self.entropy_dir, f'{Path(img_path).stem}.npy')
            if not os.path.isfile(ent_path):
                raise FileNotFoundError(f'Entropy map not found: {ent_path}')
            ent_np = np.load(ent_path).astype(np.float32)
            ent_sq = _letterbox_entropy(ent_np, self.sz, new_w, new_h)
        else:
            ent_path = ''
            ent_sq = None

        # ---- 训练增强：随机水平翻转 ----
        if self.is_train and random.random() < 0.5:
            pil_img = TF.hflip(pil_img)
            # 匹配 LINEA hflip 坐标约定：x_new = sz - x_old
            rx1 = self.sz - rx1
            rx2 = self.sz - rx2
            if ent_sq is not None:
                ent_sq = ent_sq[:, ::-1].copy()

        # ---- 训练增强：ColorJitter ----
        if self.is_train:
            pil_img = TF.adjust_brightness(pil_img, random.uniform(0.6, 1.4))
            pil_img = TF.adjust_contrast(pil_img, random.uniform(0.6, 1.4))
            pil_img = TF.adjust_saturation(pil_img, random.uniform(0.6, 1.4))

        # ---- 转 tensor + LINEA 标准归一化 ----
        image_tensor = TF.to_tensor(pil_img)                   # [3, sz, sz]
        image_tensor = TF.normalize(image_tensor, MEAN, STD)

        # ---- 端点排序（匹配 LINEA Normalize 约定） ----
        #   保证 x1 <= x2；若 x1 == x2 则 y1 >= y2
        if rx1 > rx2 or (rx1 == rx2 and ry1 < ry2):
            rx1, ry1, rx2, ry2 = rx2, ry2, rx1, ry1

        # ---- 归一化到 [0, 1]（匹配 LINEA Normalize 的 lines / [w, h, w, h]） ----
        lines_px = torch.tensor([[rx1, ry1, rx2, ry2]], dtype=torch.float32)
        lines_norm = lines_px / float(self.sz)

        target = {
            'lines': lines_norm,                                # [1, 4] in [0, 1]
            'labels': torch.tensor([0], dtype=torch.int64),     # class 0
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([self.sz, self.sz]),
            'area': torch.tensor([float(lines_px[0, 2:].sub(lines_px[0, :2]).norm())]),
            'iscrowd': torch.tensor([0]),
        }

        # ---- 返回格式取决于模式 ----
        meta = {
            'stem': stem,
            'img_path': img_path,
            'entropy_path': ent_path,
            'scale': scale,
            'new_w': new_w,
            'new_h': new_h,
        }
        if self.include_entropy:
            ent_tensor = torch.from_numpy(ent_sq).unsqueeze(0)  # [1, sz, sz]
            if self.multiscale_entropy:
                # MSLEP: 用不同大小的平均池化构造多尺度 entropy 通道
                channels = []
                for k in self._MS_KERNELS:
                    if k <= 1:
                        channels.append(ent_tensor)
                    else:
                        pad = k // 2
                        smoothed = torch.nn.functional.avg_pool2d(
                            ent_tensor.unsqueeze(0), kernel_size=k,
                            stride=1, padding=pad,
                        ).squeeze(0)  # [1, sz, sz]
                        channels.append(smoothed)
                entropy_tensor = torch.cat(channels, dim=0)  # [3, sz, sz]
            else:
                entropy_tensor = ent_tensor  # [1, sz, sz]
            return image_tensor, target, entropy_tensor, meta
        else:
            return image_tensor, target, meta


def main():
    ds = MUSIDLineaEntropyDataset(CSV_FILE, IMG_DIR, ENTROPY_DIR,
                                  img_size=IMG_SIZE, include_entropy=True)
    print(f'len = {len(ds)}')
    for i in range(min(NUM_SAMPLES_TO_PRINT, len(ds))):
        img, tgt, ent, meta = ds[i]
        print(f'[{i}] img={tuple(img.shape)} lines={tgt["lines"]} '
              f'ent={tuple(ent.shape)} scale={meta["scale"]:.4f} '
              f'new_wh=({meta["new_w"]},{meta["new_h"]}) stem={meta["stem"]}')


if __name__ == '__main__':
    main()