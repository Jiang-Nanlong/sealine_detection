"""
musid_entropy_dataset.py — MU-SID Dataset（完整版，带 entropy map）

功能：
  读取 MU-SID 图像 + 预计算局部熵图 + CSV 标注，返回统一格式的样本 dict。
  支持 resize，端点坐标会按缩放比例同步调整。

返回格式：
  {
    "image":       FloatTensor [3, H, W], RGB, [0, 1]
    "entropy_map": FloatTensor [1, H, W], clip(ent/8.0, 0, 1)
    "annotation":  dict — 原始 + resize 后的端点、中点、角度
    "meta":        dict — 文件路径、缩放系数等元信息
  }

与 musid_dataset.py 的区别：
  本文件包含完整的 resize 元数据和端点坐标缩放逻辑，
  适合直接用于训练/评估。
"""
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ============================================================
# Global config for PyCharm / server-side direct execution
# Edit these variables directly before running this file.
# This file is for dataset self-check and for being imported.
# ============================================================
# 自动定位项目根目录（stage1_scalelsd_entropy/data/ 往上两级）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

CSV_FILE = os.path.join(_PROJECT_ROOT, "splits_musid", "GroundTruth_train.csv")
IMG_DIR = os.path.join(_PROJECT_ROOT, "Hashmani's Dataset", "MU-SID")
ENTROPY_DIR = os.path.join(_PROJECT_ROOT, "Hashmani's Dataset", "MU-SID_entropy_blue")
IMG_SIZE = (576, 1024)   # (H, W)
NUM_SAMPLES_TO_PRINT = 3

IMAGE_EXTS = ('', '.JPG', '.jpg', '.png', '.jpeg', '.JPEG', '.PNG')


def _parse_hw(img_size: Union[int, Tuple[int, int]]):
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        h, w = int(img_size[0]), int(img_size[1])
        return max(1, h), max(1, w)
    s = int(img_size)
    return max(1, s), max(1, s)


def resolve_image_path(img_dir: str, name_in_csv: str):
    base = os.path.join(img_dir, str(name_in_csv))
    for suffix in IMAGE_EXTS:
        p = base if suffix == '' else base + suffix
        if os.path.exists(p):
            return p
    return None


def resize_rgb_u8(image_rgb_u8: np.ndarray, out_h: int, out_w: int):
    h, w = image_rgb_u8.shape[:2]
    sx = out_w / float(w)
    sy = out_h / float(h)
    interp = cv2.INTER_AREA if (out_w < w or out_h < h) else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (out_w, out_h), interpolation=interp)
    meta = dict(scale_x=sx, scale_y=sy, orig_w=w, orig_h=h, out_w=out_w, out_h=out_h)
    return resized, meta


class MUSIDEntropyDataset(Dataset):
    """
    Stage-1 dataset for local entropy preparation.
    Returns:
      image       : torch.FloatTensor [3, H, W] in [0,1]
      entropy_map : torch.FloatTensor [1, H, W] in [0,1]
      annotation  : raw + resized endpoints
      meta        : file path / resize metadata
    """

    def __init__(self, csv_file: str, img_dir: str, entropy_dir: str, img_size=(576, 1024), gray_scale=False):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f'CSV file not found: {csv_file}')
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'IMG_DIR not found: {img_dir}')
        if not os.path.isdir(entropy_dir):
            raise FileNotFoundError(f'ENTROPY_DIR not found: {entropy_dir}')

        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.entropy_dir = entropy_dir
        self.out_h, self.out_w = _parse_hw(img_size)
        self.gray_scale = gray_scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        stem = str(row.iloc[0])
        x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        xmid = float(row.iloc[5]) if len(row) > 5 else (x1 + x2) / 2.0
        ymid = float(row.iloc[6]) if len(row) > 6 else (y1 + y2) / 2.0
        angle = float(row.iloc[7]) if len(row) > 7 else 0.0

        img_path = resolve_image_path(self.img_dir, stem)
        if img_path is None:
            raise FileNotFoundError(f'Image not found for stem: {stem}')

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')

        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb, meta = resize_rgb_u8(rgb0, self.out_h, self.out_w)

        sx, sy = meta['scale_x'], meta['scale_y']
        p1 = (float(x1 * sx), float(y1 * sy))
        p2 = (float(x2 * sx), float(y2 * sy))
        pm = (float(xmid * sx), float(ymid * sy))

        ent_path = os.path.join(self.entropy_dir, f'{Path(img_path).stem}.npy')
        if not os.path.isfile(ent_path):
            raise FileNotFoundError(f'Entropy map not found: {ent_path}')

        ent_map = np.load(ent_path).astype(np.float32)
        if ent_map.shape[:2] != (self.out_h, self.out_w):
            ent_map = cv2.resize(ent_map, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        # Fixed-scale normalization, never per-image max normalization.
        ent_map = np.clip(ent_map / 8.0, 0.0, 1.0).astype(np.float32)

        if self.gray_scale:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            image_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        entropy_tensor = torch.from_numpy(ent_map).unsqueeze(0)

        annotation = {
            'stem': stem,
            'raw_endpoints': np.array([[x1, y1], [x2, y2]], dtype=np.float32),
            'resized_endpoints': np.array([p1, p2], dtype=np.float32),
            'raw_midpoint': np.array([xmid, ymid], dtype=np.float32),
            'resized_midpoint': np.array(pm, dtype=np.float32),
            'angle': angle,
        }
        meta_out = {
            'img_path': img_path,
            'entropy_path': ent_path,
            'orig_size': (meta['orig_h'], meta['orig_w']),
            'out_size': (self.out_h, self.out_w),
            'scale_x': sx,
            'scale_y': sy,
        }

        return {
            'image': image_tensor,
            'entropy_map': entropy_tensor,
            'annotation': annotation,
            'meta': meta_out,
        }


def main():
    ds = MUSIDEntropyDataset(CSV_FILE, IMG_DIR, ENTROPY_DIR, img_size=IMG_SIZE)
    print('len =', len(ds))
    for i in range(min(NUM_SAMPLES_TO_PRINT, len(ds))):
        sample = ds[i]
        print(f'[{i}] image={tuple(sample["image"].shape)} entropy={tuple(sample["entropy_map"].shape)} stem={sample["annotation"]["stem"]}')


if __name__ == '__main__':
    main()
