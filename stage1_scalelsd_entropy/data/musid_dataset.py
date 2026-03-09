#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
musid_dataset.py — MU-SID Dataset（局部熵子系统版）

返回 image + entropy_map + 原始标注信息，不做 ScaleLSD 格式转换。

CSV 格式（无表头，8 列，来自 splits_musid/GroundTruth_{split}.csv）:
  col0: image_stem    (e.g. DSC_0051_9)
  col1: x1            (海天线端点，1920×1080 坐标空间)
  col2: y1
  col3: x2
  col4: y2
  col5: x_mid
  col6: y_mid
  col7: angle (degrees)

返回:
  {
    "image":       FloatTensor [3, H, W], RGB, [0, 1]
    "entropy_map": FloatTensor [1, H, W], 固定归一化 clip(ent/8.0, 0, 1)
    "annotation":  dict — 原始标注信息（端点、角度等）
    "meta":        dict — 图像 stem、路径等元信息
  }

划分文件复用:
  splits_musid/GroundTruth_train.csv
  splits_musid/GroundTruth_val.csv
  splits_musid/GroundTruth_test.csv
"""

import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
ENTROPY_SCALE = 8.0  # 固定归一化系数：clip(ent / 8.0, 0, 1)

CSV_COLUMNS = ["stem", "x1", "y1", "x2", "y2", "xmid", "ymid", "angle"]

COMMON_EXTS = ("", ".JPG", ".jpg", ".jpeg", ".png", ".JPEG", ".PNG")


# ──────────────────────────────────────────────
# 图像读取工具
# ──────────────────────────────────────────────
def _resolve_image_path(img_dir: str, stem: str) -> str:
    """尝试常见后缀找到图像文件路径，找不到返回空字符串。"""
    for ext in COMMON_EXTS:
        path = os.path.join(img_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return ""


def _read_image_rgb(img_dir: str, stem: str) -> np.ndarray:
    """读取图像，返回 RGB uint8 [H, W, 3]。读取失败返回 None。"""
    path = _resolve_image_path(img_dir, stem)
    if not path:
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class MUSIDEntropyDataset(Dataset):
    """
    MU-SID dataset，加载 image + 预计算 entropy map + 原始标注。

    Parameters:
        csv_path:     split CSV 路径 (e.g. splits_musid/GroundTruth_train.csv)
        img_dir:      MU-SID 图像目录
        entropy_dir:  预计算 entropy .npy 目录
        img_h, img_w: 输出尺寸（image 和 entropy_map 统一 resize）
        orig_h, orig_w: CSV 坐标所在的原始分辨率
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        entropy_dir: str,
        img_h: int = 512,
        img_w: int = 512,
        orig_h: int = 1080,
        orig_w: int = 1920,
    ):
        # 读取 CSV，无表头，8 列
        self.df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
        self.img_dir = img_dir
        self.entropy_dir = entropy_dir
        self.img_h = img_h
        self.img_w = img_w
        self.orig_h = orig_h
        self.orig_w = orig_w

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        stem = str(row["stem"]).strip()

        # ── 读取原始标注 ──
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        xmid = float(row["xmid"])
        ymid = float(row["ymid"])
        angle = float(row["angle"])

        annotation = {
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "xmid": xmid, "ymid": ymid,
            "angle": angle,
            # 归一化端点（方便后续任意分辨率使用）
            "x1_norm": x1 / self.orig_w,
            "y1_norm": y1 / self.orig_h,
            "x2_norm": x2 / self.orig_w,
            "y2_norm": y2 / self.orig_h,
        }

        # ── 读取图像 ──
        rgb = _read_image_rgb(self.img_dir, stem)
        if rgb is None:
            # 兜底：返回零图（不应在正常运行时触发）
            rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            rgb = cv2.resize(rgb, (self.img_w, self.img_h),
                             interpolation=cv2.INTER_AREA)

        # ── 读取 entropy map ──
        ent_path = os.path.join(self.entropy_dir, f"{stem}.npy")
        if os.path.isfile(ent_path):
            ent_map = np.load(ent_path).astype(np.float32)
            # 如果 entropy map 尺寸与目标不一致，resize
            if ent_map.shape[0] != self.img_h or ent_map.shape[1] != self.img_w:
                ent_map = cv2.resize(ent_map, (self.img_w, self.img_h),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            ent_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)

        # ── 固定尺度归一化 entropy map ──
        # 明确禁止使用 ent_map / ent_map.max()
        ent_map = np.clip(ent_map / ENTROPY_SCALE, 0.0, 1.0)

        # ── 转 Tensor ──
        # image: [3, H, W], float32, [0, 1]
        img_tensor = torch.from_numpy(
            rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # entropy_map: [1, H, W], float32, [0, 1]
        ent_tensor = torch.from_numpy(ent_map).unsqueeze(0)

        meta = {
            "stem": stem,
            "img_path": _resolve_image_path(self.img_dir, stem),
            "ent_path": ent_path,
            "orig_h": self.orig_h,
            "orig_w": self.orig_w,
            "target_h": self.img_h,
            "target_w": self.img_w,
        }

        return {
            "image": img_tensor,
            "entropy_map": ent_tensor,
            "annotation": annotation,
            "meta": meta,
        }


# ──────────────────────────────────────────────
# 快速自测
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test: load MUSIDEntropyDataset")
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument("--csv", type=str,
                        default=os.path.join(_project_root, "splits_musid", "GroundTruth_train.csv"),
                        help="Split CSV path")
    parser.add_argument("--img_dir", type=str,
                        default=os.path.join(_project_root, "Hashmani's Dataset", "MU-SID"),
                        help="Image directory")
    parser.add_argument("--entropy_dir", type=str,
                        default=os.path.join(_project_root, "Hashmani's Dataset", "MU-SID_entropy_blue"),
                        help="Precomputed entropy .npy directory")
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to print")
    args = parser.parse_args()

    ds = MUSIDEntropyDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        entropy_dir=args.entropy_dir,
        img_h=args.img_h,
        img_w=args.img_w,
    )
    print(f"Dataset size: {len(ds)}")
    print(f"CSV columns: {CSV_COLUMNS}")
    print()

    for i in range(min(args.num_samples, len(ds))):
        sample = ds[i]
        img = sample["image"]
        ent = sample["entropy_map"]
        ann = sample["annotation"]
        meta = sample["meta"]
        print(f"[{i}] stem={meta['stem']}")
        print(f"     image:       shape={tuple(img.shape)}, "
              f"dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")
        print(f"     entropy_map: shape={tuple(ent.shape)}, "
              f"dtype={ent.dtype}, range=[{ent.min():.3f}, {ent.max():.3f}]")
        print(f"     annotation:  x1={ann['x1']}, y1={ann['y1']}, "
              f"x2={ann['x2']}, y2={ann['y2']}, angle={ann['angle']}")
        print()
