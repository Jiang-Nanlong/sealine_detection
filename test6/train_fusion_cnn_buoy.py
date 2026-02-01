# -*- coding: utf-8 -*-
"""
Train Fusion-CNN on Buoy dataset with rain+fog augmentation (Experiment 6: In-Domain).

Inputs:
  - test6/FusionCache_Buoy/train/
  - test6/FusionCache_Buoy/val/
  - test6/splits_buoy/train_indices.npy
  - test6/splits_buoy/val_indices.npy

Outputs:
  - test6/weights/best_fusion_cnn_buoy.pth
  - test6/train_log_buoy.json

PyCharm: 直接运行此文件
"""

import os
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_model import HorizonResNet  # noqa: E402


# ============================
# PyCharm 配置区
# ============================
EPOCHS = 120
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
NUM_WORKERS = 0
SEED = 42

# 数据增强概率 (仅雨和雾)
AUG_RAIN_PROB = 0.3
AUG_FOG_PROB = 0.3
AUG_CLEAN_PROB = 0.4  # 无增强
# ============================

# Paths
TEST6_DIR = PROJECT_ROOT / "test6"
CACHE_TRAIN = TEST6_DIR / "FusionCache_Buoy" / "train"
CACHE_VAL = TEST6_DIR / "FusionCache_Buoy" / "val"
SPLIT_DIR = TEST6_DIR / "splits_buoy"
WEIGHTS_DIR = TEST6_DIR / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best_fusion_cnn_buoy.pth"
LOG_PATH = TEST6_DIR / "train_log_buoy.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sinogram size
SINO_H, SINO_W = 2240, 180


# ============================
# Augmentation Functions
# ============================
def add_rain_to_sinogram(sino: np.ndarray, intensity_range=(0.05, 0.2),
                         num_lines=(50, 150), width=(1, 3)):
    """Add rain streak simulation to sinogram."""
    sino = sino.copy()
    h, w = sino.shape
    intensity = np.random.uniform(*intensity_range)
    n_lines = np.random.randint(*num_lines)
    for _ in range(n_lines):
        col = np.random.randint(0, w)
        start = np.random.randint(0, h // 2)
        length = np.random.randint(h // 4, h // 2)
        line_w = np.random.randint(*width)
        c_start = max(0, col - line_w // 2)
        c_end = min(w, col + line_w // 2 + 1)
        sino[start:min(h, start + length), c_start:c_end] += intensity
    return np.clip(sino, 0.0, 1.0)


def add_fog_to_sinogram(sino: np.ndarray, fog_level=(0.1, 0.4)):
    """Add fog-like haze to sinogram by reducing contrast and shifting mean."""
    sino = sino.copy()
    fog = np.random.uniform(*fog_level)
    sino = sino * (1 - fog) + fog * 0.5
    return np.clip(sino, 0.0, 1.0)


def augment_sinogram(sino_4ch: np.ndarray):
    """Apply augmentation to sinogram based on probabilities."""
    r = random.random()
    if r < AUG_RAIN_PROB:
        sino_4ch[0] = add_rain_to_sinogram(sino_4ch[0])
    elif r < AUG_RAIN_PROB + AUG_FOG_PROB:
        sino_4ch[0] = add_fog_to_sinogram(sino_4ch[0])
    return sino_4ch


# ============================
# Dataset
# ============================
class BuoyCacheDataset(Dataset):
    def __init__(self, cache_dir: str, indices: list, augment: bool = False):
        self.cache_dir = cache_dir
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        data = np.load(path, allow_pickle=True).item()
        x = data["input"].astype(np.float32)
        y = data["label"].astype(np.float32)

        if self.augment:
            x = augment_sinogram(x.copy())

        return torch.from_numpy(x), torch.from_numpy(y)


def load_split_indices(split_dir):
    train_idx = np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64)
    val_idx = np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64)
    return train_idx.tolist(), val_idx.tolist()


# ============================
# Loss Function
# ============================
class HorizonPeriodicLoss(nn.Module):
    def __init__(self, period: float = 1.0):
        super().__init__()
        self.period = period

    def forward(self, pred, target):
        rho_loss = (pred[:, 0] - target[:, 0]) ** 2
        theta_pred, theta_gt = pred[:, 1], target[:, 1]
        diff = (theta_pred - theta_gt).abs()
        theta_loss = torch.min(diff, self.period - diff) ** 2
        return (rho_loss.mean() + theta_loss.mean()) * 0.5


# ============================
# Training
# ============================
def train_one_epoch(model, loader, optimizer, scaler, loss_fn):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            pred = model(xb)
            loss = loss_fn(pred, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with autocast():
            pred = model(xb)
            loss = loss_fn(pred, yb)
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def train(model, train_dl, val_dl, loss_fn, epochs, lr, weight_decay, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=7)
    scaler = GradScaler()

    best_loss = float("inf")
    wait = 0
    log = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, scaler, loss_fn)
        val_loss = validate(model, val_dl, loss_fn)
        scheduler.step(val_loss)

        log["train"].append(train_loss)
        log["val"].append(val_loss)

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            wait = 0
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)
        else:
            wait += 1

        mark = "*" if improved else ""
        print(f"Epoch {ep:03d} | train={train_loss:.5f} | val={val_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.2e} {mark}")

        if wait >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[Saved] {LOG_PATH}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60)
    print("Train Fusion-CNN on Buoy Dataset (In-Domain)")
    print(f"Augmentation: rain={AUG_RAIN_PROB}, fog={AUG_FOG_PROB}, clean={AUG_CLEAN_PROB}")
    print("=" * 60)

    if not CACHE_TRAIN.exists() or not CACHE_VAL.exists():
        print("[Error] Cache not found. Run make_fusion_cache_buoy_train.py first.")
        return 1

    train_idx, val_idx = load_split_indices(SPLIT_DIR)
    print(f"[Train] {len(train_idx)} | [Val] {len(val_idx)}")

    ds_train = BuoyCacheDataset(str(CACHE_TRAIN), train_idx, augment=True)
    ds_val = BuoyCacheDataset(str(CACHE_VAL), val_idx, augment=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=DEVICE.startswith("cuda"))
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=DEVICE.startswith("cuda"))

    model = HorizonResNet(in_channels=4, img_h=SINO_H, img_w=SINO_W).to(DEVICE)
    loss_fn = HorizonPeriodicLoss(period=1.0)

    train(model, dl_train, dl_val, loss_fn, EPOCHS, LR, WEIGHT_DECAY, PATIENCE)
    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
