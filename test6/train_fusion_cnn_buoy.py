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
# PyCharm 配置区（与主训练策略对齐）
# ============================
SEED = 40  # 与主训练一致
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-4

# 学习率调度（与主训练一致）
PLATEAU_PATIENCE = 10
PLATEAU_FACTOR = 0.5
EARLY_STOP_PATIENCE = 100  # 实际禁用早停，保证跑满

# 数据增强：船体直线干扰（与主训练一致）
AUG_ENABLE = True
AUG_SPURIOUS_P = 0.60          # 60%概率注入虚假峰值
AUG_MAX_PEAKS = 3              # 每个样本最多3个虚假峰值
AUG_AMP_MIN, AUG_AMP_MAX = 0.15, 0.60
AUG_SIGMA_RHO = 18.0           # rho方向高斯半径
AUG_SIGMA_THETA = 1.8          # theta方向高斯半径
AUG_TARGET_CHANNELS = (0, 1, 2)  # 只影响传统Radon通道

# AMP
USE_AMP = True
GRAD_CLIP_NORM = 1.0
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


# =========================
# Data augmentation (ship-line / spurious straight-line interference)
# 与主训练代码完全一致
# =========================
def _inject_gaussian_peak(x: torch.Tensor, ch: int, rho0: int, th0: int, amp: float,
                          sigma_rho: float, sigma_th: float) -> None:
    """In-place add a small 2D Gaussian peak to x[ch] at (rho0, th0).

    This is used to simulate strong non-horizon straight-line responses (e.g., ship decks/masts)
    that can create competing peaks in the Radon domain.
    """
    _, H, W = x.shape
    # local window for efficiency
    sr = int(max(3, min(H // 2, round(3.0 * sigma_rho))))
    st = int(max(2, min(W // 2, round(3.0 * sigma_th))))
    r1 = max(0, rho0 - sr)
    r2 = min(H, rho0 + sr + 1)
    t1 = max(0, th0 - st)
    t2 = min(W, th0 + st + 1)

    rr = torch.arange(r1, r2, device=x.device, dtype=x.dtype) - float(rho0)
    tt = torch.arange(t1, t2, device=x.device, dtype=x.dtype) - float(th0)
    gr = torch.exp(-(rr * rr) / (2.0 * sigma_rho * sigma_rho))
    gt = torch.exp(-(tt * tt) / (2.0 * sigma_th * sigma_th))
    g2d = gr[:, None] * gt[None, :]
    x[ch, r1:r2, t1:t2] = torch.clamp(x[ch, r1:r2, t1:t2] + amp * g2d, 0.0, 1.0)


def augment_fusion_tensor(x: torch.Tensor) -> torch.Tensor:
    """On-the-fly augmentation for FusionCache tensors.

    x: [C, H, W], values assumed in [0,1].
    """
    if (not AUG_ENABLE) or (torch.rand(1).item() > AUG_SPURIOUS_P):
        return x

    C, H, W = x.shape
    n_peaks = int(torch.randint(1, AUG_MAX_PEAKS + 1, (1,)).item())
    for _ in range(n_peaks):
        ch = int(AUG_TARGET_CHANNELS[int(torch.randint(0, len(AUG_TARGET_CHANNELS), (1,)).item())])
        rho0 = int(torch.randint(0, H, (1,)).item())
        th0 = int(torch.randint(0, W, (1,)).item())
        amp = float(AUG_AMP_MIN + (AUG_AMP_MAX - AUG_AMP_MIN) * torch.rand(1).item())
        _inject_gaussian_peak(x, ch, rho0, th0, amp, AUG_SIGMA_RHO, AUG_SIGMA_THETA)
    return x


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
        x = torch.from_numpy(data["input"].astype(np.float32))
        y = torch.from_numpy(data["label"].astype(np.float32))

        if self.augment:
            x = augment_fusion_tensor(x)

        return x, y


def load_split_indices(split_dir):
    train_idx = np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64)
    val_idx = np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64)
    return train_idx.tolist(), val_idx.tolist()


# ============================
# Loss Function（与主训练代码对齐）
# ============================
class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0, rho_beta=0.02, theta_beta=0.02):
        super().__init__()
        self.rho_weight = float(rho_weight)
        self.theta_weight = float(theta_weight)
        self.rho_loss = nn.SmoothL1Loss(beta=rho_beta)
        self.theta_loss = nn.SmoothL1Loss(beta=theta_beta)

    def forward(self, preds, targets):
        loss_rho = self.rho_loss(preds[:, 0], targets[:, 0])

        theta_p = preds[:, 1] * np.pi
        theta_t = targets[:, 1] * np.pi
        sin_p, cos_p = torch.sin(theta_p), torch.cos(theta_p)
        sin_t, cos_t = torch.sin(theta_t), torch.cos(theta_t)

        loss_theta = self.theta_loss(sin_p, sin_t) + self.theta_loss(cos_p, cos_t)
        return self.rho_weight * loss_rho + self.theta_weight * loss_theta


# ============================
# Training（与主训练代码对齐）
# ============================
def autocast_ctx():
    if not USE_AMP or not DEVICE.startswith("cuda"):
        from contextlib import nullcontext
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=True)


def make_scaler():
    if not USE_AMP or not DEVICE.startswith("cuda"):
        return None
    return torch.amp.GradScaler(device="cuda", enabled=True)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with autocast_ctx():
            pred = model(x)
            loss = criterion(pred, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def train_one_epoch(model, loader, optimizer, scaler, criterion):
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in tqdm(loader, desc="train", ncols=80):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast_ctx():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total_loss / max(1, n)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60)
    print("Train Fusion-CNN on Buoy Dataset (Experiment 6)")
    print("=" * 60)

    if not CACHE_TRAIN.exists() or not CACHE_VAL.exists():
        print("[Error] Cache not found. Run make_fusion_cache_buoy_train.py first.")
        return 1

    train_idx, val_idx = load_split_indices(SPLIT_DIR)
    print(f"[Splits] train={len(train_idx)}, val={len(val_idx)}")

    ds_train = BuoyCacheDataset(str(CACHE_TRAIN), train_idx, augment=True)
    ds_val = BuoyCacheDataset(str(CACHE_VAL), val_idx, augment=False)

    pin = DEVICE.startswith("cuda")
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=pin)

    model = HorizonResNet(in_channels=4).to(DEVICE)
    criterion = HorizonPeriodicLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, verbose=True
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(f"[Device] {DEVICE}")
    print(f"[Train] {len(ds_train)} samples")
    print(f"[Val]   {len(ds_val)} samples")
    print("")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, scaler, criterion)
        va_loss = evaluate(model, dl_val, criterion)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}] lr={lr_now:.2e} train={tr_loss:.6f} val={va_loss:.6f}")

        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "val_loss": va_loss})
        scheduler.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> best updated: {best_val:.6f} (epoch={best_epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[Early Stop] no improvement for {EARLY_STOP_PATIENCE} epochs")
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    # Save log
    payload = {
        "dataset": "Buoy",
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Log -> {LOG_PATH}")
    print(f"[Saved] Weights -> {WEIGHTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
