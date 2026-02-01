# -*- coding: utf-8 -*-
"""
Train Fusion-CNN on SMD dataset (Experiment 6: In-Domain Training).

退化类型：只包含雨和雾（简单有效）

Inputs:
  - test6/FusionCache_SMD/{train,val,test}/
  - test6/splits_smd/

Outputs:
  - test6/weights/best_fusion_cnn_smd.pth
  - test6/train_log_smd.json

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn_model import HorizonResNet  # noqa: E402


# ============================
# PyCharm 配置区
# ============================
SEED = 2026
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-4

# 学习率调度
PLATEAU_PATIENCE = 10
PLATEAU_FACTOR = 0.5
EARLY_STOP_PATIENCE = 30  # 30个epoch不提升则停止

# 数据增强：只用雨和雾
AUG_ENABLE = True
AUG_RAIN_PROB = 0.3      # 雨的概率
AUG_FOG_PROB = 0.3       # 雾的概率
AUG_CLEAN_PROB = 0.4     # 干净图像的概率

# AMP
USE_AMP = True
GRAD_CLIP_NORM = 1.0
# ============================

# Paths
TEST6_DIR = PROJECT_ROOT / "test6"
CACHE_ROOT = TEST6_DIR / "FusionCache_SMD"
SPLIT_DIR = TEST6_DIR / "splits_smd"
WEIGHTS_DIR = TEST6_DIR / "weights"
BEST_PATH = WEIGHTS_DIR / "best_fusion_cnn_smd.pth"
LOG_PATH = TEST6_DIR / "train_log_smd.json"

FALLBACK_SHAPE = (4, 2240, 180)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ============================
# 简单退化函数（雨和雾）
# ============================
def add_rain_to_sinogram(x: torch.Tensor) -> torch.Tensor:
    """在 sinogram 上添加模拟雨的竖条纹噪声"""
    C, H, W = x.shape
    # 随机添加几条竖线（模拟雨滴在Radon域的影响）
    n_lines = random.randint(2, 8)
    for _ in range(n_lines):
        col = random.randint(0, W - 1)
        intensity = random.uniform(0.1, 0.4)
        width = random.randint(1, 3)
        for c in range(min(3, C)):  # 只影响前3个通道
            col_start = max(0, col - width // 2)
            col_end = min(W, col + width // 2 + 1)
            x[c, :, col_start:col_end] = torch.clamp(
                x[c, :, col_start:col_end] + intensity * torch.rand(H, col_end - col_start),
                0, 1
            )
    return x


def add_fog_to_sinogram(x: torch.Tensor) -> torch.Tensor:
    """在 sinogram 上添加模拟雾的低频噪声"""
    C, H, W = x.shape
    # 添加低频高斯噪声模拟雾的影响
    fog_intensity = random.uniform(0.05, 0.2)
    for c in range(min(3, C)):
        noise = torch.randn(H, W) * fog_intensity
        # 低通滤波（简单平均）
        kernel_size = random.choice([5, 7, 9])
        noise = torch.nn.functional.avg_pool2d(
            noise.unsqueeze(0).unsqueeze(0),
            kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze()
        x[c] = torch.clamp(x[c] + noise, 0, 1)
    return x


def augment_sinogram(x: torch.Tensor) -> torch.Tensor:
    """简单数据增强：只用雨和雾"""
    if not AUG_ENABLE:
        return x
    
    r = random.random()
    if r < AUG_CLEAN_PROB:
        # 保持干净
        return x
    elif r < AUG_CLEAN_PROB + AUG_RAIN_PROB:
        # 添加雨
        return add_rain_to_sinogram(x)
    else:
        # 添加雾
        return add_fog_to_sinogram(x)


# ============================
# Dataset
# ============================
class SMDCacheDataset(Dataset):
    def __init__(self, cache_dir: str, indices: list, augment: bool = False):
        self.cache_dir = cache_dir
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        
        if not os.path.exists(path):
            # Fallback
            x = torch.zeros(FALLBACK_SHAPE, dtype=torch.float32)
            y = torch.zeros(2, dtype=torch.float32)
            return x, y

        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()
        y = torch.from_numpy(data["label"]).float()
        
        if self.augment:
            x = augment_sinogram(x)
        
        return x, y


def load_split_indices(split_dir):
    return {
        "train": np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64).tolist(),
        "val": np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64).tolist(),
        "test": np.load(os.path.join(split_dir, "test_indices.npy")).astype(np.int64).tolist(),
    }


# ============================
# Loss
# ============================
class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0):
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
        self.rho_loss = nn.SmoothL1Loss(beta=0.02)
        self.theta_loss = nn.SmoothL1Loss(beta=0.02)

    def forward(self, preds, targets):
        loss_rho = self.rho_loss(preds[:, 0], targets[:, 0])
        
        theta_p = preds[:, 1] * np.pi
        theta_t = targets[:, 1] * np.pi
        sin_p, cos_p = torch.sin(theta_p), torch.cos(theta_p)
        sin_t, cos_t = torch.sin(theta_t), torch.cos(theta_t)
        
        loss_theta = self.theta_loss(sin_p, sin_t) + self.theta_loss(cos_p, cos_t)
        return self.rho_weight * loss_rho + self.theta_weight * loss_theta


# ============================
# AMP helpers
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


# ============================
# Train / Eval
# ============================
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


# ============================
# Main
# ============================
def main():
    seed_everything(SEED)
    ensure_dir(WEIGHTS_DIR)

    print("=" * 60)
    print("Train Fusion-CNN on SMD (Experiment 6)")
    print("=" * 60)

    # Check cache
    train_cache = CACHE_ROOT / "train"
    val_cache = CACHE_ROOT / "val"
    test_cache = CACHE_ROOT / "test"

    if not train_cache.exists():
        print(f"[Error] Train cache not found: {train_cache}")
        print("  Please run make_fusion_cache_smd_train.py first.")
        return 1

    # Load splits
    splits = load_split_indices(SPLIT_DIR)
    print(f"[Splits] train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Datasets
    train_ds = SMDCacheDataset(str(train_cache), splits["train"], augment=True)
    val_ds = SMDCacheDataset(str(val_cache), splits["val"], augment=False)
    test_ds = SMDCacheDataset(str(test_cache), splits["test"], augment=False)

    pin = DEVICE.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    model = HorizonResNet(in_channels=4).to(DEVICE)
    criterion = HorizonPeriodicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, verbose=True
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(f"[Device] {DEVICE}")
    print(f"[Train] {len(train_ds)} samples")
    print(f"[Val]   {len(val_ds)} samples")
    print(f"[Test]  {len(test_ds)} samples")
    print("")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        va_loss = evaluate(model, val_loader, criterion)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}] lr={lr_now:.2e} train={tr_loss:.6f} val={va_loss:.6f}")

        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "val_loss": va_loss})
        scheduler.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  -> best updated: {best_val:.6f} (epoch={best_epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[Early Stop] no improvement for {EARLY_STOP_PATIENCE} epochs")
                break

    # Final test
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    test_loss = evaluate(model, test_loader, criterion)
    print(f"[Test Loss] {test_loss:.6f}")

    # Save log
    payload = {
        "dataset": "SMD",
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "history": history,
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Log -> {LOG_PATH}")
    print(f"[Saved] Weights -> {BEST_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
