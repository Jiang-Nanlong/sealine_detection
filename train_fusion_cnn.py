# train_fusion_cnn.py
import os
import random
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cnn_model import HorizonResNet

# =========================
# Config
# =========================
# 指向刚才生成的缓存目录
CACHE_ROOT = r"Hashmani's Dataset/FusionCache_C2"
TRAIN_CACHE_DIR = os.path.join(CACHE_ROOT, "train")
VAL_CACHE_DIR   = os.path.join(CACHE_ROOT, "val")
TEST_CACHE_DIR  = os.path.join(CACHE_ROOT, "test")

SPLIT_DIR = r"splits_musid"

SEED = 42
BATCH_SIZE = 16 # CNN 比较轻，可以加大 Batch Size
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_split_indices(split_dir: str):
    primary = {
        "train": os.path.join(split_dir, "train_indices.npy"),
        "val":   os.path.join(split_dir, "val_indices.npy"),
        "test":  os.path.join(split_dir, "test_indices.npy"),
    }
    if all(os.path.exists(p) for p in primary.values()):
        return {
            "train": np.load(primary["train"]).astype(np.int64).tolist(),
            "val":   np.load(primary["val"]).astype(np.int64).tolist(),
            "test":  np.load(primary["test"]).astype(np.int64).tolist(),
        }
    raise FileNotFoundError(f"Indices not found in {split_dir}")

class SplitCacheDataset(Dataset):
    def __init__(self, cache_dir, indices):
        self.cache_dir = cache_dir
        self.indices = list(indices)
        # 预检
        self.valid_indices = []
        for idx in self.indices:
            if os.path.exists(os.path.join(self.cache_dir, f"{idx}.npy")):
                self.valid_indices.append(idx)
        print(f"Dataset {os.path.basename(cache_dir)}: {len(self.valid_indices)}/{len(self.indices)} files found.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()   # [4, 2240, 180]
        y = torch.from_numpy(data["label"]).float()   # [2]
        return x, y

class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=10.0): # 加大 theta 权重，因为角度很关键
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
        self.l1 = nn.SmoothL1Loss()

    def forward(self, preds, targets):
        # rho: Linear
        loss_rho = self.l1(preds[:, 0], targets[:, 0])

        # theta: Periodic (sin/cos distance)
        # targets[:, 1] is 0~1 mapping to 0~180 degrees
        theta_p = preds[:, 1] * np.pi
        theta_t = targets[:, 1] * np.pi
        
        # distance on circle
        diff = torch.abs(theta_p - theta_t)
        diff = torch.min(diff, np.pi - diff) # 考虑周期性虽非必要(0-180不闭合)，但作为角度回归好习惯
        # 对于海天线，0度和180度其实物理上是反的，所以这里简单用 L1 即可
        # 但要注意 0 和 1 (0度和180度) 在数值上很远。
        # 鉴于海天线通常是水平的 (90度/0.5附近)，边界问题不大。
        loss_theta = self.l1(preds[:, 1], targets[:, 1])

        return self.rho_weight * loss_rho + self.theta_weight * loss_theta

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(1, n)

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        if USE_AMP and DEVICE=="cuda":
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)

def main():
    seed_everything(SEED)
    splits = load_split_indices(SPLIT_DIR)

    train_ds = SplitCacheDataset(TRAIN_CACHE_DIR, splits["train"])
    val_ds   = SplitCacheDataset(VAL_CACHE_DIR,   splits["val"])
    
    # 自动适配通道数 (通常是 4)
    if len(train_ds) > 0:
        in_ch = train_ds[0][0].shape[0]
    else:
        in_ch = 4 

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = HorizonResNet(in_channels=in_ch).to(DEVICE)
    criterion = HorizonPeriodicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(USE_AMP and DEVICE=="cuda"))

    best_val = float("inf")
    best_path = os.path.join(SPLIT_DIR, "best_fusion_cnn.pth")

    print(f"Start Training CNN. In Channels: {in_ch}")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        va_loss = evaluate(model, val_loader, criterion)

        print(f"Ep {epoch}/{NUM_EPOCHS}: Train={tr_loss:.6f} Val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print("  -> New Best!")

    print(f"Done. Best Val Loss: {best_val:.6f}")

if __name__ == "__main__":
    main()