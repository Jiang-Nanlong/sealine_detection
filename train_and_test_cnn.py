import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from cnn_model import HorizonResNet


# =========================
# 全局可调参数
# =========================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"

# 传统三通道缓存（prepare_offline_data.py 生成的目录）
CACHE_DIR = r"Hashmani's Dataset/OfflineCache"

SPLIT_DIR = r"splits_musid"
SEED = 42
TEST_RATIO = 0.2

BATCH_SIZE = 8
NUM_EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

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


def count_csv_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return -1
    n = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def infer_total_len(cache_dir: str, csv_path: str = None) -> int:
    n_csv = count_csv_rows(csv_path) if csv_path else -1
    if n_csv and n_csv > 0:
        return n_csv

    max_id = -1
    for fn in os.listdir(cache_dir):
        if not fn.endswith(".npy"):
            continue
        stem = os.path.splitext(fn)[0]
        if stem.isdigit():
            max_id = max(max_id, int(stem))
    if max_id >= 0:
        return max_id + 1

    return len([f for f in os.listdir(cache_dir) if f.endswith(".npy")])


def load_or_make_split(total_len: int, test_ratio: float, seed: int, split_dir: str):
    ensure_dir(split_dir)
    train_p = os.path.join(split_dir, "train_idx.npy")
    test_p = os.path.join(split_dir, "test_idx.npy")

    if os.path.exists(train_p) and os.path.exists(test_p):
        train_idx = np.load(train_p).tolist()
        test_idx = np.load(test_p).tolist()
        if len(train_idx) + len(test_idx) == total_len:
            return train_idx, test_idx

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_len, generator=g).tolist()
    n_test = max(1, int(total_len * test_ratio))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    np.save(train_p, np.array(train_idx, dtype=np.int64))
    np.save(test_p, np.array(test_idx, dtype=np.int64))
    return train_idx, test_idx


class NpyCacheDataset(Dataset):
    def __init__(self, cache_dir: str, total_len: int, fallback_shape=(3, 2240, 180)):
        self.cache_dir = cache_dir
        self.total_len = int(total_len)
        self.fallback_shape = tuple(fallback_shape)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        if not os.path.exists(path):
            x = torch.zeros(self.fallback_shape, dtype=torch.float32)
            y = torch.zeros(2, dtype=torch.float32)
            return x, y

        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()
        y = torch.from_numpy(data["label"]).float()
        return x, y


class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0, rho_beta=0.02, theta_beta=0.02):
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
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

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE.startswith("cuda"):
            with torch.cuda.amp.autocast():
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

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(1, n)


def main():
    seed_everything(SEED)
    ensure_dir(SPLIT_DIR)

    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(f"找不到 CACHE_DIR: {CACHE_DIR}")

    total_len = infer_total_len(CACHE_DIR, CSV_PATH)
    if total_len <= 0:
        raise RuntimeError("无法推断数据集长度：请检查 CSV_PATH 或 CACHE_DIR。")

    train_idx, test_idx = load_or_make_split(total_len, TEST_RATIO, SEED, SPLIT_DIR)

    # 推断 fallback_shape
    fallback_shape = (3, 2240, 180)
    for i in range(min(total_len, 500)):
        p = os.path.join(CACHE_DIR, f"{i}.npy")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            fallback_shape = tuple(d["input"].shape)
            break

    dataset = NpyCacheDataset(CACHE_DIR, total_len, fallback_shape=fallback_shape)
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    def worker_init_fn(worker_id):
        seed = SEED + worker_id
        np.random.seed(seed)
        random.seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    x0, _ = dataset[train_idx[0]]
    in_ch = int(x0.shape[0])
    model = HorizonResNet(in_channels=in_ch).to(DEVICE)

    criterion = HorizonPeriodicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))

    best = float("inf")
    best_path = os.path.join(SPLIT_DIR, "best_offline_cnn.pth")

    print(f"[INFO] total_len={total_len} | train={len(train_set)} test={len(test_set)} | in_ch={in_ch}")
    print(f"[INFO] split saved at: {os.path.join(SPLIT_DIR, 'train_idx.npy')} / test_idx.npy")
    print(f"[INFO] model save path: {best_path}")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        te_loss = evaluate(model, test_loader, criterion)

        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  train_loss={tr_loss:.6f}  test_loss={te_loss:.6f}")

        if te_loss < best:
            best = te_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> best updated: {best:.6f}")

    print(f"Done. Best test_loss={best:.6f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
