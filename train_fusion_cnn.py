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
# 全局可调参数（PyCharm 里改这里最方便）
# =========================
# 三个 split 的 cache 目录（make_fusion_cache.py 生成的 SAVE_ROOT 下的子目录）
TRAIN_CACHE_DIR = r"Hashmani's Dataset/FusionCache_split/train"
VAL_CACHE_DIR   = r"Hashmani's Dataset/FusionCache_split/val"
TEST_CACHE_DIR  = r"Hashmani's Dataset/FusionCache_split/test"

# split 索引目录（优先读取 *_indices.npy）
SPLIT_DIR = r"splits_musid"

# 训练参数
SEED = 42
BATCH_SIZE = 6
NUM_EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================


# =========================
# Utils
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_split_indices(split_dir: str):
    """
    优先读取：
      train_indices.npy / val_indices.npy / test_indices.npy
    兼容读取：
      train_idx.npy / test_idx.npy（如果你只保存了两份）
    """
    primary = {
        "train": os.path.join(split_dir, "train_indices.npy"),
        "val":   os.path.join(split_dir, "val_indices.npy"),
        "test":  os.path.join(split_dir, "test_indices.npy"),
    }
    alt = {
        "train": os.path.join(split_dir, "train_idx.npy"),
        "test":  os.path.join(split_dir, "test_idx.npy"),
    }

    if all(os.path.exists(p) for p in primary.values()):
        return {
            "train": np.load(primary["train"]).astype(np.int64).tolist(),
            "val":   np.load(primary["val"]).astype(np.int64).tolist(),
            "test":  np.load(primary["test"]).astype(np.int64).tolist(),
        }

    if os.path.exists(alt["train"]) and os.path.exists(alt["test"]):
        return {
            "train": np.load(alt["train"]).astype(np.int64).tolist(),
            "val":   [],  # 没有 val 就留空
            "test":  np.load(alt["test"]).astype(np.int64).tolist(),
        }

    raise FileNotFoundError(
        "找不到 split 索引文件。请确认 SPLIT_DIR 下存在：\n"
        f"  {primary['train']}\n  {primary['val']}\n  {primary['test']}\n"
        "或至少存在：\n"
        f"  {alt['train']}\n  {alt['test']}\n"
    )


def infer_fallback_shape(cache_dir: str, indices: list, default_shape=(4, 2240, 180), scan_limit=500):
    """
    优先从该 split 的真实样本推断 input shape；否则回退到 default_shape。
    """
    if not os.path.isdir(cache_dir):
        return default_shape

    cnt = 0
    for idx in indices:
        p = os.path.join(cache_dir, f"{idx}.npy")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            if "input" in d:
                return tuple(d["input"].shape)
        cnt += 1
        if cnt >= scan_limit:
            break
    return default_shape


# =========================
# Dataset：按 split 的 indices 读取 {idx}.npy
# =========================
class SplitCacheDataset(Dataset):
    def __init__(self, cache_dir: str, indices: list, fallback_shape=(4, 2240, 180), strict_missing: bool = False):
        self.cache_dir = cache_dir
        self.indices = list(indices)
        self.fallback_shape = tuple(fallback_shape)
        self.strict_missing = strict_missing

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        if not os.path.exists(path):
            if self.strict_missing:
                raise FileNotFoundError(f"Missing cache file: {path}")
            x = torch.zeros(self.fallback_shape, dtype=torch.float32)
            y = torch.zeros(2, dtype=torch.float32)
            return x, y

        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()   # [C,H,W]
        y = torch.from_numpy(data["label"]).float()   # [2]
        return x, y


# =========================
# Loss（theta 用 sin/cos 做周期距离）
# =========================
class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0, rho_beta=0.02, theta_beta=0.02):
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
        self.rho_loss = nn.SmoothL1Loss(beta=rho_beta)
        self.theta_loss = nn.SmoothL1Loss(beta=theta_beta)

    def forward(self, preds, targets):
        # rho: 线性 0~1
        loss_rho = self.rho_loss(preds[:, 0], targets[:, 0])

        # theta: 0~1 -> 0~pi（半周期），用 sin/cos 表示环形距离
        theta_p = preds[:, 1] * np.pi
        theta_t = targets[:, 1] * np.pi
        sin_p, cos_p = torch.sin(theta_p), torch.cos(theta_p)
        sin_t, cos_t = torch.sin(theta_t), torch.cos(theta_t)

        loss_theta = self.theta_loss(sin_p, sin_t) + self.theta_loss(cos_p, cos_t)
        return self.rho_weight * loss_rho + self.theta_weight * loss_theta


# =========================
# Train / Eval
# =========================
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

    # 1) 读 split indices
    splits = load_split_indices(SPLIT_DIR)
    train_indices = splits["train"]
    val_indices = splits.get("val", [])
    test_indices = splits["test"]

    if not os.path.isdir(TRAIN_CACHE_DIR):
        raise FileNotFoundError(f"找不到 TRAIN_CACHE_DIR: {TRAIN_CACHE_DIR}")
    if not os.path.isdir(VAL_CACHE_DIR):
        raise FileNotFoundError(f"找不到 VAL_CACHE_DIR: {VAL_CACHE_DIR}")
    if not os.path.isdir(TEST_CACHE_DIR):
        print(f"[WARN] 找不到 TEST_CACHE_DIR: {TEST_CACHE_DIR}（将跳过最终 test 评估）")

    if len(train_indices) == 0:
        raise RuntimeError("train_indices 为空，请检查 SPLIT_DIR 下的索引文件。")
    if len(val_indices) == 0:
        print("[WARN] val_indices 为空：将用 test 当作 val（不推荐，最好生成 val_indices.npy）")
        val_indices = list(test_indices)

    # 2) 推断各 split 的 input shape（避免缺失样本返回全 0 时通道数不对）
    train_shape = infer_fallback_shape(TRAIN_CACHE_DIR, train_indices, default_shape=(4, 2240, 180))
    val_shape   = infer_fallback_shape(VAL_CACHE_DIR,   val_indices,   default_shape=train_shape)
    test_shape  = infer_fallback_shape(TEST_CACHE_DIR,  test_indices,  default_shape=train_shape)

    # 3) 构建数据集/加载器
    train_ds = SplitCacheDataset(TRAIN_CACHE_DIR, train_indices, fallback_shape=train_shape, strict_missing=False)
    val_ds   = SplitCacheDataset(VAL_CACHE_DIR,   val_indices,   fallback_shape=val_shape,   strict_missing=False)
    test_ds  = SplitCacheDataset(TEST_CACHE_DIR,  test_indices,  fallback_shape=test_shape,  strict_missing=False) \
              if os.path.isdir(TEST_CACHE_DIR) else None

    def worker_init_fn(worker_id):
        seed = SEED + worker_id
        np.random.seed(seed)
        random.seed(seed)

    pin = DEVICE.startswith("cuda")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    ) if test_ds is not None else None

    # 4) 建模：自动匹配通道数
    x0, _ = train_ds[0]
    in_ch = int(x0.shape[0])
    model = HorizonResNet(in_channels=in_ch).to(DEVICE)

    criterion = HorizonPeriodicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.startswith("cuda")))

    # 5) 保存路径：best 依据 val
    best_val = float("inf")
    best_path = os.path.join(SPLIT_DIR, "best_fusion_cnn.pth")

    print(f"[INFO] in_ch={in_ch}, device={DEVICE}, amp={USE_AMP}")
    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds) if test_ds else 0}")
    print(f"[INFO] cache_dirs:\n  train={TRAIN_CACHE_DIR}\n  val  ={VAL_CACHE_DIR}\n  test ={TEST_CACHE_DIR}")
    print(f"[INFO] best model will be saved to: {best_path}")

    # 6) 训练：每 epoch 评估 val，并按 val 保存 best
    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        va_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> best updated (val): {best_val:.6f}")

    # 7) 最终：加载 best，在 test 上评估一次
    final_test = None
    if test_loader is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        final_test = evaluate(model, test_loader, criterion)
        print(f"[FINAL] best_val={best_val:.6f}  test_loss={final_test:.6f}")
    else:
        print(f"[FINAL] best_val={best_val:.6f}  (test skipped)")

    # 8) 写一个结果文件，方便你实验对比（特别是不同 MORPH_CLOSE）
    out_json = os.path.join(SPLIT_DIR, "fusion_cnn_result.json")
    payload = {
        "train_cache_dir": TRAIN_CACHE_DIR,
        "val_cache_dir": VAL_CACHE_DIR,
        "test_cache_dir": TEST_CACHE_DIR,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "in_channels": in_ch,
        "best_val_loss": best_val,
        "test_loss": final_test,
        "best_model_path": best_path,
        "history": history,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_json}")


if __name__ == "__main__":
    main()
