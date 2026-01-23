# train_fusion_cnn_channel_ablation.py
# ------------------------------------------------------------
# Train HorizonResNet on offline FusionCache produced by make_fusion_cache.py,
# but using only a subset of the cached channels (ablation study).
#
# Cached channels (from make_fusion_cache.py):
#   ch0, ch1, ch2 : traditional multi-scale Radon features (scales=[1,2,3])
#   ch3           : Radon of segmentation-edge mask
#
# Example:
#   # train on 3 traditional channels
#   python train_fusion_cnn_channel_ablation.py --mode trad3
#
#   # train on segmentation-edge channel only
#   python train_fusion_cnn_channel_ablation.py --mode seg1
#
# You can also specify channels explicitly:
#   python train_fusion_cnn_channel_ablation.py --mode custom --channels 0,2,3 --tag custom_023
# ------------------------------------------------------------

import argparse
import json
import os
import random
from typing import List, Sequence

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cnn_model import HorizonResNet

# =========================
# PyCharm quick-run config
# =========================
USE_PYCHARM_CONFIG = True   # <- 想用命令行就改 False

PY_MODE = "trad3"           # "trad3" or "seg1" or "custom"
PY_CHANNELS = "0,1,2"       # only used when PY_MODE == "custom"
PY_TAG = ""                 # e.g. "abl_trad3" / "abl_seg1" (empty -> use default)

PY_CACHE_ROOT = r"Hashmani's Dataset/FusionCache_1024x576"
PY_SPLIT_DIR  = r"splits_musid"

PY_NUM_WORKERS = 4          # 如果你仍然报多进程错误，就先改成 0
PY_BATCH_SIZE  = 16
PY_EPOCHS      = 100
PY_LR          = 2e-4
PY_WD          = 1e-4

PY_USE_AMP     = True
PY_AUG         = False       # None=按默认(trad3开/seg1关), True=强制开, False=强制关
PY_GRAD_CLIP   = 1.0
PY_SEED        = 40

# =========================
# Reproducibility
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Split loader
# =========================
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
    raise FileNotFoundError(f"Cannot find split indices in {split_dir}")


# =========================
# Data augmentation (spurious straight-line interference in Radon domain)
#   NOTE:
#     - This augmentation is meaningful for traditional Radon channels (0/1/2).
#     - For seg-only training (seg1), we disable it by default.
# =========================
def _inject_gaussian_peak(
    x: torch.Tensor,
    ch: int,
    rho0: int,
    th0: int,
    amp: float,
    sigma_rho: float,
    sigma_th: float,
) -> None:
    """In-place add a small 2D Gaussian peak to x[ch] at (rho0, th0)."""
    _, H, W = x.shape
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


def augment_radon_tensor(
    x: torch.Tensor,
    target_channels: Sequence[int],
    spurious_p: float = 0.60,
    max_peaks: int = 3,
    amp_min: float = 0.15,
    amp_max: float = 0.60,
    sigma_rho: float = 18.0,
    sigma_th: float = 1.8,
) -> torch.Tensor:
    """
    x: [C, H, W] in [0,1]
    target_channels: channel indices in *current sliced tensor x*
    """
    if len(target_channels) == 0:
        return x
    if torch.rand(1).item() > spurious_p:
        return x

    C, H, W = x.shape
    n_peaks = int(torch.randint(1, max_peaks + 1, (1,)).item())
    for _ in range(n_peaks):
        ch = int(target_channels[int(torch.randint(0, len(target_channels), (1,)).item())])
        ch = max(0, min(C - 1, ch))
        rho0 = int(torch.randint(0, H, (1,)).item())
        th0 = int(torch.randint(0, W, (1,)).item())
        amp = float(amp_min + (amp_max - amp_min) * torch.rand(1).item())
        _inject_gaussian_peak(x, ch, rho0, th0, amp, sigma_rho, sigma_th)
    return x


# =========================
# Dataset: load from cache and slice channels
# =========================
class SplitCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        indices: list,
        channel_indices: Sequence[int],
        fallback_hw=(2240, 180),
        strict_missing: bool = True,
        augment: bool = False,
        augment_target_channels: Sequence[int] = (),
    ):
        self.cache_dir = cache_dir
        self.indices = list(indices)
        self.channel_indices = list(channel_indices)
        self.fallback_hw = tuple(fallback_hw)
        self.strict_missing = bool(strict_missing)
        self.augment = bool(augment)
        self.augment_target_channels = list(augment_target_channels)

        if len(self.channel_indices) == 0:
            raise ValueError("channel_indices cannot be empty")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        path = os.path.join(self.cache_dir, f"{idx}.npy")
        if not os.path.exists(path):
            if self.strict_missing:
                raise FileNotFoundError(f"Missing cache file: {path}")
            C = len(self.channel_indices)
            H, W = self.fallback_hw
            x = torch.zeros((C, H, W), dtype=torch.float32)
            y = torch.zeros(2, dtype=torch.float32)
            return x, y

        data = np.load(path, allow_pickle=True).item()
        x_all = torch.from_numpy(data["input"]).float()   # [4,H,W]
        y = torch.from_numpy(data["label"]).float()       # [2]

        # slice channels (based on ORIGINAL cache ordering)
        x = x_all[self.channel_indices, :, :].contiguous()

        # augmentation (on sliced tensor)
        if self.augment:
            x = augment_radon_tensor(x, target_channels=self.augment_target_channels)

        return x, y


# =========================
# Loss (rho linear, theta periodic)
# =========================
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


# =========================
# AMP helpers
# =========================
try:
    import torch.amp as torch_amp
    _HAS_TORCH_AMP = True
except Exception:
    _HAS_TORCH_AMP = False
    torch_amp = None


def autocast_ctx(use_amp: bool, device: str):
    if (not use_amp) or (not device.startswith("cuda")):
        from contextlib import nullcontext
        return nullcontext()
    if _HAS_TORCH_AMP:
        return torch_amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_scaler(use_amp: bool, device: str):
    if (not use_amp) or (not device.startswith("cuda")):
        return None
    if _HAS_TORCH_AMP:
        try:
            return torch_amp.GradScaler(device="cuda", enabled=True)
        except TypeError:
            return torch_amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion, device: str, use_amp: bool):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx(use_amp, device):
            pred = model(x)
            loss = criterion(pred, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device: str, use_amp: bool, grad_clip_norm: float):
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in tqdm(loader, desc="train", ncols=90):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast_ctx(use_amp, device):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total_loss / max(1, n)


# =========================
# Args / Main
# =========================
def parse_channels(ch_str: str) -> List[int]:
    ch_str = ch_str.strip()
    if not ch_str:
        raise ValueError("--channels cannot be empty")
    out = []
    for part in ch_str.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    if len(out) == 0:
        raise ValueError("--channels parsed to empty list")
    return out


def main():
    parser = argparse.ArgumentParser(description="Train ablation CNN on FusionCache with selected channels")
    parser.add_argument("--cache_root", type=str, default=r"Hashmani's Dataset/FusionCache_1024x576")
    parser.add_argument("--split_dir", type=str, default=r"splits_musid")

    parser.add_argument("--mode", type=str, default="trad3", choices=["trad3", "seg1", "custom"],
                        help="trad3: use [0,1,2]; seg1: use [3]; custom: use --channels")
    parser.add_argument("--channels", type=str, default="", help="When --mode=custom, e.g. 0,1,2")
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to output filenames")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true", default=False, help="Disable AMP")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--plateau_patience", type=int, default=10)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--early_stop_patience", type=int, default=100)

    parser.add_argument("--augment", action="store_true", default=None,
                        help="Enable spurious-peak augmentation (default: on for trad3, off for seg1)")
    parser.add_argument("--no_augment", action="store_true", default=False,
                        help="Force disable augmentation")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.use_amp) and (not args.no_amp)

    seed_everything(args.seed)
    ensure_dir(args.split_dir)

    # resolve channels in ORIGINAL cache order
    if args.mode == "trad3":
        channels = [0, 1, 2]
        default_tag = "trad3"
    elif args.mode == "seg1":
        channels = [3]
        default_tag = "seg1"
    else:
        channels = parse_channels(args.channels)
        default_tag = "custom_" + "".join(str(c) for c in channels)

    tag = args.tag.strip() or default_tag

    cache_root = args.cache_root
    train_cache_dir = os.path.join(cache_root, "train")
    val_cache_dir   = os.path.join(cache_root, "val")
    test_cache_dir  = os.path.join(cache_root, "test")

    for d in [train_cache_dir, val_cache_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Cache dir not found: {d}. Please run make_fusion_cache.py first.")

    splits = load_split_indices(args.split_dir)
    train_indices = splits["train"]
    val_indices = splits.get("val", [])
    test_indices = splits.get("test", [])

    if len(val_indices) == 0:
        raise RuntimeError("val_indices is empty.")

    # augmentation default:
    #   - trad3: enabled by default; apply to all channels in sliced tensor (0..C-1)
    #   - seg1 : disabled by default
    if args.no_augment:
        use_aug = False
    elif args.augment is not None:
        use_aug = True
    else:
        use_aug = (args.mode == "trad3")

    # For sliced input, augmentation target channels are indices within sliced tensor:
    # trad3 -> [0,1,2], seg1 -> [] (disabled)
    if use_aug:
        aug_targets = list(range(len(channels)))
    else:
        aug_targets = []

    # fallback HW (fixed by your cache)
    fallback_hw = (2240, 180)

    train_ds = SplitCacheDataset(
        train_cache_dir, train_indices,
        channel_indices=channels,
        fallback_hw=fallback_hw,
        strict_missing=True,
        augment=use_aug,
        augment_target_channels=aug_targets,
    )
    val_ds = SplitCacheDataset(
        val_cache_dir, val_indices,
        channel_indices=channels,
        fallback_hw=fallback_hw,
        strict_missing=True,
        augment=False,
    )
    test_ds = SplitCacheDataset(
        test_cache_dir, test_indices,
        channel_indices=channels,
        fallback_hw=fallback_hw,
        strict_missing=True,
        augment=False,
    ) if os.path.isdir(test_cache_dir) else None

    pin = device.startswith("cuda")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin
    ) if test_ds is not None else None

    # model
    in_ch = len(channels)
    model = HorizonResNet(in_channels=in_ch).to(device)

    criterion = HorizonPeriodicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(use_amp, device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience, verbose=True
    )

    best_path = os.path.join(args.split_dir, f"best_fusion_cnn_1024x576_{tag}.pth")
    out_json = os.path.join(args.split_dir, f"train_fusion_cnn_1024x576_{tag}.json")

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(f"[INFO] mode={args.mode} tag={tag} channels(original)={channels} in_channels={in_ch}")
    print(f"[INFO] Train={len(train_ds)}  Val={len(val_ds)}  Test={(len(test_ds) if test_ds is not None else 0)}")
    print(f"[INFO] AMP={use_amp}  AUG={use_aug}")
    print(f"[INFO] best_path={best_path}")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp, args.grad_clip)
        va_loss = evaluate(model, val_loader, criterion, device, use_amp)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:03d}/{args.epochs}]  lr={lr_now:.2e}  train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "val_loss": va_loss})
        scheduler.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> best updated: {best_val:.6f} (epoch={best_epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f"[EARLY STOP] no improvement for {args.early_stop_patience} epochs. best_epoch={best_epoch}")
                break

    # final eval on test
    final_test = None
    if test_loader is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        final_test = evaluate(model, test_loader, criterion, device, use_amp)
        print(f"[FINAL TEST] loss={final_test:.6f}")

    payload = {
        "tag": tag,
        "mode": args.mode,
        "channels_original": channels,
        "in_channels": len(channels),
        "cache_root": args.cache_root,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_loss": final_test,
        "history": history,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Done] best_path={best_path}")
    print(f"[Done] log_json={out_json}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    if USE_PYCHARM_CONFIG:
        argv = [
            "--cache_root", PY_CACHE_ROOT,
            "--split_dir",  PY_SPLIT_DIR,
            "--mode",       PY_MODE,
            "--batch_size", str(PY_BATCH_SIZE),
            "--num_workers", str(PY_NUM_WORKERS),
            "--epochs",     str(PY_EPOCHS),
            "--lr",         str(PY_LR),
            "--weight_decay", str(PY_WD),
            "--seed",       str(PY_SEED),
            "--grad_clip",  str(PY_GRAD_CLIP),
        ]

        if PY_TAG:
            argv += ["--tag", PY_TAG]

        if PY_MODE == "custom":
            argv += ["--channels", PY_CHANNELS]

        if not PY_USE_AMP:
            argv += ["--no_amp"]

        # augmentation switch
        if PY_AUG is True:
            argv += ["--augment"]
        elif PY_AUG is False:
            argv += ["--no_augment"]

        # 关键：用“构造出来的 argv”喂给 argparse
        import sys
        sys.argv = ["train_fusion_cnn_channel_ablation.py"] + argv

    main()
