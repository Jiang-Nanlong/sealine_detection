# -*- coding: utf-8 -*-
"""
train_joint_finetune.py

联合微调脚本：将 UNet (bridge_conv) 和 ResNet 端到端联合训练。

数据流:
  原始图像 → UNet → bridge_feats (3ch) → 可微 Radon → 3ch sinograms (在线计算, 有梯度)
  离线缓存 → 传统 4ch sinograms (固定, 无梯度)
  拼接 7ch → ResNet → (ρ, θ) → HorizonPeriodicLoss

核心思路:
  - 传统 4 通道从 Stage 2 的缓存中读取 (不可微操作已经完成, 无需重新计算)
  - Bridge 3 通道在线计算, 梯度可以从 ResNet 回传到 UNet 的 bridge_conv 和复原分支
  - 选择性解冻: bridge_conv + UNet 复原分支后半 + ResNet 前半

前置条件:
  - Stage 1 完成: UNet 权重已训练 (weights_new/rghnet_best_c2.pth)
  - Stage 2 完成: FusionCache 已生成 (7ch .npy 文件)
  - Stage 3 完成: ResNet 权重已训练 (weights_new/best_fusion_cnn_1024x576.pth)

输出:
  - weights_new/joint_unet_best.pth   (联合微调后的 UNet 权重)
  - weights_new/joint_cnn_best.pth    (联合微调后的 ResNet 权重)
  - splits_musid/train_joint_finetune.json (训练日志)
"""

import os
import json
import math
import random
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from unet_model import RestorationGuidedHorizonNet
from cnn_model import HorizonResNet

# ============================
# 配置
# ============================
# 路径
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
SPLIT_DIR = r"splits_musid"
CACHE_ROOT = r"Hashmani's Dataset/FusionCache_new_1024x576"

# 预训练权重
UNET_WEIGHTS = r"weights_new/rghnet_best_c2.pth"
CNN_WEIGHTS = r"weights_new/best_fusion_cnn_1024x576.pth"
DCE_WEIGHTS = r"weights/Epoch99.pth"

# 输出
JOINT_UNET_BEST = r"weights_new/joint_unet_best.pth"
JOINT_CNN_BEST = r"weights_new/joint_cnn_best.pth"
OUT_JSON = os.path.join(SPLIT_DIR, "train_joint_finetune.json")

# UNet 输入尺寸
UNET_IN_W = 1024
UNET_IN_H = 576

# Radon sinogram 统一尺寸
RESIZE_H = 2240
RESIZE_W = 180

# 训练参数
SEED = 42
BATCH_SIZE = 2          # 端到端显存较大, 小 batch
NUM_EPOCHS = 20
NUM_WORKERS = 2

# 学习率
LR_BRIDGE = 1e-4        # bridge_conv: 学习率最高
LR_UNET = 1e-5          # UNet 解冻部分: 小学习率
LR_RESNET = 1e-5        # ResNet 解冻部分: 小学习率
WEIGHT_DECAY = 1e-4

USE_AMP = True
GRAD_CLIP_NORM = 1.0

PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# 工具函数
# ============================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _read_image_any_ext(img_dir, img_stem_or_name):
    """根据文件名 stem 自动尝试多种扩展名读取图像"""
    name = str(img_stem_or_name)
    lower = name.lower()
    candidates = [name] if (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")) else []
    candidates += [name + ".JPG", name + ".jpg", name + ".jpeg", name + ".png"]
    for fn in candidates:
        p = os.path.join(img_dir, fn)
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is not None:
                return im
    return None


# ============================
# 可微 Radon 变换 (纯 PyTorch, 支持梯度回传)
# ============================
class DifferentiableRadon(nn.Module):
    """
    基于 affine_grid + grid_sample 的可微 Radon 变换。
    输入: (B, 1, H, W) 的 tensor
    输出: (B, 1, diagonal, num_angles) 的 sinogram tensor
    
    与 gradient_radon.py 中 _radon_gpu 的算法完全一致,
    但全部使用 PyTorch 操作, 保留计算图以支持反向传播。
    """

    def __init__(self, num_angles=180, target_h=2240, target_w=180):
        super().__init__()
        self.num_angles = num_angles
        self.target_h = target_h
        self.target_w = target_w
        # 预计算角度 (弧度)
        theta_deg = torch.linspace(0.0, 180.0, num_angles, dtype=torch.float32)
        self.register_buffer("theta_rad", torch.deg2rad(theta_deg))

    def forward(self, x):
        """
        x: (B, 1, H, W) — 单通道2D特征图, 值域 [0, 1]
        return: (B, 1, target_h, target_w) — 归一化后的 sinogram
        """
        B, C, H, W = x.shape
        assert C == 1, "Radon 输入必须是单通道"

        diagonal = int(math.ceil(math.sqrt(H * H + W * W)))
        pad_h = (diagonal - H) // 2
        pad_w = (diagonal - W) // 2
        pad_bottom = diagonal - H - pad_h
        pad_right = diagonal - W - pad_w

        # Padding 到正方形
        x_padded = F.pad(x, (pad_w, pad_right, pad_h, pad_bottom))  # (B, 1, D, D)

        sinogram_cols = []
        for angle in self.theta_rad:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # 旋转矩阵 (2, 3) — 对每个 batch 元素相同
            rot_mat = torch.stack([
                torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)]),
                torch.stack([sin_a,  cos_a, torch.zeros_like(cos_a)]),
            ]).unsqueeze(0).expand(B, -1, -1).to(x.device)  # (B, 2, 3)

            grid = F.affine_grid(rot_mat, x_padded.size(), align_corners=False)
            rotated = F.grid_sample(x_padded, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)

            # 沿 Height 方向求和 → 投影
            projection = rotated.sum(dim=2)  # (B, 1, D)
            sinogram_cols.append(projection)

        # (B, 1, D) × num_angles → stack → (B, 1, num_angles, D) → permute → (B, 1, D, num_angles)
        sinogram = torch.stack(sinogram_cols, dim=2)  # (B, 1, num_angles, D)
        sinogram = sinogram.permute(0, 1, 3, 2)       # (B, 1, D, num_angles)

        # 归一化到 [0, 1]
        # 对每个样本独立归一化
        for b in range(B):
            s = sinogram[b, 0]
            smin, smax = s.min(), s.max()
            if smax - smin > 1e-6:
                sinogram[b, 0] = (s - smin) / (smax - smin)
            else:
                sinogram[b, 0] = torch.zeros_like(s)

        # Resize 到统一尺寸
        sinogram = F.interpolate(sinogram, size=(self.target_h, self.target_w),
                                 mode="bilinear", align_corners=False)

        return sinogram  # (B, 1, target_h, target_w)


# ============================
# Loss
# ============================
class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0, rho_beta=0.02, theta_beta=0.02):
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
        self.rho_loss = nn.SmoothL1Loss(beta=rho_beta)
        self.theta_loss = nn.SmoothL1Loss(beta=theta_beta)

    def forward(self, preds, targets):
        loss_rho = self.rho_loss(preds[:, 0], targets[:, 0])
        theta_p = preds[:, 1] * math.pi
        theta_t = targets[:, 1] * math.pi
        loss_theta = (self.theta_loss(torch.sin(theta_p), torch.sin(theta_t))
                      + self.theta_loss(torch.cos(theta_p), torch.cos(theta_t)))
        return self.rho_weight * loss_rho + self.theta_weight * loss_theta


# ============================
# Dataset: 读原图 + 读缓存中的传统4通道 + label
# ============================
class JointFinetuneDataset(Dataset):
    """
    每个样本返回:
      - img_rgb: (3, H, W) — 原始图像 (UNet 输入, 用于在线计算 bridge_feats)
      - trad_4ch: (4, 2240, 180) — 来自缓存的传统 4 通道 sinogram (固定)
      - label: (2,) — [rho_norm, theta_norm]
    """

    def __init__(self, cache_dir, indices, df, img_dir, unet_w, unet_h):
        self.cache_dir = cache_dir
        self.indices = list(indices)
        self.df = df
        self.img_dir = img_dir
        self.unet_w = unet_w
        self.unet_h = unet_h

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        # --- 从缓存读取传统 4 通道 sinogram 和 label ---
        cache_path = os.path.join(self.cache_dir, f"{idx}.npy")
        data = np.load(cache_path, allow_pickle=True).item()
        full_input = data["input"]  # (7, 2240, 180) — 前4是传统, 后3是bridge
        label = data["label"]       # (2,)

        # 只取前 4 通道 (传统 Radon), bridge 3 通道将在线计算
        trad_4ch = torch.from_numpy(full_input[:4]).float()
        label = torch.from_numpy(label).float()

        # --- 读取原始图像 ---
        row = self.df.iloc[idx]
        img_name = str(row.iloc[0])
        bgr = _read_image_any_ext(self.img_dir, img_name)

        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.unet_w, self.unet_h), interpolation=cv2.INTER_AREA)
            img_rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)
        else:
            # fallback: 黑图
            img_rgb = torch.zeros(3, self.unet_h, self.unet_w, dtype=torch.float32)

        return img_rgb, trad_4ch, label


# ============================
# 参数分组
# ============================
def build_param_groups(unet, cnn):
    """
    选择性解冻, 分 3 组不同学习率:
      1. bridge_conv — LR_BRIDGE (最高)
      2. UNet 复原分支后半部分 (rest_fuse, rest_strip, ca2~ca5) — LR_UNET
      3. ResNet 前端 (conv1, bn1, layer1) + temperature — LR_RESNET
    其余全部冻结。
    """

    # 先冻结所有参数
    for p in unet.parameters():
        p.requires_grad = False
    for p in cnn.parameters():
        p.requires_grad = False

    # --- 组 1: bridge_conv ---
    bridge_params = []
    for p in unet.bridge_conv.parameters():
        p.requires_grad = True
        bridge_params.append(p)

    # --- 组 2: UNet 复原分支后半 ---
    unet_unfreeze_modules = [
        unet.rest_fuse, unet.rest_strip, unet.rest_out,
        unet.ca2, unet.ca3, unet.ca4, unet.ca5,
        unet.film_head,  # [L2] FiLM 头: 们要特实现 L2 就必须训练它
    ]
    unet_params = []
    for module in unet_unfreeze_modules:
        for p in module.parameters():
            p.requires_grad = True
            unet_params.append(p)

    # --- 组 3: ResNet 前端 ---
    resnet_unfreeze_modules = [cnn.conv1, cnn.bn1, cnn.layer1]
    resnet_params = []
    for module in resnet_unfreeze_modules:
        for p in module.parameters():
            p.requires_grad = True
            resnet_params.append(p)
    # temperature 也解冻
    cnn.temperature.requires_grad = True
    resnet_params.append(cnn.temperature)

    param_groups = [
        {"params": bridge_params,  "lr": LR_BRIDGE, "name": "bridge"},
        {"params": unet_params,    "lr": LR_UNET,   "name": "unet_rest"},
        {"params": resnet_params,  "lr": LR_RESNET,  "name": "resnet_front"},
    ]

    # 统计
    total_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    total_all = sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in cnn.parameters())
    print(f"[参数] 可训练: {total_trainable:,} / 总共: {total_all:,} "
          f"({100.0 * total_trainable / total_all:.1f}%)")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {n:,} params, lr={g['lr']}")

    return param_groups


# ============================
# AMP
# ============================
def make_autocast():
    if USE_AMP and DEVICE_TYPE == "cuda":
        import torch.amp as ta
        return ta.autocast(device_type="cuda", enabled=True)
    from contextlib import nullcontext
    return nullcontext()


def make_scaler():
    if USE_AMP and DEVICE_TYPE == "cuda":
        import torch.amp as ta
        try:
            return ta.GradScaler(device="cuda", enabled=True)
        except TypeError:
            return ta.GradScaler(enabled=True)
    return None


# ============================
# 训练 / 验证
# ============================
def train_one_epoch(unet, cnn, radon, train_loader, optimizer, scaler, criterion):
    unet.train()
    cnn.train()

    total_loss = 0.0
    n = 0

    for img_rgb, trad_4ch, label in tqdm(train_loader, desc="joint-train", ncols=90):
        img_rgb = img_rgb.to(DEVICE, non_blocking=True)    # (B, 3, 576, 1024)
        trad_4ch = trad_4ch.to(DEVICE, non_blocking=True)  # (B, 4, 2240, 180)
        label = label.to(DEVICE, non_blocking=True)         # (B, 2)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with make_autocast():
                pred = _forward_joint(unet, cnn, radon, img_rgb, trad_4ch)
                loss = criterion(pred, label)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = _forward_joint(unet, cnn, radon, img_rgb, trad_4ch)
            loss = criterion(pred, label)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item() * img_rgb.size(0)
        n += img_rgb.size(0)

    return total_loss / max(1, n)


def _forward_joint(unet, cnn, radon, img_rgb, trad_4ch):
    """
    联合前向: UNet → (bridge_feats + film_params) → 可微 Radon + FiLM → ResNet。
    
    梯度流 (两条路径):
      [L1] ResNet ← 7ch sinogram ← [trad_4ch(detached), bridge_3ch(有梯度)]
                                                           ↑
                                                可微 Radon ← bridge_feats ← bridge_conv ← r ← UNet复原分支
      [L2] ResNet layer2 ← FiLM(γ,β) ← film_head ← c5(GAP) ← UNet encoder
    """
    B = img_rgb.size(0)

    # --- UNet 前向 (只需要复原分支, 不需要分割) ---
    restored, _, _, bridge_feats, film_params = unet(img_rgb, None,
                                        enable_restoration=True,
                                        enable_segmentation=False)
    # bridge_feats: (B, 3, 576, 1024), 値域 [0, 1], 有梯度
    # film_params:  (B, 256), 有梯度

    # --- 对 bridge_feats 的每个通道做可微 Radon ---
    bridge_sinos = []
    for ch in range(3):
        ch_feat = bridge_feats[:, ch:ch+1, :, :]  # (B, 1, H, W)
        sino = radon(ch_feat)                       # (B, 1, 2240, 180)
        bridge_sinos.append(sino)
    bridge_3ch = torch.cat(bridge_sinos, dim=1)    # (B, 3, 2240, 180), 有梯度

    # --- 拼接: 传统 4ch (无梯度) + bridge 3ch (有梯度) ---
    full_7ch = torch.cat([trad_4ch.detach(), bridge_3ch], dim=1)  # (B, 7, 2240, 180)

    # --- ResNet 前向 (bridge 3ch有梯度, film_params有梯度) ---
    pred = cnn(full_7ch, film_params=film_params)  # (B, 2)

    return pred


@torch.no_grad()
def evaluate(unet, cnn, radon, val_loader, criterion):
    unet.eval()
    cnn.eval()

    total_loss = 0.0
    n = 0

    for img_rgb, trad_4ch, label in val_loader:
        img_rgb = img_rgb.to(DEVICE, non_blocking=True)
        trad_4ch = trad_4ch.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        with make_autocast():
            pred = _forward_joint(unet, cnn, radon, img_rgb, trad_4ch)
            loss = criterion(pred, label)

        total_loss += loss.item() * img_rgb.size(0)
        n += img_rgb.size(0)

    return total_loss / max(1, n)


# ============================
# Main
# ============================
def main():
    seed_everything(SEED)
    ensure_dir(os.path.dirname(JOINT_UNET_BEST))
    ensure_dir(SPLIT_DIR)

    print("=" * 60)
    print("  联合微调: UNet (bridge) + ResNet")
    print("=" * 60)

    # --- 加载数据 ---
    df = pd.read_csv(CSV_PATH, header=None)
    splits = {
        "train": np.load(os.path.join(SPLIT_DIR, "train_indices.npy")).astype(np.int64).tolist(),
        "val":   np.load(os.path.join(SPLIT_DIR, "val_indices.npy")).astype(np.int64).tolist(),
    }

    train_ds = JointFinetuneDataset(
        os.path.join(CACHE_ROOT, "train"), splits["train"], df, IMG_DIR, UNET_IN_W, UNET_IN_H)
    val_ds = JointFinetuneDataset(
        os.path.join(CACHE_ROOT, "val"), splits["val"], df, IMG_DIR, UNET_IN_W, UNET_IN_H)

    pin = DEVICE.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)

    print(f"[数据] Train={len(train_ds)}  Val={len(val_ds)}")

    # --- 加载预训练模型 ---
    print(f"[模型] UNet: {UNET_WEIGHTS}")
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE), strict=False)

    print(f"[模型] ResNet: {CNN_WEIGHTS}")
    cnn_ckpt = torch.load(CNN_WEIGHTS, map_location=DEVICE)
    in_ch = cnn_ckpt["conv1.weight"].shape[1] if "conv1.weight" in cnn_ckpt else 7
    cnn = HorizonResNet(in_channels=in_ch).to(DEVICE)
    cnn.load_state_dict(cnn_ckpt, strict=True)

    # --- 可微 Radon ---
    radon = DifferentiableRadon(num_angles=RESIZE_W, target_h=RESIZE_H, target_w=RESIZE_W).to(DEVICE)

    # --- 参数分组 & 优化器 ---
    param_groups = build_param_groups(unet, cnn)
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, verbose=True)
    criterion = HorizonPeriodicLoss()

    # --- 训练循环 ---
    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(unet, cnn, radon, train_loader, optimizer, scaler, criterion)
        va_loss = evaluate(unet, cnn, radon, val_loader, criterion)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  lr={lr_now:.2e}  "
              f"train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "val_loss": va_loss})
        scheduler.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            torch.save(unet.state_dict(), JOINT_UNET_BEST)
            torch.save(cnn.state_dict(), JOINT_CNN_BEST)
            print(f"  -> best updated: val_loss={best_val:.6f} (epoch={best_epoch})")

    # --- 保存日志 ---
    payload = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 最佳 epoch={best_epoch}, val_loss={best_val:.6f}")
    print(f"  UNet 权重: {JOINT_UNET_BEST}")
    print(f"  CNN 权重:  {JOINT_CNN_BEST}")
    print(f"  日志:      {OUT_JSON}")


if __name__ == "__main__":
    main()
