#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stage1.py — Stage-1 训练入口（真实 ScaleLSD + entropy injection）

训练模式:
  baseline:  原始 ScaleLSD（冻结 backbone 微调 or 全量微调）
  entropy:   ScaleLSDWithEntropy（path_1 处加性注入 entropy 特征）

训练目标:
  复用 ScaleLSD 自身的 HAFM loss（md + dis + res + jloc + joff），
  不再走 radon regression 路线。

MU-SID 标注格式:
  每张图只有 1 条海天线（2 个端点），转换为：
    junctions = [[x1, y1], [x2, y2]]   (2 个 junction)
    line_map  = [[0,1],[1,0]]           (1 条边)
  传入 HAFMencoder 生成 HAFM targets。

用法:
  在 PyCharm 中直接修改顶部全局变量后运行。
"""

import os
import sys
import json
import random
from collections import defaultdict
from contextlib import nullcontext

# 将 scalelsd 仓库目录加入 sys.path，使 from scalelsd.ssl.* 可用
_SCALELSD_REPO = os.path.join(os.path.dirname(__file__), "..", "scalelsd")
if os.path.isdir(_SCALELSD_REPO) and _SCALELSD_REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCALELSD_REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1_scalelsd_entropy.data.musid_entropy_dataset import MUSIDEntropyDataset

# ============================================================
# 全局配置（在 PyCharm 中直接修改后运行）
# ============================================================

# 模式："baseline"（原始 ScaleLSD）or "entropy"（ScaleLSDWithEntropy）
MODE = "entropy"

# 数据路径
IMG_DIR       = "Hashmani's Dataset/MU-SID"
ENTROPY_DIR   = "Hashmani's Dataset/MU-SID_entropy_blue"
CSV_TRAIN     = "splits_musid/GroundTruth_train.csv"
CSV_VAL       = "splits_musid/GroundTruth_val.csv"

# 图像尺寸（resize 到这个尺寸送入模型）
IMG_H = 576
IMG_W = 1024

# 训练参数
NUM_EPOCHS       = 100
BATCH_SIZE       = 2        # DPT-ViT 显存占用大，酌情调整
NUM_WORKERS      = 4
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
GRAD_CLIP_NORM   = 1.0
USE_AMP          = True
SEED             = 42

# 学习率调度
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR   = 0.5

# 早停
EARLY_STOP_PATIENCE = 30

# loss 权重（用于加权求和 5 个子 loss）
LOSS_WEIGHTS = {
    "loss_md":   1.0,
    "loss_dis":  1.0,
    "loss_res":  0.5,
    "loss_jloc": 1.0,
    "loss_joff": 0.5,
}

# 预训练权重路径（原始 ScaleLSD checkpoint，None 表示不加载）
PRETRAINED_WEIGHTS = None

# 输出
SAVE_DIR   = "stage1_scalelsd_entropy/weights"
LOG_JSON   = "stage1_scalelsd_entropy/weights/train_log_stage1.json"

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 工具函数
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Collate: MUSIDEntropyDataset dict → ScaleLSD 训练格式
# ============================================================
def collate_scalelsd(batch, stride, img_w, img_h):
    """
    自定义 collate：从 MUSIDEntropyDataset 返回的 dict 样本中，
    组装 ScaleLSD 训练所需的 images + entropy_maps + annotations。

    annotations dict:
      batch_size : int
      stride     : int
      width      : int
      height     : int
      junctions  : List[Tensor[2, 2]]  — 每个样本的端点 (x, y)
      line_map   : List[Tensor[2, 2]]  — 上三角邻接矩阵
      valid_mask : Tensor[B, 1, H//stride, W//stride]
    """
    import torch

    images = torch.stack([s["image"] for s in batch], dim=0)
    entropy_maps = torch.stack([s["entropy_map"] for s in batch], dim=0)

    junc_list = []
    lmap_list = []
    stem_list = []
    for s in batch:
        ep = s["annotation"]["resized_endpoints"]  # np [2, 2], (x, y)
        junctions = torch.from_numpy(ep.copy())
        junctions[:, 0].clamp_(0, img_w - 1)
        junctions[:, 1].clamp_(0, img_h - 1)
        junc_list.append(junctions)

        # 对于一条无向线段，只保留上三角 adjacency，避免把同一条边记两次
        lm = torch.zeros(2, 2, dtype=torch.bool)
        lm[0, 1] = True
        lmap_list.append(lm)

        stem_list.append(s["annotation"]["stem"])

    hs = img_h // stride
    ws = img_w // stride

    annotations = {
        "batch_size": len(batch),
        "stride": stride,
        "width": img_w,
        "height": img_h,
        "junctions": junc_list,
        "line_map": lmap_list,
        "valid_mask": torch.ones(len(batch), 1, hs, ws),
    }

    return images, entropy_maps, annotations, stem_list

# ============================================================
# 模型构建
# ============================================================
def build_model(mode, pretrained_weights=None):
    """
    构建模型：baseline 用原始 ScaleLSD，entropy 用 ScaleLSDWithEntropy。
    """
    if mode == "baseline":
        from scalelsd.ssl.models.detector import ScaleLSD
        model = ScaleLSD(gray_scale=True)
    elif mode == "entropy":
        from stage1_scalelsd_entropy.models.scalelsd_with_entropy import ScaleLSDWithEntropy
        model = ScaleLSDWithEntropy(gray_scale=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if pretrained_weights and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        # 兼容不同 checkpoint 格式
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[load] missing keys  ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

    return model


# ============================================================
# 汇总 loss
# ============================================================
def aggregate_loss(loss_dict, weights):
    """将 ScaleLSD 返回的 5 个子 loss 加权求和。"""
    total = 0.0
    for key, w in weights.items():
        if key in loss_dict:
            total = total + w * loss_dict[key]
    return total


# ============================================================
# AMP
# ============================================================
def make_autocast_ctx(use_amp, device):
    if not use_amp or not device.startswith("cuda"):
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=True)


def make_scaler(use_amp, device):
    if not use_amp or not device.startswith("cuda"):
        return None
    return torch.amp.GradScaler(device="cuda", enabled=True)


# ============================================================
# Train / Evaluate
# ============================================================
def train_one_epoch(model, loader, optimizer, scaler, device, use_amp,
                    grad_clip, loss_weights, mode):
    model.train()
    total_loss = 0.0
    loss_accum = defaultdict(float)
    n = 0

    for images, entropy_maps, annotations, stems in tqdm(loader, desc="train", ncols=90):
        images = images.to(device, non_blocking=True)
        entropy_maps = entropy_maps.to(device, non_blocking=True) if mode == "entropy" else None

        # 将 annotations 中的 tensor 移到 device
        annotations["junctions"] = [j.to(device) for j in annotations["junctions"]]
        annotations["line_map"] = [m.to(device) for m in annotations["line_map"]]
        annotations["valid_mask"] = annotations["valid_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with make_autocast_ctx(use_amp, device):
            if mode == "entropy":
                loss_dict, _ = model.forward_train(images, annotations=annotations,
                                                   entropy_map=entropy_maps)
            else:
                loss_dict, _ = model.forward_train(images, annotations=annotations)

            loss = aggregate_loss(loss_dict, loss_weights)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        for k, v in loss_dict.items():
            loss_accum[k] += float(v) * bs if torch.is_tensor(v) else float(v) * bs
        n += bs

    avg_total = total_loss / max(1, n)
    avg_sub = {k: v / max(1, n) for k, v in loss_accum.items()}
    return avg_total, avg_sub


@torch.no_grad()
def evaluate(model, loader, device, use_amp, loss_weights, mode):
    model.eval()
    total_loss = 0.0
    loss_accum = defaultdict(float)
    n = 0

    for images, entropy_maps, annotations, stems in loader:
        images = images.to(device, non_blocking=True)
        entropy_maps = entropy_maps.to(device, non_blocking=True) if mode == "entropy" else None

        annotations["junctions"] = [j.to(device) for j in annotations["junctions"]]
        annotations["line_map"] = [m.to(device) for m in annotations["line_map"]]
        annotations["valid_mask"] = annotations["valid_mask"].to(device)

        with make_autocast_ctx(use_amp, device):
            if mode == "entropy":
                loss_dict, _ = model.forward_train(images, annotations=annotations,
                                                   entropy_map=entropy_maps)
            else:
                loss_dict, _ = model.forward_train(images, annotations=annotations)

            loss = aggregate_loss(loss_dict, loss_weights)

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        for k, v in loss_dict.items():
            loss_accum[k] += float(v) * bs if torch.is_tensor(v) else float(v) * bs
        n += bs

    avg_total = total_loss / max(1, n)
    avg_sub = {k: v / max(1, n) for k, v in loss_accum.items()}
    return avg_total, avg_sub


# ============================================================
# Main
# ============================================================
def main():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = DEVICE
    mode = MODE

    print("=" * 60)
    print(f"Stage-1 Training — mode={mode}")
    print(f"Device: {device}")
    print(f"Image size: {IMG_H}x{IMG_W}")
    print("=" * 60)

    # ---- Model ----
    model = build_model(mode, pretrained_weights=PRETRAINED_WEIGHTS)
    model.to(device)
    stride = model.stride
    print(f"Backbone stride: {stride}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} total, {n_train:,} trainable")

    if mode == "entropy":
        alpha_val = model.backbone.alpha.item()
        ent_params = sum(p.numel() for n, p in model.named_parameters()
                         if "entropy_branch" in n or n.endswith("alpha"))
        print(f"Entropy params: {ent_params:,}, alpha init: {alpha_val:.4f}")

    # ---- Dataset ----
    common_kwargs = dict(
        img_dir=IMG_DIR, entropy_dir=ENTROPY_DIR,
        img_size=(IMG_H, IMG_W), gray_scale=True,
    )
    train_ds = MUSIDEntropyDataset(csv_file=CSV_TRAIN, **common_kwargs)
    val_ds = MUSIDEntropyDataset(csv_file=CSV_VAL, **common_kwargs)

    collate_fn = lambda batch: collate_scalelsd(batch, stride=stride,
                                                img_w=IMG_W, img_h=IMG_H)
    pin = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            collate_fn=collate_fn)

    print(f"Train={len(train_ds)}, Val={len(val_ds)}")

    # ---- Optimizer / Scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler(USE_AMP, device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, verbose=True,
    )

    # ---- Training loop ----
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = os.path.join(SAVE_DIR, f"best_stage1_{mode}.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_sub = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            USE_AMP, GRAD_CLIP_NORM, LOSS_WEIGHTS, mode,
        )
        va_loss, va_sub = evaluate(
            model, val_loader, device, USE_AMP, LOSS_WEIGHTS, mode,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{mode}] Epoch [{epoch:03d}/{NUM_EPOCHS}]  lr={lr_now:.2e}  "
              f"train={tr_loss:.6f}  val={va_loss:.6f}")
        # 打印子 loss
        sub_str = "  ".join(f"{k}={v:.4f}" for k, v in va_sub.items())
        print(f"  val sub: {sub_str}")

        if mode == "entropy":
            print(f"  alpha={model.backbone.alpha.item():.6f}")

        record = {
            "epoch": epoch, "mode": mode, "lr": lr_now,
            "train_loss": tr_loss, "val_loss": va_loss,
            "train_sub": {k: float(v) for k, v in tr_sub.items()},
            "val_sub": {k: float(v) for k, v in va_sub.items()},
        }
        history.append(record)

        scheduler.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> best updated: {best_val:.6f} (epoch={best_epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[EARLY STOP] no improvement for {EARLY_STOP_PATIENCE} epochs. "
                      f"best_epoch={best_epoch}")
                break

    # ---- Save log ----
    payload = {
        "mode": mode,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "config": {
            "img_h": IMG_H, "img_w": IMG_W, "batch_size": BATCH_SIZE,
            "lr": LR, "num_epochs": NUM_EPOCHS, "loss_weights": LOSS_WEIGHTS,
        },
        "history": history,
    }
    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Done] best_epoch={best_epoch}  best_val={best_val:.6f}")
    print(f"  Model saved: {best_path}")
    print(f"  Log saved:   {LOG_JSON}")


if __name__ == "__main__":
    main()
