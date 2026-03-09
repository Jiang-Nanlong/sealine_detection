#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stage1.py — Stage-1 训练入口

训练模式:
  1. baseline:  纯图像 ResNet 回归 (不用 entropy)
  2. entropy:   图像 + 局部熵特征注入

用法:
  # 默认使用 config 文件
  python -m stage1_scalelsd_entropy.train_stage1 \
      --config stage1_scalelsd_entropy/configs/musid_entropy_stage1.yaml

  # 命令行覆盖模式
  python -m stage1_scalelsd_entropy.train_stage1 \
      --config stage1_scalelsd_entropy/configs/musid_entropy_stage1.yaml \
      --mode baseline

验证成功标志:
  - 训练 loss 逐步下降
  - val loss 在数十 epoch 后趋于稳定
  - weights/ 下生成 best_stage1.pth
  - log json 文件记录每个 epoch 的 train/val loss
"""

import os
import sys
import json
import random
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保可以 import 项目根目录模块
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stage1_scalelsd_entropy.data.musid_dataset import MUSIDEntropyDataset
from stage1_scalelsd_entropy.models.scalelsd_entropy_wrapper import ScaleLSDEntropyWrapper


# ==================================================================
# Config loading
# ==================================================================
def load_yaml_config(path: str) -> dict:
    """简易 YAML 加载（避免强依赖 PyYAML 时回退到手动解析）。"""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # 如果没有 PyYAML，使用 json 方式或返回默认值
        raise ImportError(
            "PyYAML is required. Install with: pip install pyyaml"
        )


def get_config(args) -> dict:
    """合并 yaml 配置和命令行参数。"""
    cfg = load_yaml_config(args.config)

    # 命令行覆盖
    if args.mode is not None:
        cfg["model"]["mode"] = args.mode
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs
    if args.entropy_dir is not None:
        cfg["data"]["entropy_dir"] = args.entropy_dir
    if args.img_dir is not None:
        cfg["data"]["img_dir"] = args.img_dir

    return cfg


# ==================================================================
# Loss (复用自 train_fusion_cnn.py)
# ==================================================================
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


# ==================================================================
# AMP helpers
# ==================================================================
def make_autocast_ctx(use_amp: bool, device: str):
    if not use_amp or not device.startswith("cuda"):
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=True)


def make_scaler(use_amp: bool, device: str):
    if not use_amp or not device.startswith("cuda"):
        return None
    return torch.amp.GradScaler(device="cuda", enabled=True)


# ==================================================================
# Reproducibility
# ==================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================================
# Train / Eval
# ==================================================================
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp, grad_clip, mode):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", ncols=90):
        img, ent, label, _ = batch
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        ent = ent.to(device, non_blocking=True) if mode == "entropy" else None

        optimizer.zero_grad(set_to_none=True)

        with make_autocast_ctx(use_amp, device):
            pred = model(img, entropy_map=ent)
            loss = criterion(pred, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * img.size(0)
        n += img.size(0)

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, mode):
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        img, ent, label, _ = batch
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        ent = ent.to(device, non_blocking=True) if mode == "entropy" else None

        with make_autocast_ctx(use_amp, device):
            pred = model(img, entropy_map=ent)
            loss = criterion(pred, label)

        total_loss += float(loss.item()) * img.size(0)
        n += img.size(0)

    return total_loss / max(1, n)


# ==================================================================
# Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage-1 Train: image + entropy horizon regression")
    parser.add_argument("--config", type=str,
                        default="stage1_scalelsd_entropy/configs/musid_entropy_stage1.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["baseline", "entropy"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--entropy_dir", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args)

    # ---- Unpack config ----
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]

    mode = model_cfg["mode"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Stage-1 Training — mode={mode}")
    print(f"Device: {device}")
    print("=" * 60)

    seed_everything(train_cfg["seed"])
    os.makedirs(out_cfg["weights_dir"], exist_ok=True)

    # ---- Datasets ----
    common_ds_kwargs = dict(
        img_dir=data_cfg["img_dir"],
        entropy_dir=data_cfg["entropy_dir"],
        img_h=data_cfg["img_h"],
        img_w=data_cfg["img_w"],
        orig_h=data_cfg["orig_h"],
        orig_w=data_cfg["orig_w"],
        sinogram_h=data_cfg["sinogram_h"],
        label_mode="radon",
    )

    train_ds = MUSIDEntropyDataset(csv_path=data_cfg["csv_train"], augment=True, **common_ds_kwargs)
    val_ds = MUSIDEntropyDataset(csv_path=data_cfg["csv_val"], augment=False, **common_ds_kwargs)

    test_ds = None
    if os.path.isfile(data_cfg["csv_test"]):
        test_ds = MUSIDEntropyDataset(csv_path=data_cfg["csv_test"], augment=False, **common_ds_kwargs)

    pin = device.startswith("cuda")
    bs = train_cfg["batch_size"]
    nw = train_cfg["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin) if test_ds else None

    print(f"Train={len(train_ds)}, Val={len(val_ds)}, "
          f"Test={len(test_ds) if test_ds else 0}")

    # ---- Model ----
    model = ScaleLSDEntropyWrapper(
        mode=mode,
        entropy_branch_ch=model_cfg["entropy_branch_ch"],
        backbone_l2_ch=model_cfg["backbone_l2_ch"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} total, {n_train_params:,} trainable")

    # ---- Loss / Optimizer / Scheduler ----
    criterion = HorizonPeriodicLoss(
        rho_weight=train_cfg["rho_weight"],
        theta_weight=train_cfg["theta_weight"],
        rho_beta=train_cfg["rho_beta"],
        theta_beta=train_cfg["theta_beta"],
    )
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    scaler = make_scaler(train_cfg["use_amp"], device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=train_cfg["scheduler_factor"],
        patience=train_cfg["scheduler_patience"],
        verbose=True,
    )

    # ---- Training loop ----
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    num_epochs = train_cfg["num_epochs"]
    early_patience = train_cfg["early_stop_patience"]
    grad_clip = train_cfg["grad_clip_norm"]
    use_amp = train_cfg["use_amp"]

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                  criterion, device, use_amp, grad_clip, mode)
        va_loss = evaluate(model, val_loader, criterion, device, use_amp, mode)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{mode}] Epoch [{epoch:03d}/{num_epochs}]  lr={lr_now:.2e}  "
              f"train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

        history.append({
            "epoch": epoch,
            "mode": mode,
            "lr": lr_now,
            "train_loss": tr_loss,
            "val_loss": va_loss,
        })

        scheduler.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), out_cfg["best_model"])
            print(f"  -> best updated: {best_val:.6f} (epoch={best_epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= early_patience:
                print(f"[EARLY STOP] no improvement for {early_patience} epochs. "
                      f"best_epoch={best_epoch}")
                break

    # ---- Final test ----
    final_test = None
    if test_loader is not None and os.path.isfile(out_cfg["best_model"]):
        model.load_state_dict(torch.load(out_cfg["best_model"], map_location=device, weights_only=True))
        final_test = evaluate(model, test_loader, criterion, device, use_amp, mode)
        print(f"[FINAL TEST] loss={final_test:.6f}")

    # ---- Save log ----
    payload = {
        "mode": mode,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_loss": final_test,
        "config": cfg,
        "history": history,
    }
    with open(out_cfg["log_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Done] Log saved to {out_cfg['log_json']}")


if __name__ == "__main__":
    main()
