# -*- coding: utf-8 -*-
"""
train_unet.py (Hardcore + B2)
- 支持 Stage: A / B / B2 / C1 / C2
- 固定 train/val 划分（可复现）
- 每个 epoch 末自动验证并打印指标
- 自动保存：
    1) last: rghnet_stage_{stage}.pth
    2) best_seg: rghnet_best_seg.pth   (按 IoU 优先，其次 Horizon MAE)
    3) best_joint: rghnet_best_joint.pth (按验证 joint_loss 最小)

Stage 语义：
- A : 只训 Restoration（encoder+rest），用 clear 图
- B : 只训 Seg（分割头），从 stage_a 初始化
- C1: 联合数据 joint，但只训 Restoration（encoder+rest），分割头冻结（用于让复原适应脏图）
- B2: “Seg Repair”——只训 Seg（分割头），从 C1（或 best_seg）初始化，让分割重新对齐 encoder 特征
- C2: 全量联合微调（encoder+rest+seg）

可选保存可视化到 ./val_vis（只在 best_seg 刷新时输出少量样本）
"""
import os
import csv
import math
import platform
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# AMP (new API)
import torch.amp as amp

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import SimpleFolderDataset, HorizonImageDataset


# =========================
# Config
# =========================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
IMG_CLEAR_DIR = r"Hashmani's Dataset/clear"
DCE_WEIGHTS = "Epoch99.pth"

# 'A' / 'B' / 'C1' / 'B2' / 'C2'
STAGE = "B2"

IMG_SIZE = 384
BATCH_SIZE = 16

# 固定划分 & 训练超参
SEED = 42
VAL_RATIO = 0.2
PRINT_EVERY = 1
EVAL_EVERY = 1

SAVE_VIS = True
VIS_MAX = 8  # 每次最多保存多少张可视化（只在 best_seg 刷新时保存）

# Stage 超参（你也可以直接改这里）
STAGE_CFG = {
    "A":  dict(lr=2e-4, epochs=50),
    "B":  dict(lr=1e-4, epochs=20),
    "C1": dict(lr=5e-5, epochs=10),
    "B2": dict(lr=5e-5, epochs=5),   # ✅ Seg Repair：一般 3~5 epoch 就够
    "C2": dict(lr=2e-5, epochs=40),
}

# joint loss 里 segmentation 的权重
JOINT_SEG_W = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils
# =========================
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_load(path: str, map_location: str):
    """torch.load without FutureWarning (weights_only=True), with backward-compat fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def get_modules(model, names):
    return [getattr(model, n, None) for n in names]


def split_dataset(ds, val_ratio: float, seed: int):
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Losses
# =========================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=torch.float32)
        kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)  # [3,1,5,5]
        self.register_buffer("kernel", kernel)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(device=img.device, dtype=img.dtype)
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate")
        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current: torch.Tensor) -> torch.Tensor:
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))


class HybridRestorationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        return self.charb(pred, target) + 0.1 * self.edge(pred, target)


# =========================
# Optimizer builder (模块名双兼容)
# =========================
def build_optimizer(model, stage: str, lr: float):
    restoration_names = [
        # FPN style
        "rest_lat2", "rest_lat3", "rest_lat4", "rest_lat5",
        "rest_fuse", "rest_strip", "rest_out",
        # legacy UNet style (if exists)
        "rest_up1", "rest_conv1", "rest_up2", "rest_conv2",
        "rest_up3", "rest_conv3", "rest_up4",
    ]
    segmentation_names = [
        "seg_lat3", "seg_lat4", "seg_lat5",
        "seg_fuse", "seg_strip", "seg_head", "seg_final",
        "inject",
        # legacy names (if exists)
        "strip_pool", "seg_conv_fuse", "injection_conv",
    ]

    restoration_modules = get_modules(model, restoration_names)
    segmentation_modules = get_modules(model, segmentation_names)

    # 0) freeze all
    set_requires_grad(model.encoder, False)
    for m in restoration_modules:
        set_requires_grad(m, False)
    for m in segmentation_modules:
        set_requires_grad(m, False)

    # DCE always frozen
    if hasattr(model, "dce_net"):
        set_requires_grad(getattr(model, "dce_net", None), False)

    def collect_params(modules):
        params = []
        for m in modules:
            if m is not None:
                params += list(m.parameters())
        return params

    encoder_params = list(model.encoder.parameters())
    rest_params = collect_params(restoration_modules)
    seg_params = collect_params(segmentation_modules)

    if stage == "A":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules:
            set_requires_grad(m, True)
        rest_params = collect_params(restoration_modules)
        return optim.AdamW(
            [{"params": encoder_params, "lr": lr * 0.1},
             {"params": rest_params, "lr": lr}],
            weight_decay=1e-4
        )

    if stage in ("B", "B2"):
        # only seg
        for m in segmentation_modules:
            set_requires_grad(m, True)
        seg_params = collect_params(segmentation_modules)
        if len(seg_params) == 0:
            raise RuntimeError(f"[build_optimizer] Stage {stage}: no segmentation params found. Check module names.")
        return optim.AdamW([{"params": seg_params, "lr": lr}], weight_decay=1e-4)

    if stage == "C1":
        # encoder + rest, seg frozen
        set_requires_grad(model.encoder, True)
        for m in restoration_modules:
            set_requires_grad(m, True)
        rest_params = collect_params(restoration_modules)
        return optim.AdamW(
            [{"params": encoder_params, "lr": lr * 0.1},
             {"params": rest_params, "lr": lr}],
            weight_decay=1e-4
        )

    if stage == "C2":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules:
            set_requires_grad(m, True)
        for m in segmentation_modules:
            set_requires_grad(m, True)
        rest_params = collect_params(restoration_modules)
        seg_params = collect_params(segmentation_modules)
        return optim.AdamW(
            [{"params": encoder_params, "lr": lr * 0.1},
             {"params": rest_params, "lr": lr},
             {"params": seg_params, "lr": lr}],
            weight_decay=1e-4
        )

    raise ValueError(f"Unknown stage: {stage}")


# =========================
# Metrics
# =========================
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def seg_metrics_from_masks(pred: torch.Tensor, gt: torch.Tensor, ignore_index: int = 255) -> Dict[str, float]:
    pred = pred.view(-1)
    gt = gt.view(-1)
    valid = gt != ignore_index
    pred = pred[valid]
    gt = gt[valid]
    if pred.numel() == 0:
        return dict(pixel_acc=0, iou_sky=0, dice=0, precision=0, recall=0)

    correct = (pred == gt).sum().item()
    total = pred.numel()
    pixel_acc = _safe_div(correct, total)

    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()

    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    return dict(pixel_acc=pixel_acc, iou_sky=iou, dice=dice, precision=precision, recall=recall)


def horizon_mae_np(pred_mask: np.ndarray, gt_mask: np.ndarray, ignore: int = 255) -> float:
    H, W = gt_mask.shape
    abs_err = []
    ys = np.arange(H)

    for x in range(W):
        gt_col = gt_mask[:, x]
        valid = gt_col != ignore
        if not np.any(valid):
            continue

        y_valid = ys[valid]
        gt_valid = gt_col[valid]

        sky_y = y_valid[gt_valid == 1]
        sea_y = y_valid[gt_valid == 0]

        if sky_y.size > 0 and sea_y.size > 0:
            y_sky = int(sky_y.max())
            y_sea = int(sea_y.min())
            y_gt = (y_sky + y_sea) / 2.0 if y_sea > y_sky else float(y_sky)
        elif sky_y.size > 0:
            y_gt = float(sky_y.max())
        elif sea_y.size > 0:
            y_gt = float(sea_y.min())
        else:
            continue

        pred_col = pred_mask[:, x]
        sky_p = np.where(pred_col == 1)[0]
        sea_p = np.where(pred_col == 0)[0]
        if sky_p.size > 0 and sea_p.size > 0:
            y_sky_p = int(sky_p.max())
            y_sea_p = int(sea_p.min())
            y_pr = (y_sky_p + y_sea_p) / 2.0 if y_sea_p > y_sky_p else float(y_sky_p)
        elif sky_p.size > 0:
            y_pr = float(sky_p.max())
        elif sea_p.size > 0:
            y_pr = float(sea_p.min())
        else:
            continue

        abs_err.append(abs(y_pr - y_gt))

    return float(np.mean(abs_err)) if len(abs_err) > 0 else 0.0


def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> float:
    mse_val = float(mse.item())
    if mse_val <= 0:
        return 99.0
    return 10.0 * math.log10((max_val * max_val) / mse_val)


# =========================
# Visualization (optional)
# =========================
def save_vis(out_dir: str, prefix: str, img_tensor: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor):
    ensure_dir(out_dir)
    img = (img_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    gt = gt_mask.detach().cpu().numpy().astype(np.uint8)
    pr = pred_mask.detach().cpu().numpy().astype(np.uint8)
    H, W, _ = img.shape

    gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
    gt_vis[gt == 1] = (255, 255, 255)
    gt_vis[gt == 0] = (0, 0, 0)
    gt_vis[gt == 255] = (127, 127, 127)

    pr_vis = np.zeros((H, W, 3), dtype=np.uint8)
    pr_vis[pr == 1] = (255, 255, 255)

    overlay = img.copy()

    gt_points, pr_points = [], []
    for x in range(W):
        gt_col = gt[:, x]
        valid = gt_col != 255
        if np.any(valid):
            ys = np.where(valid)[0]
            gv = gt_col[valid]
            sky = ys[gv == 1]
            sea = ys[gv == 0]
            if sky.size > 0 and sea.size > 0:
                y_sky = int(sky.max()); y_sea = int(sea.min())
                y = int((y_sky + y_sea) / 2) if y_sea > y_sky else y_sky
            elif sky.size > 0:
                y = int(sky.max())
            elif sea.size > 0:
                y = int(sea.min())
            else:
                y = None
            if y is not None:
                gt_points.append((x, y))

        pr_col = pr[:, x]
        sky_p = np.where(pr_col == 1)[0]
        sea_p = np.where(pr_col == 0)[0]
        if sky_p.size > 0 and sea_p.size > 0:
            y_sky = int(sky_p.max()); y_sea = int(sea_p.min())
            y = int((y_sky + y_sea) / 2) if y_sea > y_sky else y_sky
        elif sky_p.size > 0:
            y = int(sky_p.max())
        elif sea_p.size > 0:
            y = int(sea_p.min())
        else:
            y = None
        if y is not None:
            pr_points.append((x, y))

    try:
        import cv2
        # ✅ 绿：GT，蓝：Pred（OpenCV 默认 BGR）
        if len(gt_points) > 1:
            cv2.polylines(overlay, [np.array(gt_points, dtype=np.int32)], False, (0, 255, 0), 2)
        if len(pr_points) > 1:
            cv2.polylines(overlay, [np.array(pr_points, dtype=np.int32)], False, (255, 0, 0), 2)
    except Exception:
        pass

    top = np.concatenate([img, gt_vis], axis=1)
    bot = np.concatenate([overlay, pr_vis], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    try:
        import cv2
        cv2.imwrite(os.path.join(out_dir, f"{prefix}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image
        Image.fromarray(grid).save(os.path.join(out_dir, f"{prefix}.png"))


# =========================
# Eval
# =========================
@dataclass
class EvalResult:
    rest_loss: float = 0.0
    seg_loss: float = 0.0
    joint_loss: float = 0.0
    psnr: float = 0.0
    pixel_acc: float = 0.0
    iou_sky: float = 0.0
    dice: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    horizon_mae: float = 0.0


@torch.no_grad()
def evaluate(model, loader, mode: str, crit_rest, crit_seg) -> EvalResult:
    model.eval()
    n = 0
    rest_loss_sum = 0.0
    seg_loss_sum = 0.0
    joint_loss_sum = 0.0
    psnr_sum = 0.0

    pixel_acc_sum = 0.0
    iou_sum = 0.0
    dice_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    horizon_sum = 0.0

    for batch in loader:
        if mode == "seg":
            img, mask = batch
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                _, seg, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
                loss_s = crit_seg(seg, mask)

            pred = seg.argmax(1)
            m = seg_metrics_from_masks(pred.detach(), mask.detach())
            pixel_acc_sum += m["pixel_acc"]
            iou_sum += m["iou_sky"]
            dice_sum += m["dice"]
            prec_sum += m["precision"]
            rec_sum += m["recall"]

            pred_np = pred.detach().cpu().numpy()
            gt_np = mask.detach().cpu().numpy()
            bsz = pred_np.shape[0]
            h_mae = 0.0
            for i in range(bsz):
                h_mae += horizon_mae_np(pred_np[i], gt_np[i])
            horizon_sum += (h_mae / max(1, bsz))

            seg_loss_sum += float(loss_s.item())
            joint_loss_sum += float(loss_s.item())
            n += 1

        elif mode == "joint":
            img, target, mask = batch
            img = img.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                restored, seg, target_dce = model(img, target, enable_restoration=True, enable_segmentation=True)
                loss_r = crit_rest(restored, target_dce)
                loss_s = crit_seg(seg, mask)
                loss_joint = loss_r + JOINT_SEG_W * loss_s

            mse = F.mse_loss(restored.detach().float(), target.detach().float())
            psnr_sum += psnr_from_mse(mse)

            pred = seg.argmax(1)
            m = seg_metrics_from_masks(pred.detach(), mask.detach())
            pixel_acc_sum += m["pixel_acc"]
            iou_sum += m["iou_sky"]
            dice_sum += m["dice"]
            prec_sum += m["precision"]
            rec_sum += m["recall"]

            pred_np = pred.detach().cpu().numpy()
            gt_np = mask.detach().cpu().numpy()
            bsz = pred_np.shape[0]
            h_mae = 0.0
            for i in range(bsz):
                h_mae += horizon_mae_np(pred_np[i], gt_np[i])
            horizon_sum += (h_mae / max(1, bsz))

            rest_loss_sum += float(loss_r.item())
            seg_loss_sum += float(loss_s.item())
            joint_loss_sum += float(loss_joint.item())
            n += 1

        elif mode == "rest":
            img, target = batch
            img = img.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                restored, _, target_dce = model(img, target, enable_restoration=True, enable_segmentation=False)
                loss_r = crit_rest(restored, target_dce)

            mse = F.mse_loss(restored.detach().float(), target.detach().float())
            psnr_sum += psnr_from_mse(mse)

            rest_loss_sum += float(loss_r.item())
            joint_loss_sum += float(loss_r.item())
            n += 1
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

    if n == 0:
        return EvalResult()

    res = EvalResult()
    res.rest_loss = rest_loss_sum / n
    res.seg_loss = seg_loss_sum / n
    res.joint_loss = joint_loss_sum / n
    res.psnr = psnr_sum / n

    res.pixel_acc = pixel_acc_sum / n
    res.iou_sky = iou_sum / n
    res.dice = dice_sum / n
    res.precision = prec_sum / n
    res.recall = rec_sum / n
    res.horizon_mae = horizon_sum / n
    return res


# =========================
# Logging
# =========================
def append_log(csv_path: str, row: Dict):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


# =========================
# Main
# =========================
def main():
    if STAGE not in STAGE_CFG:
        raise ValueError(f"STAGE must be one of {list(STAGE_CFG.keys())}, got {STAGE}")

    seed_everything(SEED)

    lr = STAGE_CFG[STAGE]["lr"]
    epochs = STAGE_CFG[STAGE]["epochs"]

    print(f"=== Start Training: Stage {STAGE} ===")
    print(f"DEVICE={DEVICE}, DEVICE_TYPE={DEVICE_TYPE}")
    print(f"LR={lr}, EPOCHS={epochs}, IMG_SIZE={IMG_SIZE}, BS={BATCH_SIZE}")
    print(f"VAL_RATIO={VAL_RATIO}, SEED={SEED}")

    # 1) Dataset
    if STAGE == "A":
        ds = SimpleFolderDataset(IMG_CLEAR_DIR, img_size=IMG_SIZE)
        eval_mode = "rest"
    elif STAGE in ("B", "B2"):
        ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="segmentation")
        eval_mode = "seg"
    else:
        ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="joint")
        eval_mode = "joint"

    train_ds, val_ds = split_dataset(ds, VAL_RATIO, SEED)
    print(f"[Split] train={len(train_ds)}, val={len(val_ds)}, total={len(ds)}")

    # 2) DataLoader
    num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    # 3) Model
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)

    # 4) Relay checkpoints
    ckpt_a = "rghnet_stage_a.pth"
    ckpt_b = "rghnet_stage_b.pth"
    ckpt_c1 = "rghnet_stage_c1.pth"
    ckpt_b2 = "rghnet_stage_b2.pth"
    ckpt_best_seg = "rghnet_best_seg.pth"

    if STAGE == "B":
        if not os.path.exists(ckpt_a):
            raise FileNotFoundError(f"Stage B requires {ckpt_a}. Run Stage A first.")
        print(f"[Init] Loading Stage A weights: {ckpt_a}")
        model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)

    if STAGE == "C1":
        if os.path.exists(ckpt_b):
            print(f"[Init] Loading Stage B weights: {ckpt_b}")
            model.load_state_dict(safe_load(ckpt_b, DEVICE), strict=False)
        elif os.path.exists(ckpt_a):
            print(f"[Init] Stage B not found, loading Stage A weights: {ckpt_a}")
            model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)
        else:
            raise FileNotFoundError(f"Stage C1 requires {ckpt_b} (preferred) or {ckpt_a}.")

    if STAGE == "B2":
        # ✅ Seg Repair：优先从 best_seg 或 stage_c1 启动
        if os.path.exists(ckpt_best_seg):
            print(f"[Init] Loading best_seg weights: {ckpt_best_seg}")
            model.load_state_dict(safe_load(ckpt_best_seg, DEVICE), strict=False)
        elif os.path.exists(ckpt_c1):
            print(f"[Init] Loading Stage C1 weights: {ckpt_c1}")
            model.load_state_dict(safe_load(ckpt_c1, DEVICE), strict=False)
        elif os.path.exists(ckpt_b):
            print(f"[Init] Stage C1 not found, loading Stage B weights: {ckpt_b}")
            model.load_state_dict(safe_load(ckpt_b, DEVICE), strict=False)
        else:
            raise FileNotFoundError(f"Stage B2 requires {ckpt_best_seg}/{ckpt_c1}/{ckpt_b} (any).")

    if STAGE == "C2":
        # ✅ C2：如果做过 B2，就优先从 stage_b2 启动
        if os.path.exists(ckpt_b2):
            print(f"[Init] Loading Stage B2 weights: {ckpt_b2}")
            model.load_state_dict(safe_load(ckpt_b2, DEVICE), strict=False)
        elif os.path.exists(ckpt_c1):
            print(f"[Init] Loading Stage C1 weights: {ckpt_c1}")
            model.load_state_dict(safe_load(ckpt_c1, DEVICE), strict=False)
        elif os.path.exists(ckpt_b):
            print(f"[Init] Stage C1 not found, loading Stage B weights: {ckpt_b}")
            model.load_state_dict(safe_load(ckpt_b, DEVICE), strict=False)
        elif os.path.exists(ckpt_a):
            print(f"[Init] Stage C1/B not found, loading Stage A weights: {ckpt_a}")
            model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)
        else:
            raise FileNotFoundError(f"Stage C2 requires {ckpt_b2}/{ckpt_c1}/{ckpt_b}/{ckpt_a} (any).")

    # 5) Loss / Optimizer / AMP
    crit_rest = HybridRestorationLoss().to(DEVICE)
    crit_seg = nn.CrossEntropyLoss(ignore_index=255).to(DEVICE)
    optimizer = build_optimizer(model, STAGE, lr)

    try:
        scaler = amp.GradScaler(device=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda"))
    except TypeError:
        scaler = amp.GradScaler(enabled=(DEVICE_TYPE == "cuda"))

    # 6) best trackers
    best_iou = -1.0
    best_mae = 1e9
    best_joint = 1e9

    log_path = f"train_log_stage_{STAGE.lower()}.csv"
    ensure_dir("val_vis")

    # 7) Train
    for epoch in range(1, epochs + 1):
        model.train()

        # ✅ Stage B / B2：encoder/restoration frozen 但 forward 会跑 -> 冻结 BN 统计
        if STAGE in ("B", "B2"):
            if hasattr(model, "encoder"):
                model.encoder.eval()
            for n in ["rest_lat2","rest_lat3","rest_lat4","rest_lat5","rest_fuse","rest_strip","rest_out",
                      "rest_up1","rest_conv1","rest_up2","rest_conv2","rest_up3","rest_conv3","rest_up4"]:
                m = getattr(model, n, None)
                if m is not None:
                    m.eval()

        loop = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}")
        loss_sum = 0.0

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            if STAGE == "A":
                img, target = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    restored, _, target_dce = model(img, target, enable_restoration=True, enable_segmentation=False)
                    loss = crit_rest(restored, target_dce)

            elif STAGE in ("B", "B2"):
                img, mask = batch
                img = img.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    _, seg, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
                    loss = crit_seg(seg, mask)

            elif STAGE == "C1":
                img, target, _mask = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    restored, _, target_dce = model(img, target, enable_restoration=True, enable_segmentation=False)
                    loss = crit_rest(restored, target_dce)

            else:  # C2
                img, target, mask = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    restored, seg, target_dce = model(img, target, enable_restoration=True, enable_segmentation=True)
                    loss_r = crit_rest(restored, target_dce)
                    loss_s = crit_seg(seg, mask)
                    loss = loss_r + JOINT_SEG_W * loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        train_loss = loss_sum / max(1, len(train_loader))
        if epoch % PRINT_EVERY == 0:
            print(f"[Train] epoch={epoch}  loss={train_loss:.6f}")

        # save last each epoch
        last_name = f"rghnet_stage_{STAGE.lower()}.pth"
        torch.save(model.state_dict(), last_name)

        # eval
        if (epoch % EVAL_EVERY) == 0:
            val_res = evaluate(model, val_loader, eval_mode, crit_rest, crit_seg)
            print(
                f"[Val] epoch={epoch}  joint={val_res.joint_loss:.6f}  rest={val_res.rest_loss:.6f}  "
                f"seg={val_res.seg_loss:.6f}  psnr={val_res.psnr:.2f}  "
                f"IoU={val_res.iou_sky:.4f}  MAE={val_res.horizon_mae:.2f}  Acc={val_res.pixel_acc:.4f}"
            )

            # best_joint
            if val_res.joint_loss < best_joint:
                best_joint = val_res.joint_loss
                torch.save(model.state_dict(), "rghnet_best_joint.pth")
                print(f"[Save] best_joint updated: {best_joint:.6f} -> rghnet_best_joint.pth")

            # best_seg（B/B2/joint 都能算）
            has_seg = (eval_mode in ["seg", "joint"])
            if has_seg:
                improve = (val_res.iou_sky > best_iou + 1e-6) or (
                    abs(val_res.iou_sky - best_iou) <= 1e-6 and val_res.horizon_mae < best_mae
                )
                if improve:
                    best_iou = val_res.iou_sky
                    best_mae = val_res.horizon_mae
                    torch.save(model.state_dict(), "rghnet_best_seg.pth")
                    print(f"[Save] best_seg updated: IoU={best_iou:.4f}, MAE={best_mae:.2f} -> rghnet_best_seg.pth")

                    if SAVE_VIS:
                        saved = 0
                        for vb in val_loader:
                            if eval_mode == "seg":
                                v_img, v_mask = vb
                                v_img = v_img.to(DEVICE)
                                v_mask = v_mask.to(DEVICE)
                                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                                    _, v_seg, _ = model(v_img, None, enable_restoration=True, enable_segmentation=True)
                                v_pred = v_seg.argmax(1)
                                for i in range(v_img.shape[0]):
                                    save_vis("val_vis", f"{STAGE}_ep{epoch}_idx{saved}",
                                             v_img[i], v_mask[i], v_pred[i])
                                    saved += 1
                                    if saved >= VIS_MAX:
                                        break
                            else:
                                v_img, v_target, v_mask = vb
                                v_img = v_img.to(DEVICE)
                                v_target = v_target.to(DEVICE)
                                v_mask = v_mask.to(DEVICE)
                                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                                    _, v_seg, _ = model(v_img, v_target, enable_restoration=True, enable_segmentation=True)
                                v_pred = v_seg.argmax(1)
                                for i in range(v_img.shape[0]):
                                    save_vis("val_vis", f"{STAGE}_ep{epoch}_idx{saved}",
                                             v_img[i], v_mask[i], v_pred[i])
                                    saved += 1
                                    if saved >= VIS_MAX:
                                        break
                            if saved >= VIS_MAX:
                                break
                        print(f"[Vis] saved {saved} images to ./val_vis")

            append_log(
                log_path,
                dict(
                    stage=STAGE,
                    epoch=epoch,
                    lr=lr,
                    train_loss=train_loss,
                    val_joint=val_res.joint_loss,
                    val_rest=val_res.rest_loss,
                    val_seg=val_res.seg_loss,
                    val_psnr=val_res.psnr,
                    val_iou=val_res.iou_sky,
                    val_mae=val_res.horizon_mae,
                    val_acc=val_res.pixel_acc,
                    val_dice=val_res.dice,
                    val_precision=val_res.precision,
                    val_recall=val_res.recall,
                ),
            )

    print(f"Stage {STAGE} done.")
    print(f"  last      : {last_name}")
    print(f"  best_seg  : rghnet_best_seg.pth")
    print(f"  best_joint: rghnet_best_joint.pth")


if __name__ == "__main__":
    main()
