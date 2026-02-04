# -*- coding: utf-8 -*-
"""
train_unet.py (Final Perfect Version)
- 核心配置: 1024x576 (无黑边), P_CLEAN=0.35, Target=Clean (无色偏)
- 权重管理: rghnet_best_{stage}.pth (互不覆盖，自动寻找上一阶段)
- 训练逻辑: 严谨的 BN 冻结 + 验证集去抖动
"""
import os
import csv
import math
import platform
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch.amp as amp

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import SimpleFolderDataset, HorizonImageDataset

# =========================
# Config
# =========================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
IMG_CLEAR_DIR = r"Hashmani's Dataset/clear"
DCE_WEIGHTS = r"weights/Epoch99.pth"

SPLIT_DIR = r"splits_musid"
TRAIN_IDX_PATH = os.path.join(SPLIT_DIR, "train_indices.npy")
VAL_IDX_PATH   = os.path.join(SPLIT_DIR, "val_indices.npy")
TEST_IDX_PATH  = os.path.join(SPLIT_DIR, "test_indices.npy")

# 当前运行阶段: 'A' -> 'B' -> 'C1' -> 'B2' -> 'C2'
STAGE = "A"

# ✅ 核心改动 1: 16:9 无黑边输入
IMG_SIZE = (576, 1024) 
BATCH_SIZE = 4

# ✅ 核心改动 2: 让网络学会“不乱改”清晰图
P_CLEAN = 0.35

SEED = 42
PRINT_EVERY = 1
EVAL_EVERY = 1
SAVE_VIS = True
VIS_MAX = 8

STAGE_CFG = {
    "A":  dict(lr=2e-4, epochs=50),
    "B":  dict(lr=1e-4, epochs=20),
    "C1": dict(lr=5e-5, epochs=1),
    "B2": dict(lr=5e-5, epochs=5),
    "C2": dict(lr=2e-5, epochs=49),
}

JOINT_SEG_W = 0.5
# C2 阶段固定使用 seg_w = 1.0

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

# 辅助函数：冻结 BN
def freeze_bn(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def safe_load(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def set_requires_grad(module, flag: bool):
    if module is None: return
    for p in module.parameters(): p.requires_grad = flag

def get_modules(model, names):
    return [getattr(model, n, None) for n in names]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_fixed_split_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (os.path.exists(TRAIN_IDX_PATH) and os.path.exists(VAL_IDX_PATH) and os.path.exists(TEST_IDX_PATH)):
        raise FileNotFoundError(f"未找到固定划分文件: {SPLIT_DIR}")
    tr = np.load(TRAIN_IDX_PATH)
    va = np.load(VAL_IDX_PATH)
    te = np.load(TEST_IDX_PATH)
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)

def build_musid_datasets(stage: str):
    train_idx, val_idx, test_idx = load_fixed_split_indices()
    if stage in ("B", "B2"):
        mode = "segmentation"
        eval_mode = "seg"
    else:
        mode = "joint"
        eval_mode = "joint"

    # 分别实例化，控制 augment 和 p_clean
    full_ds_train = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode=mode, augment=True, p_clean=P_CLEAN)
    full_ds_val   = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode=mode, augment=False, p_clean=P_CLEAN)

    train_ds = Subset(full_ds_train, train_idx.tolist())
    val_ds   = Subset(full_ds_val,   val_idx.tolist())

    print(f"[Fixed Split] Train(Aug=True)={len(train_ds)} | Val(Aug=False)={len(val_ds)} | Test={len(test_idx)}")
    return train_ds, val_ds, eval_mode

# ✅ 核心改动 3: 智能权重加载 (自动找上一阶段的 Best)
def load_checkpoint_smart(model, current_stage: str, device: str):
    if current_stage == "A": return # A 从头训
    
    # 依赖链
    priority_map = {
        "B":  [("A", "best_joint"), ("A", "last")], 
        "C1": [("B", "best_seg"), ("B", "last"), ("A", "best_joint")], 
        "B2": [("C1", "best_joint"), ("C1", "last"), ("B", "best_seg")], 
        "C2": [("B2", "best_seg"), ("B2", "last"), ("C1", "best_joint")]
    }
    
    if current_stage not in priority_map: return
    
    for prev_stage, kind in priority_map[current_stage]:
        prev = prev_stage.lower()
        if kind == "best_joint": fname = f"weights_new/rghnet_best_{prev}.pth"
        elif kind == "best_seg": fname = f"weights_new/rghnet_best_seg_{prev}.pth"
        elif kind == "last": fname = f"weights_new/rghnet_last_{prev}.pth"
        else: continue

        if os.path.exists(fname):
            print(f"[Init] Stage {current_stage}: Found predecessor weight '{fname}'")
            try:
                state = safe_load(fname, device)
                model.load_state_dict(state, strict=False)
                print(f"       -> Successfully loaded.")
                return
            except Exception as e:
                print(f"       -> Load failed: {e}, trying next...")
    
    print(f"[Init] Warning: No suitable weights found for Stage {current_stage}. Starting from scratch/backbone.")


# =========================
# Losses & Metrics
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
        kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
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

# ✅ 核心改动 4: 加入 FFT Loss (频域损失)
class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        return self.criterion(torch.abs(pred_fft), torch.abs(target_fft))

class HybridRestorationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.fft = FFTLoss() 

    def forward(self, pred, target):
        # 加上频域损失，抑制周期性海浪噪声
        return self.charb(pred, target) + 0.1 * self.edge(pred, target) + 0.1 * self.fft(pred, target)

def build_optimizer(model, stage: str, lr: float):
    # 模块定义
    restoration_names = [
        "rest_lat2", "rest_lat3", "rest_lat4", "rest_lat5",
        "ca2", "ca3", "ca4", "ca5", # CoordAtt
        "rest_fuse", "rest_strip", "rest_out",
        "rest_up1", "rest_conv1", "rest_up2", "rest_conv2", "rest_up3", "rest_conv3", "rest_up4",
    ]
    segmentation_names = [
        "seg_lat3", "seg_lat4", "seg_lat5",
        "seg_ca3", "seg_ca4", "seg_ca5", # CoordAtt
        "seg_fuse", "seg_strip", "seg_head", "seg_final", "inject",
        "strip_pool", "seg_conv_fuse", "injection_conv",
    ]
    restoration_modules = get_modules(model, restoration_names)
    segmentation_modules = get_modules(model, segmentation_names)

    set_requires_grad(model.encoder, False)
    for m in restoration_modules: set_requires_grad(m, False)
    for m in segmentation_modules: set_requires_grad(m, False)
    if hasattr(model, "dce_net"): set_requires_grad(getattr(model, "dce_net", None), False)

    encoder_params = list(model.encoder.parameters())
    
    def collect_params(modules):
        params = []
        for m in modules:
            if m is not None: params += list(m.parameters())
        return params

    rest_params = collect_params(restoration_modules)
    seg_params = collect_params(segmentation_modules)

    if stage == "A":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules: set_requires_grad(m, True)
        return optim.AdamW([{"params": encoder_params, "lr": lr * 0.1}, {"params": rest_params, "lr": lr}], weight_decay=1e-4)

    if stage in ("B", "B2"):
        for m in segmentation_modules: set_requires_grad(m, True)
        return optim.AdamW([{"params": seg_params, "lr": lr}], weight_decay=1e-4)

    if stage == "C1":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules: set_requires_grad(m, True)
        return optim.AdamW([{"params": encoder_params, "lr": lr * 0.1}, {"params": rest_params, "lr": lr}], weight_decay=1e-4)

    if stage == "C2":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules: set_requires_grad(m, True)
        for m in segmentation_modules: set_requires_grad(m, True)
        return optim.AdamW(
            [{"params": encoder_params, "lr": lr * 0.1},
             {"params": rest_params, "lr": lr},
             {"params": seg_params, "lr": lr}], weight_decay=1e-4)

    raise ValueError(f"Unknown stage: {stage}")

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0

def seg_metrics_from_masks(pred: torch.Tensor, gt: torch.Tensor, ignore_index: int = 255) -> Dict[str, float]:
    pred = pred.view(-1); gt = gt.view(-1)
    valid = gt != ignore_index
    pred = pred[valid]; gt = gt[valid]
    if pred.numel() == 0: return dict(pixel_acc=0, iou_sky=0, dice=0, precision=0, recall=0)
    correct = (pred == gt).sum().item()
    total = pred.numel()
    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    return dict(
        pixel_acc=_safe_div(correct, total),
        iou_sky=_safe_div(tp, tp + fp + fn),
        dice=_safe_div(2 * tp, 2 * tp + fp + fn),
        precision=_safe_div(tp, tp + fp),
        recall=_safe_div(tp, tp + fn)
    )

def horizon_mae_np(pred_mask: np.ndarray, gt_mask: np.ndarray, ignore: int = 255) -> float:
    H, W = gt_mask.shape
    abs_err = []
    ys = np.arange(H)
    for x in range(W):
        gt_col = gt_mask[:, x]; valid = gt_col != ignore
        if not np.any(valid): continue
        y_valid = ys[valid]; gt_valid = gt_col[valid]
        sky_y = y_valid[gt_valid == 1]; sea_y = y_valid[gt_valid == 0]
        if sky_y.size > 0 and sea_y.size > 0:
            y_sky = int(sky_y.max()); y_sea = int(sea_y.min())
            y_gt = (y_sky + y_sea) / 2.0 if y_sea > y_sky else float(y_sky)
        elif sky_y.size > 0: y_gt = float(sky_y.max())
        elif sea_y.size > 0: y_gt = float(sea_y.min())
        else: continue

        pred_col = pred_mask[:, x]
        sky_p = np.where(pred_col == 1)[0]; sea_p = np.where(pred_col == 0)[0]
        if sky_p.size > 0 and sea_p.size > 0:
            y_sky_p = int(sky_p.max()); y_sea_p = int(sea_p.min())
            y_pr = (y_sky_p + y_sea_p) / 2.0 if y_sea_p > y_sky_p else float(y_sky_p)
        elif sky_p.size > 0: y_pr = float(sky_p.max())
        elif sea_p.size > 0: y_pr = float(sea_p.min())
        else: continue
        abs_err.append(abs(y_pr - y_gt))
    return float(np.mean(abs_err)) if len(abs_err) > 0 else 0.0

def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> float:
    mse_val = float(mse.item())
    if mse_val <= 0: return 99.0
    return 10.0 * math.log10((max_val * max_val) / mse_val)

def save_vis(out_dir: str, prefix: str, img_tensor: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor):
    ensure_dir(out_dir)
    img = (img_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    gt = gt_mask.detach().cpu().numpy().astype(np.uint8)
    pr = pred_mask.detach().cpu().numpy().astype(np.uint8)
    H, W, _ = img.shape
    gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
    gt_vis[gt == 1] = (255, 255, 255); gt_vis[gt == 255] = (127, 127, 127)
    pr_vis = np.zeros((H, W, 3), dtype=np.uint8)
    pr_vis[pr == 1] = (255, 255, 255)
    overlay = img.copy()
    
    top = np.concatenate([img, gt_vis], axis=1)
    bot = np.concatenate([overlay, pr_vis], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    try:
        import cv2
        cv2.imwrite(os.path.join(out_dir, f"{prefix}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image
        Image.fromarray(grid).save(os.path.join(out_dir, f"{prefix}.png"))

@dataclass
class EvalResult:
    rest_loss: float = 0.0; seg_loss: float = 0.0; joint_loss: float = 0.0
    psnr: float = 0.0; pixel_acc: float = 0.0; iou_sky: float = 0.0
    dice: float = 0.0; precision: float = 0.0; recall: float = 0.0; horizon_mae: float = 0.0

@torch.no_grad()
def evaluate(model, loader, mode: str, crit_rest, crit_seg, seg_w: float = 0.5) -> EvalResult:
    model.eval()
    n = 0
    sums = {k: 0.0 for k in ["rest", "seg", "joint", "psnr", "acc", "iou", "dice", "prec", "rec", "mae"]}

    for batch in loader:
        if mode == "seg":
            img, mask = batch
            img = img.to(DEVICE, non_blocking=True); mask = mask.to(DEVICE, non_blocking=True)
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                _, seg, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
                loss_s = crit_seg(seg, mask)
            pred = seg.argmax(1)
            sums["seg"] += float(loss_s.item()); sums["joint"] += float(loss_s.item())
        elif mode == "joint":
            img, target, mask = batch
            img = img.to(DEVICE, non_blocking=True); target = target.to(DEVICE, non_blocking=True); mask = mask.to(DEVICE, non_blocking=True)
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                restored, seg, target_dce = model(img, target, enable_restoration=True, enable_segmentation=True)
                # ✅ 核心改动 5: Loss 监督 Clean Target (消除色偏)
                loss_r = crit_rest(restored, target) 
                loss_s = crit_seg(seg, mask)
                loss_joint = loss_r + seg_w * loss_s
            pred = seg.argmax(1)
            mse = F.mse_loss(restored.detach().float(), target.detach().float())
            sums["psnr"] += psnr_from_mse(mse)
            sums["rest"] += float(loss_r.item()); sums["seg"] += float(loss_s.item()); sums["joint"] += float(loss_joint.item())
        elif mode == "rest":
            img, target = batch
            img = img.to(DEVICE, non_blocking=True); target = target.to(DEVICE, non_blocking=True)
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                restored, _, _ = model(img, target, enable_restoration=True, enable_segmentation=False)
                # ✅ 核心改动 5: Loss 监督 Clean Target
                loss_r = crit_rest(restored, target)
            mse = F.mse_loss(restored.detach().float(), target.detach().float())
            sums["psnr"] += psnr_from_mse(mse)
            sums["rest"] += float(loss_r.item()); sums["joint"] += float(loss_r.item())
            pred = None
        else: raise ValueError(f"Unknown eval mode: {mode}")

        if pred is not None:
            m = seg_metrics_from_masks(pred.detach(), mask.detach())
            sums["acc"] += m["pixel_acc"]; sums["iou"] += m["iou_sky"]; sums["dice"] += m["dice"]
            sums["prec"] += m["precision"]; sums["rec"] += m["recall"]
            pred_np = pred.detach().cpu().numpy(); gt_np = mask.detach().cpu().numpy()
            bsz = pred_np.shape[0]; h_mae = 0.0
            for i in range(bsz): h_mae += horizon_mae_np(pred_np[i], gt_np[i])
            sums["mae"] += (h_mae / max(1, bsz))
        n += 1

    if n == 0: return EvalResult()
    return EvalResult(
        rest_loss=sums["rest"]/n, seg_loss=sums["seg"]/n, joint_loss=sums["joint"]/n, psnr=sums["psnr"]/n,
        pixel_acc=sums["acc"]/n, iou_sky=sums["iou"]/n, dice=sums["dice"]/n,
        precision=sums["prec"]/n, recall=sums["rec"]/n, horizon_mae=sums["mae"]/n
    )

def append_log(csv_path: str, row: Dict):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new: w.writeheader()
        w.writerow(row)

# =========================
# Main
# =========================
def main():
    if STAGE not in STAGE_CFG: raise ValueError(f"STAGE must be one of {list(STAGE_CFG.keys())}, got {STAGE}")
    seed_everything(SEED)
    lr = STAGE_CFG[STAGE]["lr"]; epochs = STAGE_CFG[STAGE]["epochs"]
    print(f"=== Start Training: Stage {STAGE} ===")
    print(f"DEVICE={DEVICE}, BS={BATCH_SIZE}")

    if STAGE == "A":
        # Stage A 也需要区分增强/非增强
        ds_train_full = SimpleFolderDataset(IMG_CLEAR_DIR, img_size=IMG_SIZE, augment=True, p_clean=P_CLEAN)
        ds_val_full   = SimpleFolderDataset(IMG_CLEAR_DIR, img_size=IMG_SIZE, augment=False, p_clean=P_CLEAN)
        n = len(ds_train_full)
        g = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(n, generator=g).tolist()
        n_val = max(1, int(n * 0.2))
        train_ds = Subset(ds_train_full, perm[n_val:]); val_ds = Subset(ds_val_full, perm[:n_val])
        eval_mode = "rest"
        print(f"[Stage A Split] train={len(train_ds)} val={len(val_ds)}")
    else:
        train_ds, val_ds, eval_mode = build_musid_datasets(STAGE)

    num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE=="cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE=="cuda"), drop_last=False)

    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    
    # 智能加载
    load_checkpoint_smart(model, STAGE, DEVICE)

    crit_rest = HybridRestorationLoss().to(DEVICE)
    crit_seg = nn.CrossEntropyLoss(ignore_index=255).to(DEVICE)
    optimizer = build_optimizer(model, STAGE, lr)
    try: scaler = amp.GradScaler(device=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda"))
    except TypeError: scaler = amp.GradScaler(enabled=(DEVICE_TYPE == "cuda"))

    best_iou = -1.0; best_mae = 1e9; best_joint = 1e9
    log_path = f"train_log_stage_{STAGE.lower()}.csv"
    ensure_dir("val_vis")
    ensure_dir("weights_new")  # 确保目录存在
    
    # ✅ 核心改动 6: 独立命名，不覆盖，保存到 weights_new/
    best_joint_name = f"weights_new/rghnet_best_{STAGE.lower()}.pth"
    best_seg_name   = f"weights_new/rghnet_best_seg_{STAGE.lower()}.pth"
    last_name       = f"weights_new/rghnet_last_{STAGE.lower()}.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        
        # BN 冻结逻辑
        if STAGE in ("B", "B2"):
            model.apply(freeze_bn) # 先全冻结BN
            # 这里的逻辑是：只让 active 的部分 train，其他的 eval
            # 但 B/B2 是只训 Seg，所以 restoration/encoder 必须 eval
            # 为了保险，再次显式设置
            if hasattr(model, "encoder"): model.encoder.eval()
            restoration_names = ["rest_lat2", "rest_lat3", "rest_lat4", "rest_lat5", "ca2", "ca3", "ca4", "ca5", 
                                 "rest_fuse", "rest_strip", "rest_out", "rest_up1", "rest_conv1", "rest_up2", "rest_conv2", "rest_up3", "rest_conv3", "rest_up4"]
            for n in restoration_names:
                m = getattr(model, n, None)
                if m is not None: m.eval()

        curr_seg_w = JOINT_SEG_W
        if STAGE == "C2": curr_seg_w = 1.0  # 全程保持 1.0，强调分割性能

        loop = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}")
        loss_sum = 0.0

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)
            
            # Forward Logic
            if STAGE=="A":
                img, target = batch
                img=img.to(DEVICE); target=target.to(DEVICE)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE=="cuda")):
                    r, _, _ = model(img, target, True, False)
                    loss = crit_rest(r, target) # Clean
            elif STAGE in ("B", "B2"):
                img, mask = batch
                img=img.to(DEVICE); mask=mask.to(DEVICE)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE=="cuda")):
                    _, s, _ = model(img, None, True, True)
                    loss = crit_seg(s, mask)
            elif STAGE == "C1":
                img, target, _ = batch
                img=img.to(DEVICE); target=target.to(DEVICE)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE=="cuda")):
                    r, _, _ = model(img, target, True, False)
                    loss = crit_rest(r, target) # Clean
            else: # C2
                img, target, mask = batch
                img=img.to(DEVICE); target=target.to(DEVICE); mask=mask.to(DEVICE)
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE=="cuda")):
                    r, s, _ = model(img, target, True, True)
                    loss = crit_rest(r, target) + curr_seg_w * crit_seg(s, mask) # Clean
            
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        train_loss = loss_sum / max(1, len(train_loader))
        if epoch % PRINT_EVERY == 0: print(f"[Train] epoch={epoch}  loss={train_loss:.6f}")
        torch.save(model.state_dict(), last_name)

        if (epoch % EVAL_EVERY) == 0:
            val_res = evaluate(model, val_loader, eval_mode, crit_rest, crit_seg, seg_w=curr_seg_w)
            print(f"[Val] J={val_res.joint_loss:.4f} R={val_res.rest_loss:.4f} S={val_res.seg_loss:.4f} IoU={val_res.iou_sky:.3f}")

            # 权重保存
            if val_res.joint_loss < best_joint:
                best_joint = val_res.joint_loss
                torch.save(model.state_dict(), best_joint_name)
                print(f"  -> New Best Joint: {best_joint_name}")

            has_seg = (eval_mode in ["seg", "joint"])
            if has_seg:
                improve = (val_res.iou_sky > best_iou + 1e-6) or (abs(val_res.iou_sky - best_iou) <= 1e-6 and val_res.horizon_mae < best_mae)
                if improve:
                    best_iou = val_res.iou_sky; best_mae = val_res.horizon_mae
                    torch.save(model.state_dict(), best_seg_name)
                    print(f"  -> New Best Seg: {best_seg_name}")
            append_log(log_path, dict(stage=STAGE, epoch=epoch, train_loss=train_loss, val_joint=val_res.joint_loss, val_iou=val_res.iou_sky, val_mae=val_res.horizon_mae))

    print(f"Stage {STAGE} done.")

if __name__ == "__main__":
    main()