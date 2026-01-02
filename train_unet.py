import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# AMP (new API, no FutureWarning)
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

# 'A' / 'B' / 'C'
STAGE = "B"

BATCH_SIZE = 16
IMG_SIZE = 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils
# =========================
def safe_load(path: str, map_location: str):
    """torch.load without FutureWarning (weights_only=True), with backward-compat fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # older torch
        return torch.load(path, map_location=map_location)


def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def set_eval(module):
    if module is None:
        return
    module.eval()


def get_modules(model, names):
    """Return list of modules (None if not exist) using getattr."""
    return [getattr(model, n, None) for n in names]


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
# Optimizer builder
# =========================
def build_optimizer(model, stage: str, lr: float):
    # Restoration modules (FPN-style + legacy UNet-style, both supported via getattr)
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
    ]

    restoration_modules = get_modules(model, restoration_names)
    segmentation_modules = get_modules(model, segmentation_names)

    # 1) Freeze all first
    set_requires_grad(model.encoder, False)
    for m in restoration_modules:
        set_requires_grad(m, False)
    for m in segmentation_modules:
        set_requires_grad(m, False)

    # DCE always frozen
    if hasattr(model, "dce_net"):
        set_requires_grad(getattr(model, "dce_net", None), False)

    # 2) Unfreeze per stage
    if stage == "A":
        # train encoder + restoration
        set_requires_grad(model.encoder, True)
        for m in restoration_modules:
            set_requires_grad(m, True)

        encoder_params = list(model.encoder.parameters())
        rest_params = []
        for m in restoration_modules:
            if m is not None:
                rest_params += list(m.parameters())

        return optim.AdamW(
            [
                {"params": encoder_params, "lr": lr * 0.1},
                {"params": rest_params, "lr": lr},
            ],
            weight_decay=1e-4,
        )

    if stage == "B":
        # only train segmentation (encoder/restoration frozen)
        seg_params = []
        for m in segmentation_modules:
            if m is not None:
                set_requires_grad(m, True)
                seg_params += list(m.parameters())

        if len(seg_params) == 0:
            raise RuntimeError(
                "[build_optimizer] Stage B: no segmentation parameters found. "
                "Check your unet_model.py module names."
            )

        return optim.AdamW([{"params": seg_params, "lr": lr}], weight_decay=1e-4)

    if stage == "C":
        # train encoder + restoration + segmentation
        set_requires_grad(model.encoder, True)
        for m in restoration_modules:
            set_requires_grad(m, True)
        for m in segmentation_modules:
            set_requires_grad(m, True)

        encoder_params = list(model.encoder.parameters())
        rest_params, seg_params = [], []
        for m in restoration_modules:
            if m is not None:
                rest_params += list(m.parameters())
        for m in segmentation_modules:
            if m is not None:
                seg_params += list(m.parameters())

        return optim.AdamW(
            [
                {"params": encoder_params, "lr": lr * 0.1},
                {"params": rest_params, "lr": lr},
                {"params": seg_params, "lr": lr},
            ],
            weight_decay=1e-4,
        )

    raise ValueError(f"Unknown stage: {stage}")


# =========================
# Main
# =========================
def main():
    print(f"=== Start Training: Stage {STAGE} ===")
    print(f"DEVICE={DEVICE}, DEVICE_TYPE={DEVICE_TYPE}")

    # 1) Dataset
    if STAGE == "A":
        LR, EPOCHS = 2e-4, 50
        print(f"Dataset: SimpleFolderDataset (Stage A) from {IMG_CLEAR_DIR}")
        ds = SimpleFolderDataset(IMG_CLEAR_DIR, img_size=IMG_SIZE)
    elif STAGE == "B":
        LR, EPOCHS = 1e-4, 20
        print("Dataset: HorizonImageDataset segmentation (Stage B)")
        ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="segmentation")
    else:  # "C"
        LR, EPOCHS = 5e-5, 50
        print("Dataset: HorizonImageDataset joint (Stage C)")
        ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="joint")

    # 2) DataLoader
    # Windows: num_workers=0 is safest (avoid random hanging)
    num_workers = 0 if os.name == "nt" else 4
    print(f"Using num_workers={num_workers}")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    # 3) Model
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)

    # 4) Checkpoint relay
    if STAGE == "B":
        ckpt_a = "rghnet_stage_a.pth"
        if not os.path.exists(ckpt_a):
            raise FileNotFoundError(f"Stage B requires {ckpt_a}, but it does not exist. Run Stage A first.")
        print(f"[Init] Loading Stage A weights: {ckpt_a}")
        model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)

    if STAGE == "C":
        ckpt_b = "rghnet_stage_b.pth"
        ckpt_a = "rghnet_stage_a.pth"
        if os.path.exists(ckpt_b):
            print(f"[Init] Loading Stage B weights: {ckpt_b}")
            model.load_state_dict(safe_load(ckpt_b, DEVICE), strict=False)
        elif os.path.exists(ckpt_a):
            print(f"[Init] Stage B not found, loading Stage A weights: {ckpt_a}")
            model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)
        else:
            print("[Init] WARNING: no checkpoint found. Training from scratch.")

    # 5) Optimizer / Loss / AMP scaler
    optimizer = build_optimizer(model, STAGE, LR)

    crit_rest = HybridRestorationLoss().to(DEVICE)
    crit_seg = nn.CrossEntropyLoss(ignore_index=255)

    scaler = amp.GradScaler(DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda"))

    # Modules list for Stage B BN-freeze (robust)
    restoration_names = [
        "rest_lat2", "rest_lat3", "rest_lat4", "rest_lat5",
        "rest_fuse", "rest_strip", "rest_out",
        "rest_up1", "rest_conv1", "rest_up2", "rest_conv2",
        "rest_up3", "rest_conv3", "rest_up4",
    ]
    restoration_modules = get_modules(model, restoration_names)

    # 6) Train
    for epoch in range(EPOCHS):
        model.train()

        # --- Critical: Stage B freezes encoder/restoration -> also freeze BN running stats ---
        if STAGE == "B":
            set_eval(model.encoder)
            for m in restoration_modules:
                set_eval(m)
            # DCE is already frozen, but keep it eval explicitly
            set_eval(getattr(model, "dce_net", None))

        loop = tqdm(loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            if STAGE == "A":
                img, target = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)

                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    restored, _, target_dce = model(
                        img, target,
                        enable_restoration=True,
                        enable_segmentation=False
                    )
                    loss = crit_rest(restored, target_dce)

            elif STAGE == "B":
                img, mask = batch
                img = img.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)

                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    # keep enable_restoration=True if your seg branch uses injection
                    _, seg, _ = model(
                        img, None,
                        enable_restoration=True,
                        enable_segmentation=True
                    )
                    loss = crit_seg(seg, mask)

            else:  # STAGE == "C"
                img, target, mask = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)

                with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    restored, seg, target_dce = model(
                        img, target,
                        enable_restoration=True,
                        enable_segmentation=True
                    )
                    loss_r = crit_rest(restored, target_dce)
                    loss_s = crit_seg(seg, mask)
                    loss = loss_r + 0.5 * loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")

        save_name = f"rghnet_stage_{STAGE.lower()}.pth"
        torch.save(model.state_dict(), save_name)

    print(f"Stage {STAGE} done. Model saved to {save_name}")


if __name__ == "__main__":
    main()
