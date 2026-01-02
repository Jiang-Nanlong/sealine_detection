import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# NOTE: 统一使用 unet_model.py 里的实现
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import SimpleFolderDataset, HorizonImageDataset

# ================= 配置区域 =================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
DCE_WEIGHTS = "Epoch99.pth"  # 确保文件存在
IMG_CLEAR_DIR = r"Hashmani's Dataset/clear" # Stage A 用的清晰图文件夹

# 'A', 'B', 'C'
STAGE = 'A'

BATCH_SIZE = 16
IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= Loss Functions =================
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
        img = F.pad(img, (kw // 2, kw // 2, kh // 2, kh // 2), mode='replicate')
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
        self.l1 = CharbonnierLoss()
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        return self.l1(pred, target) + 0.1 * self.edge(pred, target)


# ================= 优化器构建 =================
def build_optimizer(model, stage: str, lr: float):
    def set_requires_grad(module, flag: bool):
        if module is None: return
        for p in module.parameters():
            p.requires_grad = flag

    # --- 模块列表 ---
    restoration_modules = [
        getattr(model, "rest_lat2", None), getattr(model, "rest_lat3", None),
        getattr(model, "rest_lat4", None), getattr(model, "rest_lat5", None),
        getattr(model, "rest_fuse", None), getattr(model, "rest_strip", None),
        getattr(model, "rest_out", None),
    ]
    segmentation_modules = [
        getattr(model, "seg_lat3", None), getattr(model, "seg_lat4", None),
        getattr(model, "seg_lat5", None), getattr(model, "seg_fuse", None),
        getattr(model, "seg_strip", None), getattr(model, "seg_final", None),
        getattr(model, "inject", None),
    ]

    # 先全关
    set_requires_grad(model.encoder, False)
    for m in restoration_modules: set_requires_grad(m, False)
    for m in segmentation_modules: set_requires_grad(m, False)

    # DCE 永远冻结
    if hasattr(model, "dce_net") and model.dce_net is not None:
        set_requires_grad(model.dce_net, False)

    # --- 按 stage 解冻 ---
    if stage == "A":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules: set_requires_grad(m, True)
        
        encoder_params = list(model.encoder.parameters())
        rest_params = []
        for m in restoration_modules:
            if m is not None: rest_params += list(m.parameters())

        params = [
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": rest_params, "lr": lr},
        ]

    elif stage == "B":
        seg_params = []
        for m in segmentation_modules:
            if m is not None:
                set_requires_grad(m, True)
                seg_params += list(m.parameters())
        params = [{"params": seg_params, "lr": lr}]

    elif stage == "C":
        set_requires_grad(model.encoder, True)
        for m in restoration_modules: set_requires_grad(m, True)
        for m in segmentation_modules: set_requires_grad(m, True)

        encoder_params = list(model.encoder.parameters())
        rest_params, seg_params = [], []
        for m in restoration_modules:
            if m is not None: rest_params += list(m.parameters())
        for m in segmentation_modules:
            if m is not None: seg_params += list(m.parameters())

        params = [
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": rest_params, "lr": lr},
            {"params": seg_params, "lr": lr},
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return optim.AdamW(params, weight_decay=1e-4)


# ================= 主函数 =================
def main():
    if STAGE == 'A':
        LR, EPOCHS = 2e-4, 50
        print(f"Dataset: SimpleFolderDataset (Stage A) from {IMG_CLEAR_DIR}")
        ds = SimpleFolderDataset(IMG_CLEAR_DIR, img_size=IMG_SIZE)
    elif STAGE == 'B':
        LR, EPOCHS = 1e-4, 20
        print("Dataset: HorizonImageDataset segmentation (Stage B)")
        ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode='segmentation')
    else:
        LR, EPOCHS = 5e-5, 50
        print("Dataset: HorizonImageDataset joint (Stage C)")
        ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode='joint')

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True
    )

    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)

    optimizer = build_optimizer(model, STAGE, LR)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    crit_rest = HybridRestorationLoss()
    crit_seg = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Ep {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            if STAGE == 'A':
                img, target = batch
                img, target = img.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                # 修复 Autocast：显式指定 device_type="cuda"
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                    # Stage A: 只训练复原
                    restored, _, target_dce = model(img, target, enable_restoration=True, enable_segmentation=False)
                    loss = crit_rest(restored, target_dce)

            elif STAGE == 'B':
                img, mask = batch
                img, mask = img.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                    # Stage B: 只训练分割
                    _, seg, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
                    loss = crit_seg(seg, mask)

            else:  # STAGE == 'C'
                img, target, mask = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                    restored, seg, target_dce = model(img, target, enable_restoration=True, enable_segmentation=True)
                    loss_r = crit_rest(restored, target_dce)
                    loss_s = crit_seg(seg, mask)
                    loss = loss_r + 0.5 * loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        print(f"Epoch {epoch + 1} Avg Loss: {epoch_loss / max(1, len(loader)):.5f}")
        torch.save(model.state_dict(), f"rghnet_stage_{STAGE.lower()}.pth")


if __name__ == "__main__":
    main()