import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import platform  # 用于检测操作系统

# NOTE: 统一使用 unet_model.py 里的实现
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import SimpleFolderDataset, HorizonImageDataset

# ================= 配置区域 =================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
DCE_WEIGHTS = "Epoch99.pth"  # 确保文件存在
IMG_CLEAR_DIR = r"Hashmani's Dataset/clear"

# 训练阶段选择: 'A', 'B', 'C'
STAGE = 'B'

BATCH_SIZE = 16
IMG_SIZE = 384

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 工具函数 =================

def safe_load(path, map_location):
    """兼容不同 PyTorch 版本的安全加载"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only 参数
        return torch.load(path, map_location=map_location)

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
        kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
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

    # 模块分组
    restoration_modules = [
        getattr(model, "rest_lat2", None), getattr(model, "rest_lat3", None),
        getattr(model, "rest_lat4", None), getattr(model, "rest_lat5", None),
        getattr(model, "rest_fuse", None), getattr(model, "rest_strip", None),
        getattr(model, "rest_out", None), getattr(model, "rest_up1", None),
        getattr(model, "rest_conv1", None), getattr(model, "rest_up2", None),
        getattr(model, "rest_conv2", None), getattr(model, "rest_up3", None),
        getattr(model, "rest_conv3", None), getattr(model, "rest_up4", None)
    ]
    segmentation_modules = [
        getattr(model, "seg_lat3", None), getattr(model, "seg_lat4", None),
        getattr(model, "seg_lat5", None), getattr(model, "seg_fuse", None),
        getattr(model, "seg_strip", None), getattr(model, "seg_final", None),
        getattr(model, "seg_head", None), getattr(model, "inject", None),
    ]

    # 1. 先全冻结
    set_requires_grad(model.encoder, False)
    for m in restoration_modules: set_requires_grad(m, False)
    for m in segmentation_modules: set_requires_grad(m, False)
    if hasattr(model, "dce_net") and model.dce_net is not None:
        set_requires_grad(model.dce_net, False) # DCE 永远冻结

    # 2. 按 Stage 解冻
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
        # 此时只解冻 segmentation
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
    print(f"=== 正在启动训练: Stage {STAGE} ===")
    
    # 1. 数据集准备
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

    # 2. DataLoader (针对 Windows 优化 num_workers)
    # Windows 下多进程容易卡死，建议设为 0；Linux 可以设为 4 或 8
    num_workers = 0 if platform.system() == 'Windows' else 8
    print(f"Using num_workers: {num_workers}")
    
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True
    )

    # 3. 模型初始化
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)

    # 4. 权重加载 (接力逻辑)
    if STAGE == 'B':
        ckpt_a = "rghnet_stage_a.pth"
        if os.path.exists(ckpt_a):
            print(f"--> [Stage B] Loading Stage A weights from {ckpt_a}...")
            model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)
        else:
            raise FileNotFoundError(f"错误：Stage B 需要加载 {ckpt_a}，但文件不存在！请先跑 Stage A。")
    
    elif STAGE == 'C':
        ckpt_b = "rghnet_stage_b.pth"
        ckpt_a = "rghnet_stage_a.pth"
        if os.path.exists(ckpt_b):
            print(f"--> [Stage C] Loading Stage B weights from {ckpt_b}...")
            model.load_state_dict(safe_load(ckpt_b, DEVICE), strict=False)
        elif os.path.exists(ckpt_a):
            print(f"--> [Stage C] Warning: Stage B not found, loading Stage A from {ckpt_a}...")
            model.load_state_dict(safe_load(ckpt_a, DEVICE), strict=False)
        else:
            print("--> [Stage C] Warning: No checkpoint found! Training from scratch (Not recommended).")

    # 5. 优化器与 Scaler
    optimizer = build_optimizer(model, STAGE, LR)
    # 修复：新版 torch.amp 接口
    scaler = torch.amp.GradScaler(DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda"))

    # 6. Loss
    crit_rest = HybridRestorationLoss()
    crit_seg = nn.CrossEntropyLoss(ignore_index=255)

    # 7. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        
        # === [核心修复] Stage B 必须强制冻结 BN 统计量 ===
        if STAGE == 'B':
            # 将 Encoder 和 Restoration 分支设为 eval 模式
            # 这样 BN 就会使用 Stage A 学到的 global mean/var，而不是当前 Batch 的
            model.encoder.eval()
            
            # 遍历 restoration 相关模块设为 eval
            # 注意：这里要确保覆盖所有 restoration 层
            modules_to_freeze = [
                model.rest_up1, model.rest_conv1, model.rest_up2, model.rest_conv2,
                model.rest_up3, model.rest_conv3, model.rest_up4, model.rest_out,
                # 以及其他的 latent convs
                model.rest_lat2, model.rest_lat3, model.rest_lat4, model.rest_lat5,
                model.rest_fuse, model.rest_strip
            ]
            for m in modules_to_freeze:
                if m is not None: m.eval()
        # ==================================================

        loop = tqdm(loader, desc=f"Ep {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            if STAGE == 'A':
                img, target = batch
                img, target = img.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    restored, _, target_dce = model(img, target, enable_restoration=True, enable_segmentation=False)
                    loss = crit_rest(restored, target_dce)

            elif STAGE == 'B':
                img, mask = batch
                img, mask = img.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                    # Stage B: 开启复原分支计算(用于特征注入)，但不算 Loss
                    _, seg, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
                    loss = crit_seg(seg, mask)

            else:  # STAGE == 'C'
                img, target, mask = batch
                img = img.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
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