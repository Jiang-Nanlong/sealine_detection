import os
import random
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import SimpleFolderDataset


# =========================
# 你只需要改这里
# =========================
IMG_CLEAR_DIR = r"Hashmani's Dataset/clear"   # Stage A 用的 clear 图目录
CKPT_PATH = r"rghnet_best_a.pth"              # 训练好的 Stage A 权重（可空字符串表示不加载）
DCE_WEIGHTS = r"Epoch99.pth"                  # 你的 DCE 权重路径（和训练一致）

OUT_PATH = r"vis/stageA_debug_grid.png"
N_SAMPLES = 8
IMG_SIZE = 1024

# 想看“训练时的数据分布”就 True；想看“验证风格（更稳定）”就 False
AUGMENT = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def tensor_to_img(t: torch.Tensor):
    """(3,H,W) [0,1] -> uint8 (H,W,3)"""
    t = t.detach().float().clamp(0, 1).cpu()
    return (t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")


def psnr(a: torch.Tensor, b: torch.Tensor, max_val=1.0) -> float:
    mse = F.mse_loss(a.detach().float(), b.detach().float()).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val * max_val) / mse)


@torch.no_grad()
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(DEVICE)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # 1) Dataset: Stage A 的 SimpleFolderDataset 应该返回 (img_degraded, target_clean)
    ds = SimpleFolderDataset(IMG_CLEAR_DIR, img_size=IMG_SIZE, augment=AUGMENT)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in: {IMG_CLEAR_DIR}")

    n = min(N_SAMPLES, len(ds))
    idx = random.sample(range(len(ds)), n)
    subset = Subset(ds, idx)
    loader = DataLoader(subset, batch_size=n, shuffle=False, num_workers=0)

    img_in, target_clean = next(iter(loader))
    img_in = img_in.to(device)
    target_clean = target_clean.to(device)

    # 2) Model
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(device)
    model.eval()

    if CKPT_PATH and os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[Load] ckpt: {CKPT_PATH}")
    else:
        print(f"[Warn] CKPT not loaded (file not found or empty): {CKPT_PATH}")

    # 3) Forward: Stage A restoration only，但模型会吐出 target_dce
    with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
        restored, _, target_dce = model(
            img_in, target_clean,
            enable_restoration=True,
            enable_segmentation=False
        )

    # 4) Plot grid: N rows x 4 cols
    cols = 4
    fig, axes = plt.subplots(nrows=n, ncols=cols, figsize=(cols * 4, n * 3))
    if n == 1:
        axes = axes.reshape(1, -1)

    titles = ["Input (degraded)", "Target (clean)", "Target (DCE)", "Restored (model)"]

    for i in range(n):
        imgs = [
            tensor_to_img(img_in[i]),
            tensor_to_img(target_clean[i]),
            tensor_to_img(target_dce[i]),
            tensor_to_img(restored[i]),
        ]

        p_clean = psnr(restored[i], target_clean[i])
        p_dce = psnr(restored[i], target_dce[i])

        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(imgs[j])
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j])

        # 只在第一列标注两种 PSNR，方便你快速判断
        axes[i, 0].text(
            0.01, 0.01,
            f"PSNR vs clean: {p_clean:.2f}\nPSNR vs DCE: {p_dce:.2f}",
            transform=axes[i, 0].transAxes,
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=3),
            va="bottom",
        )

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"[Saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
