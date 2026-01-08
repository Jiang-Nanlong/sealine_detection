import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import HorizonImageDataset

# ================= 配置区域 =================
# [切换开关] 这里填 'B' 或 'B2'
CURRENT_SUB_STAGE = "B"

# 权重路径自动生成
# Stage B/B2 主要看 Best Seg 权重
CKPT_PATH = f"rghnet_best_seg_{CURRENT_SUB_STAGE.lower()}.pth"

CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
DCE_WEIGHTS = "Epoch99.pth"

IMG_SIZE = 1024
NUM_SAMPLES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# ===========================================

def tensor_to_img(t):
    """(C,H,W) -> (H,W,C) numpy uint8"""
    t = t.squeeze().detach().cpu().float().clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

def mask_to_color(mask):
    """0=Sea(Black), 1=Sky(Blue), 255=Ignore(Gray)"""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask == 0] = [0, 0, 0]       # Sea
    vis[mask == 1] = [100, 200, 255] # Sky
    vis[mask == 255] = [128, 128, 128]
    return vis

def overlay_mask(img_rgb, mask, alpha=0.5):
    color_mask = mask_to_color(mask)
    mask_bool = (mask != 255) # 只叠加有效区域
    out = img_rgb.copy()
    out[mask_bool] = cv2.addWeighted(img_rgb[mask_bool], 1-alpha, color_mask[mask_bool], alpha, 0)
    return out

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"--- Visualizing Stage {CURRENT_SUB_STAGE} (Segmentation Focus) ---")
    print(f"Loading: {CKPT_PATH}")

    # 1. 数据 (Val模式: augment=False)
    ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="joint", augment=False)
    indices = random.sample(range(len(ds)), NUM_SAMPLES)
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # 2. 模型
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    else:
        print(f"[Error] Checkpoint not found: {CKPT_PATH}")
        return
    model.eval()

    # 3. 绘图
    cols = 3
    fig, axes = plt.subplots(nrows=NUM_SAMPLES, ncols=cols, figsize=(12, 3.5 * NUM_SAMPLES))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    headers = ["Degraded Input", f"Pred Mask ({CURRENT_SUB_STAGE})", "Overlay"]
    if NUM_SAMPLES > 1:
        for ax, h in zip(axes[0], headers): ax.set_title(h)
    else:
        for ax, h in zip(axes, headers): ax.set_title(h)

    with torch.no_grad():
        for i, (img, _, _) in enumerate(loader): # Stage B 不需要 target
            img = img.to(DEVICE)
            
            # Forward
            _, seg_logits, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
            
            vis_in = tensor_to_img(img[0])
            pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            vis_mask = mask_to_color(pred_mask)
            vis_overlay = overlay_mask(vis_in, pred_mask)

            cur_ax = axes[i] if NUM_SAMPLES > 1 else axes
            
            cur_ax[0].imshow(vis_in)
            cur_ax[1].imshow(vis_mask)
            cur_ax[2].imshow(vis_overlay)

            for ax in cur_ax: ax.axis('off')

    out_file = f"vis_stage_{CURRENT_SUB_STAGE}.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"[Done] Saved to {out_file}")

if __name__ == "__main__":
    main()