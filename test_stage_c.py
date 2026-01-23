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
# [切换开关] 这里填 'C1' 或 'C2'
CURRENT_SUB_STAGE = "C2"

# 权重路径
# C1 保存的是 best_joint (restoration loss), C2 也是 best_joint
CKPT_PATH = f"weights/rghnet_best_{CURRENT_SUB_STAGE.lower()}.pth"

CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
DCE_WEIGHTS = "weights/Epoch99.pth"

# [核心修正] 必须改成 (H, W) 元组，对应 16:9 无黑边
IMG_SIZE = (576, 1024)

NUM_SAMPLES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42 # 保持和 Stage B 一样的种子，方便对比同一张图的变化
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
    vis[mask == 255] = [128, 128, 128] # Ignore
    return vis

def overlay_mask(img_rgb, mask, alpha=0.5):
    color_mask = mask_to_color(mask)
    mask_bool = (mask != 255)
    out = img_rgb.copy()
    out[mask_bool] = cv2.addWeighted(img_rgb[mask_bool], 1-alpha, color_mask[mask_bool], alpha, 0)
    return out

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"--- Visualizing Stage {CURRENT_SUB_STAGE} (Joint/Restoration Focus) ---")
    print(f"Loading: {CKPT_PATH}")
    print(f"Image Size: {IMG_SIZE}")

    # 1. 数据加载 (Val模式: augment=False)
    # 注意：dataset_loader 会根据 tuple 尺寸自动使用 resize 模式 (无黑边)
    ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="joint", augment=False)
    
    indices = random.sample(range(len(ds)), NUM_SAMPLES)
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # 2. 模型加载
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print("Weights loaded successfully.")
    else:
        print(f"[Error] Checkpoint not found: {CKPT_PATH}")
        return
    model.eval()

    # 3. 绘图布局
    if CURRENT_SUB_STAGE == "C1":
        # C1: 重点看复原是否崩了 (Restoration Tuning)
        cols = 3
        headers = ["Degraded Input", "Restored (C1)", "Clean Target"]
    else:
        # C2: 终极展示 (Restored + Seg + Overlay)
        cols = 5
        headers = ["Input", "Restored", "Pred Mask", "Overlay", "GT Mask"]

    # 调整画布比例适应 16:9
    fig_h_unit = 2.5
    fig, axes = plt.subplots(nrows=NUM_SAMPLES, ncols=cols, figsize=(4*cols, fig_h_unit * NUM_SAMPLES))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    if NUM_SAMPLES > 1:
        for ax, h in zip(axes[0], headers): ax.set_title(h)
    else:
        for ax, h in zip(axes, headers): ax.set_title(h)

    with torch.no_grad():
        for i, (img, target, mask) in enumerate(loader):
            img = img.to(DEVICE)
            
            # Forward
            restored, seg_logits, _ = model(img, None, enable_restoration=True, enable_segmentation=True)
            
            vis_in = tensor_to_img(img[0])
            vis_restored = tensor_to_img(restored[0])
            vis_target = tensor_to_img(target[0])
            
            if seg_logits is not None:
                pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            else:
                pred_mask = np.zeros(vis_in.shape[:2], dtype=np.uint8)
                
            gt_mask = mask[0].cpu().numpy().astype(np.uint8)
            
            vis_pred = mask_to_color(pred_mask)
            vis_gt = mask_to_color(gt_mask)
            
            # 在复原图上叠加Mask，效果最好
            vis_overlay = overlay_mask(vis_restored, pred_mask)

            cur_ax = axes[i] if NUM_SAMPLES > 1 else axes
            
            if CURRENT_SUB_STAGE == "C1":
                cur_ax[0].imshow(vis_in)
                cur_ax[1].imshow(vis_restored)
                cur_ax[2].imshow(vis_target)
            else: # C2
                cur_ax[0].imshow(vis_in)
                cur_ax[1].imshow(vis_restored)
                cur_ax[2].imshow(vis_pred)
                cur_ax[3].imshow(vis_overlay)
                cur_ax[4].imshow(vis_gt)

            for ax in cur_ax: ax.axis('off')

    out_file = f"vis_stage_{CURRENT_SUB_STAGE}.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"[Done] Saved to {out_file}")

if __name__ == "__main__":
    main()