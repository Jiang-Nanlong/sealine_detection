import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import synthesize_rain_fog, letterbox_rgb_u8
from gradient_radon import TextureSuppressedMuSCoWERT

# ================= 配置 =================
IMG_DIR = r"Hashmani's Dataset/clear"
CKPT_PATH = "weights/rghnet_best_a.pth"
DCE_WEIGHTS = "weights/Epoch99.pth"

IMG_SIZE = 1024
NUM_SAMPLES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# =======================================

def tensor_to_bgr_uint8(tensor):
    img = tensor.squeeze().detach().cpu().float().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def draw_all_candidates_on_img(img_bgr, candidates_list, best_tuple):
    """
    画出所有候选线 + 最佳线
    Scale 1 (小): 蓝色 (255, 0, 0)
    Scale 2 (中): 绿色 (0, 255, 0)
    Scale 3 (大): 黄色 (0, 255, 255)
    Best: 红色粗线 (0, 0, 255)
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    cx = w / 2
    
    # 颜色映射 (BGR)
    colors = {
        1: (255, 0, 0),   # Blue
        2: (0, 255, 0),   # Green
        3: (0, 255, 255)  # Yellow
    }

    # 1. 先画所有候选线 (细线)
    # 按分数从低到高排序，让强的线盖在弱的线上面
    candidates_list = sorted(candidates_list, key=lambda x: x['score'])
    
    for cand in candidates_list:
        scale = cand['scale']
        Y, alpha = cand['Y'], cand['alpha']
        
        t = np.tan(np.deg2rad(alpha))
        y1 = int(Y - t * cx)
        y2 = int(Y + t * (w - cx))
        
        color = colors.get(scale, (200, 200, 200))
        # 线宽 1，半透明效果靠眼睛脑补（或者本来就细）
        cv2.line(vis, (0, y1), (w, y2), color, 1)

    # 2. 再画最佳线 (粗红线，覆盖在上面)
    if best_tuple is not None:
        Y, alpha = best_tuple
        t = np.tan(np.deg2rad(alpha))
        y1 = int(Y - t * cx)
        y2 = int(Y + t * (w - cx))
        cv2.line(vis, (0, y1), (w, y2), (0, 0, 255), 3)

    return vis

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"[Init] Loading model: {CKPT_PATH}")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    else:
        print(f"[Error] Checkpoint not found: {CKPT_PATH}")
        return 
    model.eval()

    # 初始化检测器
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

    all_imgs = glob.glob(os.path.join(IMG_DIR, "*.[jJ][pP]*[gG]")) + \
               glob.glob(os.path.join(IMG_DIR, "*.png"))
    
    if len(all_imgs) < NUM_SAMPLES:
        print("Not enough images.")
        return
    
    all_imgs = sorted(list(set(all_imgs)))
    samples = random.sample(all_imgs, NUM_SAMPLES)
    
    fig, axes = plt.subplots(nrows=NUM_SAMPLES, ncols=4, figsize=(20, 4 * NUM_SAMPLES))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    cols = ["1. Degraded Input", "2. ROI (Green Box)", "3. ROI Feature Map", "4. Multi-Scale Radon"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, pad=10)

    print(f"[Run] Processing {NUM_SAMPLES} images...")

    for i, path in enumerate(samples):
        bgr_raw = cv2.imread(path)
        if bgr_raw is None: continue
        rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
        
        # 1. 预处理
        rgb_resized, meta = letterbox_rgb_u8(rgb_raw, IMG_SIZE, pad_value=0)
        
        # 2. 模拟退化
        rgb_degraded_float = synthesize_rain_fog(rgb_resized)
        inp_tensor = torch.from_numpy(rgb_degraded_float).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # 3. UNet 推理
        with torch.no_grad():
            restored_tensor, _, _ = model(inp_tensor, enable_restoration=True, enable_segmentation=False)
        
        restored_bgr = tensor_to_bgr_uint8(restored_tensor)
        input_bgr = tensor_to_bgr_uint8(inp_tensor)

        # ================= [核心：ROI 提取] =================
        pad_top = int(meta['pad_top'])
        new_h = int(meta['new_h'])
        ARTIFACT_WIDTH = 16 
        
        roi_y1 = pad_top + ARTIFACT_WIDTH
        roi_y2 = pad_top + new_h - ARTIFACT_WIDTH
        
        if roi_y2 <= roi_y1:
            roi_y1 = pad_top; roi_y2 = pad_top + new_h

        roi_bgr = restored_bgr[roi_y1 : roi_y2, :]
        
        # ================= [核心：检测与坐标映射] =================
        
        # 1. 检测 (拿到所有候选)
        final_res_roi, all_candidates_roi, debug_info_roi, _ = detector.detect(roi_bgr)
        
        # 2. 映射所有候选线坐标
        all_candidates_full = []
        for cand in all_candidates_roi:
            c = cand.copy()
            c['Y'] += roi_y1  # Y轴偏移
            all_candidates_full.append(c)
            
        # 3. 映射最佳结果坐标
        final_res_full = None
        if final_res_roi is not None:
            y_roi, alpha = final_res_roi
            final_res_full = (y_roi + roi_y1, alpha)
            
        # 4. 准备特征图用于显示 (Scale 2)
        edge_map_roi = debug_info_roi[2]['map']
        edge_map_full = np.zeros((IMG_SIZE, IMG_SIZE), dtype=edge_map_roi.dtype)
        edge_map_full[roi_y1 : roi_y2, :] = edge_map_roi
        
        # ================= [可视化] =================
        
        # 这里的函数改为了 draw_all_candidates_on_img
        final_vis = draw_all_candidates_on_img(restored_bgr, all_candidates_full, final_res_full)
        
        # ROI 示意框
        vis_restored = restored_bgr.copy()
        cv2.rectangle(vis_restored, (0, roi_y1), (IMG_SIZE-1, roi_y2), (0, 255, 0), 2)

        axes[i, 0].imshow(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
        axes[i, 1].imshow(cv2.cvtColor(vis_restored, cv2.COLOR_BGR2RGB))
        axes[i, 2].imshow(edge_map_full, cmap='gray')
        axes[i, 3].imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))

        for ax in axes[i]: ax.axis('off')

    out_file = "vis_radon_multiscale.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=100)
    plt.show()
    print(f"[Done] Result saved to {os.path.abspath(out_file)}")

if __name__ == "__main__":
    main()