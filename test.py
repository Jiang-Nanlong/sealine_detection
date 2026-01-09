import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import math

# 引入你的模型定义
from unet_model import RestorationGuidedHorizonNet
from cnn_model import HorizonResNet
from gradient_radon import TextureSuppressedMuSCoWERT
from dataset_loader import synthesize_rain_fog, letterbox_rgb_u8

# ================= 配置区域 =================
IMG_DIR = r"Hashmani's Dataset/clear"  # 测试图片目录
OUT_DIR = r"demo_results"              # 结果保存目录

# 权重路径
UNET_CKPT = "rghnet_best_c2.pth"       
CNN_CKPT = "splits_musid/best_fusion_cnn.pth" 
DCE_WEIGHTS = "Epoch99.pth"

# [新功能] 是否开启雨雾退化开关
# True  = 模拟恶劣天气 (加雨加雾 -> 复原 -> 检测)
# False = 使用原始清晰图 (直接复原 -> 检测，验证模型在好天气下是否正常工作)
ENABLE_DEGRADATION = True  

# 参数
IMG_SIZE_UNET = 1024
RESIZE_H = 2240
RESIZE_W = 180

NUM_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
# ===========================================

def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """归一化并 Padding 到 CNN 需要的尺寸"""
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6:
        sino_norm = (sino - mi) / (ma - mi)
    else:
        sino_norm = np.zeros_like(sino, dtype=np.float32)

    h_curr = sino_norm.shape[0]
    container = np.zeros((target_h, target_w), dtype=np.float32)
    start_h = (target_h - h_curr) // 2

    if h_curr <= target_h:
        container[start_h:start_h + h_curr, :] = sino_norm
    else:
        crop_start = (h_curr - target_h) // 2
        container[:, :] = sino_norm[crop_start:crop_start + target_h, :]
    return container

def draw_line_from_rho_theta(img, rho_norm, theta_norm, color=(0, 0, 255), thickness=3):
    """根据 CNN 预测的归一化 rho, theta 在图片上画线"""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    diag_len = np.sqrt(h**2 + w**2)

    theta_deg = theta_norm * 180.0
    theta_rad = np.deg2rad(theta_deg)

    pad_top_sino = (RESIZE_H - diag_len) / 2.0
    rho_idx = rho_norm * (RESIZE_H - 1)
    rho_real = rho_idx - pad_top_sino - (diag_len / 2.0)

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    x0 = cos_t * rho_real
    y0 = sin_t * rho_real
    
    scale = max(h, w) * 2
    pt1_x = int(cx + x0 - scale * sin_t)
    pt1_y = int(cy + y0 + scale * cos_t)
    pt2_x = int(cx + x0 + scale * sin_t)
    pt2_y = int(cy + y0 - scale * cos_t)
    
    cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, thickness, cv2.LINE_AA)
    return img

def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Loading Models ===")
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(UNET_CKPT):
        unet.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE), strict=False)
        print(f"Loaded UNet: {UNET_CKPT}")
    else:
        print(f"[Error] UNet weights not found: {UNET_CKPT}")
        return
    unet.eval()

    cnn = HorizonResNet(in_channels=4).to(DEVICE)
    if os.path.exists(CNN_CKPT):
        cnn.load_state_dict(torch.load(CNN_CKPT, map_location=DEVICE))
        print(f"Loaded CNN: {CNN_CKPT}")
    else:
        print(f"[Error] CNN weights not found: {CNN_CKPT}")
        return
    cnn.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    all_imgs = glob.glob(os.path.join(IMG_DIR, "*.[jJ][pP]*[gG]")) + glob.glob(os.path.join(IMG_DIR, "*.png"))
    all_imgs = sorted(list(set(all_imgs)))
    
    if len(all_imgs) == 0:
        print(f"No images found in {IMG_DIR}")
        return

    samples = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))

    mode_str = "Degraded" if ENABLE_DEGRADATION else "Clean"
    print(f"=== Starting Inference ({mode_str} Mode) ===")

    for i, path in enumerate(samples):
        # A. 读取
        bgr_raw = cv2.imread(path)
        if bgr_raw is None: continue
        rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
        
        # Letterbox
        rgb_sq, meta = letterbox_rgb_u8(rgb_raw, IMG_SIZE_UNET, pad_value=0)
        
        # [开关控制] 决定是否加雨雾
        if ENABLE_DEGRADATION:
            # 模拟恶劣环境：输出 float32 [0,1]
            rgb_input_np = synthesize_rain_fog(rgb_sq)
        else:
            # 保持清晰环境：直接归一化 float32 [0,1]
            rgb_input_np = rgb_sq.astype(np.float32) / 255.0
        
        inp_tensor = torch.from_numpy(rgb_input_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # B. UNet 推理
        with torch.no_grad():
            restored_t, seg_logits, _ = unet(inp_tensor, None, True, True)
        
        restored_sq = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        mask_sq = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # C. ROI 提取 (去黑边)
        pad_top = int(meta["pad_top"])
        new_h = int(meta["new_h"])
        new_w = int(meta["new_w"])
        pad_left = int(meta["pad_left"])
        
        roi_restored = restored_sq[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        roi_mask = mask_sq[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        
        h_orig, w_orig = rgb_raw.shape[:2]
        roi_restored_orig = cv2.resize(roi_restored, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        roi_mask_orig = cv2.resize(roi_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        roi_restored_bgr = cv2.cvtColor(roi_restored_orig, cv2.COLOR_RGB2BGR)

        # D. 特征提取
        # 1. 传统特征
        try:
            _, _, _, trad_sinos = detector.detect(roi_restored_bgr)
        except: trad_sinos = []
        
        processed_stack = []
        for s in trad_sinos[:3]:
            processed_stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))
            
        # 2. 语义特征
        edges = cv2.Canny((roi_mask_orig * 255).astype(np.uint8), 50, 150)
        k = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)
        seg_sino_raw = detector._radon_gpu(edges, theta_scan)
        processed_stack.append(process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W))
        
        # 堆叠 -> Tensor
        cnn_input = np.stack(processed_stack, axis=0).astype(np.float32)
        cnn_input_t = torch.from_numpy(cnn_input).unsqueeze(0).to(DEVICE)

        # E. CNN 回归
        with torch.no_grad():
            preds = cnn(cnn_input_t).cpu().numpy()[0]
        
        pred_rho, pred_theta = preds[0], preds[1]

        # F. 结果可视化
        vis_final = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        vis_final = draw_line_from_rho_theta(vis_final, pred_rho, pred_theta, color=(0, 0, 255), thickness=3)

        # 拼接展示
        vis_input_show = cv2.resize(cv2.cvtColor(rgb_input_np, cv2.COLOR_RGB2BGR), (w_orig, h_orig))
        vis_input_show = (vis_input_show * 255).astype(np.uint8) # float to uint8 for display
        
        vis_restored_roi = roi_restored_bgr
        
        h_show = 500
        def resize_h(img, h):
            r = h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1]*r), h))
            
        show_1 = resize_h(vis_input_show, h_show)
        show_2 = resize_h(vis_restored_roi, h_show)
        show_3 = resize_h(vis_final, h_show)
        
        combined = np.hstack([show_1, show_2, show_3])
        
        # 文件名带上模式前缀，防止覆盖
        prefix = "degraded" if ENABLE_DEGRADATION else "clean"
        save_path = os.path.join(OUT_DIR, f"{prefix}_result_{i}.png")
        cv2.imwrite(save_path, combined)
        print(f"[{i+1}/{len(samples)}] Saved: {save_path}")

    print("Done!")

if __name__ == "__main__":
    main()