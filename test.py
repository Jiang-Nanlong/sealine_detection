import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

from unet_model import RestorationGuidedHorizonNet
from cnn_model import HorizonResNet
from gradient_radon import TextureSuppressedMuSCoWERT
from dataset_loader import synthesize_rain_fog

# ================= 配置 =================
IMG_DIR = r"Hashmani's Dataset/clear"
OUT_DIR = r"demo_results"

UNET_CKPT = "rghnet_best_c2.pth"       
CNN_CKPT = "splits_musid/best_fusion_cnn_1024x576.pth" 
DCE_WEIGHTS = "Epoch99.pth"

ENABLE_DEGRADATION = True
UNET_IN_W, UNET_IN_H = 1024, 576
# Backward-compatible aliases
UNET_W, UNET_H = UNET_IN_W, UNET_IN_H
RESIZE_H, RESIZE_W = 2240, 180

NUM_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
# =======================================

def process_sinogram(sino, th, tw):
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6: sino_norm = (sino - mi) / (ma - mi)
    else: sino_norm = np.zeros_like(sino, dtype=np.float32)
    h = sino_norm.shape[0]
    c = np.zeros((th, tw), dtype=np.float32)
    st = (th - h) // 2
    if h <= th: c[st:st+h, :] = sino_norm
    else: 
        cs = (h - th) // 2
        c[:, :] = sino_norm[cs:cs+th, :]
    return c

def post_process_mask(mask_np):
    # 简单的连通域处理，保持一致性
    valid = (mask_np != 255)
    sky = ((mask_np == 1) & valid).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_TOP] <= 0: keep[labels == i] = 1
    out = mask_np.copy()
    out[(mask_np == 1) & (keep == 0)] = 0
    return out

def get_line_ends(rho_norm, theta_norm, w, h):
    # 1. 还原到 resize_h 尺度
    diag = np.sqrt(w**2 + h**2)
    pad_top = (RESIZE_H - diag) / 2.0
    rho_real = rho_norm * (RESIZE_H - 1) - pad_top - (diag / 2.0)
    
    theta_rad = np.deg2rad(theta_norm * 180.0)
    
    # 2. 计算端点
    cx, cy = w / 2.0, h / 2.0
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x0 = cos_t * rho_real
    y0 = sin_t * rho_real
    
    scale = max(w, h) * 2
    pt1 = (int(cx + x0 - scale * sin_t), int(cy + y0 + scale * cos_t))
    pt2 = (int(cx + x0 + scale * sin_t), int(cy + y0 - scale * cos_t))
    return pt1, pt2

def main():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    random.seed(SEED); torch.manual_seed(SEED)
    
    print("Loading models...")
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    unet.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE), strict=False)
    unet.eval()

    cnn = HorizonResNet(in_channels=4).to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location=DEVICE))
    cnn.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
    samples = random.sample(imgs, min(NUM_SAMPLES, len(imgs)))

    print(f"Testing {len(samples)} images...")

    for i, path in enumerate(samples):
        bgr = cv2.imread(path)
        if bgr is None: continue
        h_orig, w_orig = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Resize to UNet size (1024x576)
        rgb_unet = cv2.resize(rgb, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_LINEAR)
        
        if ENABLE_DEGRADATION:
            rgb_in_np = synthesize_rain_fog(rgb_unet, p_clean=0.0)
        else:
            rgb_in_np = rgb_unet.astype(np.float32) / 255.0

        t_in = torch.from_numpy(rgb_in_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2. UNet
        with torch.no_grad():
            res_t, seg_t, _ = unet(t_in, None, True, True)
        
        res_np = (res_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask_np = seg_t.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 3. Post process
        res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)
        mask_pp = post_process_mask(mask_np)

        # 4. Feature Extraction (on 1024x576)
        try: _, _, _, trads = detector.detect(res_bgr)
        except: trads = []
        
        stack = []
        for s in trads[:3]: stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(stack) < 3: stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))
        
        edges = cv2.Canny((mask_pp * 255).astype(np.uint8), 50, 150)
        seg_sino = detector._radon_gpu(edges, theta_scan)
        stack.append(process_sinogram(seg_sino, RESIZE_H, RESIZE_W))

        cnn_in = torch.from_numpy(np.stack(stack)).float().unsqueeze(0).to(DEVICE)

        # 5. CNN Predict
        with torch.no_grad():
            preds = cnn(cnn_in).cpu().numpy()[0]
        
        # 6. Visualize
        # 注意：preds 是基于 1024x576 的，我们需要把它画在原图 1920x1080 上
        # 所以先在 1024x576 上算出端点，再按比例缩放端点坐标
        
        pt1_s, pt2_s = get_line_ends(preds[0], preds[1], UNET_IN_W, UNET_IN_H)
        
        scale_x = w_orig / UNET_IN_W
        scale_y = h_orig / UNET_IN_H
        
        pt1_orig = (int(pt1_s[0] * scale_x), int(pt1_s[1] * scale_y))
        pt2_orig = (int(pt2_s[0] * scale_x), int(pt2_s[1] * scale_y))

        vis_final = bgr.copy()
        cv2.line(vis_final, pt1_orig, pt2_orig, (0, 0, 255), 3)

        # Save
        vis_in_show = cv2.resize(cv2.cvtColor(rgb_in_np, cv2.COLOR_RGB2BGR), (w_orig, h_orig))
        res_show = cv2.resize(res_bgr, (w_orig, h_orig))
        vis_in_show = (vis_in_show * 255).astype(np.uint8)

        comb = np.hstack([vis_in_show, res_show, vis_final])
        
        # Resize for display convenience (too wide)
        h_disp = 400
        r = h_disp / h_orig
        comb_disp = cv2.resize(comb, (int(comb.shape[1]*r), h_disp))
        
        prefix = "degraded" if ENABLE_DEGRADATION else "clean"
        cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_{i}.png"), comb_disp)
        print(f"Saved {i}")

    print("Done")

if __name__ == "__main__":
    main()