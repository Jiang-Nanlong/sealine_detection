# make_fusion_cache.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入你的模块
from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# ================= 配置 =================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
SAVE_DIR = r"Hashmani's Dataset/FusionCache_v2"  # 建议用新目录名，避免覆盖旧数据

# 权重路径
RGHNET_CKPT = "rghnet_best_joint.pth" # 用你训练最好的那个
DCE_WEIGHTS = "Epoch99.pth"

# 尺寸配置
IMG_SIZE_UNET = 384
RESIZE_H = 2240
RESIZE_W = 180

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================================

def calculate_radon_label(x1, y1, x2, y2, img_w, img_h, resize_h, resize_w):
    """计算归一化的 rho, theta 标签 (0~1)"""
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1
    line_angle = np.arctan2(dy, dx)
    theta_rad = line_angle - np.pi / 2
    while theta_rad < 0: theta_rad += np.pi
    while theta_rad >= np.pi: theta_rad -= np.pi
    
    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)
    
    # 归一化 (逻辑必须和 dataset_loader_gradient_radon_cnn 一致)
    label_theta = np.degrees(theta_rad) / 180.0
    
    # 我们用 padding 模式，所以 rho 的归一化是相对于 padded height
    # 先计算 rho 在原始 sinogram 中的像素位置
    original_diag = np.sqrt(img_w**2 + img_h**2)
    rho_pixel_pos = rho + original_diag / 2.0
    
    # 再映射到 padding 后的坐标
    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top
    
    label_rho = final_rho_idx / (resize_h - 1)
    
    return np.clip(label_rho, 0, 1), np.clip(label_theta, 0, 1)

def process_sinogram(sino, target_h, target_w):
    """归一化并 Padding 到统一尺寸"""
    mi, ma = sino.min(), sino.max()
    if ma - mi > 1e-6:
        sino_norm = (sino - mi) / (ma - mi)
    else:
        sino_norm = np.zeros_like(sino)
    
    h_curr = sino_norm.shape[0]
    container = np.zeros((target_h, target_w), dtype=np.float32)
    start_h = (target_h - h_curr) // 2
    
    if h_curr <= target_h:
        container[start_h : start_h + h_curr, :] = sino_norm
    else:
        crop_start = (h_curr - target_h) // 2
        container[:, :] = sino_norm[crop_start : crop_start + target_h, :]
    return container

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # 1. 加载 RG-HNet
    print(f"Loading RG-HNet from {RGHNET_CKPT}...")
    seg_model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(RGHNET_CKPT):
        state = torch.load(RGHNET_CKPT, map_location=DEVICE)
        seg_model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {RGHNET_CKPT}")
    seg_model.eval()
    
    # 2. 加载传统提取器
    print("Loading Traditional Extractor...")
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    
    # 3. 读取数据
    df = pd.read_csv(CSV_PATH, header=None)
    
    theta_scan = np.linspace(0., 180., RESIZE_W, endpoint=False)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_name = str(row.iloc[0])
        try:
            x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        except:
            continue
            
        # 读图
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path): img_path += '.JPG'
        if not os.path.exists(img_path): continue
        
        image = cv2.imread(img_path) # BGR
        if image is None: continue
        h_img, w_img = image.shape[:2]
        
        # --- 统一前端处理 ---
        # 1. 预处理 (resize, to tensor, to device)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE_UNET, IMG_SIZE_UNET))
        inp_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
        
        # 2. RG-HNet 推理 (一次推理，得到两个重要输出)
        with torch.no_grad():
            # 得到复原图 和 分割Logits
            restored_img_tensor, seg_logits, _ = seg_model(inp_tensor, None)
            
            # 提取 Mask
            pred_mask = seg_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # --- A. 传统特征流 (使用复原图) ---
        # 1. 将复原图 Tensor 转为 OpenCV 格式 (Numpy, BGR, 0-255)
        restored_np_rgb = (restored_img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # resize 回原图尺寸，再做梯度计算，精度更高
        restored_np_full = cv2.resize(restored_np_rgb, (w_img, h_img))
        restored_cv2_bgr = cv2.cvtColor(restored_np_full, cv2.COLOR_RGB2BGR)
        
        # 2. **关键修改：输入复原图**
        _, _, _, trad_sinograms = detector.detect(restored_cv2_bgr)
        
        # 3. 处理
        processed_trad = []
        for s in trad_sinograms[:3]:
            processed_trad.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_trad) < 3:
            processed_trad.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))
            
        # --- B. 深度语义流 (使用分割Mask) ---
        pred_mask_full = cv2.resize(pred_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        
        edges = cv2.Canny(pred_mask_full * 255, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        seg_sino_raw = detector._radon_gpu(edges, theta_scan)
        processed_seg = process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W)
        
        # --- C. 融合与保存 ---
        combined_input = np.stack(processed_trad + [processed_seg], axis=0)
        
        # 标签计算 (与传统方法保持一致)
        l_rho, l_theta = calculate_radon_label(x1, y1, x2, y2, w_img, h_img, RESIZE_H, RESIZE_W)
        
        np.save(os.path.join(SAVE_DIR, f"{idx}.npy"), {
            'input': combined_input,
            'label': np.array([l_rho, l_theta])
        })

    print(f"Done! Cache saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()