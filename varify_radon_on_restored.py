import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# 导入你的模块
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import synthesize_rain_fog, letterbox_rgb_u8
from gradient_radon import TextureSuppressedMuSCoWERT

# ================= 配置 =================
# 图片路径 (找一张海天线明显的图测试)
IMG_PATH = r"Hashmani's Dataset/MU-SID/DSC_0622_2.JPG" 
# 或者是 dataset_loader 能找到的任意一张图

# 权重路径
UNET_CKPT = "rghnet_best_joint.pth"  # 或者 rghnet_stage_a.pth
DCE_WEIGHTS = "Epoch99.pth"

IMG_SIZE = 1024  # UNet 推理尺寸
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tensor_to_numpy(tensor):
    """(C, H, W) tensor -> (H, W, C) numpy uint8 BGR for OpenCV"""
    img = tensor.squeeze().detach().cpu().float().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    img = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGB -> BGR

def draw_candidates(image, candidates, color=(0, 255, 255), alpha=0.3):
    """画出所有候选线"""
    vis = image.copy()
    h, w = image.shape[:2]
    cx = w / 2
    
    # 按照分数从低到高画，这样高分的线会盖在上面
    candidates = sorted(candidates, key=lambda x: x['score'])
    
    overlay = vis.copy()
    for cand in candidates:
        Y, a = cand['Y'], cand['alpha']
        t = np.tan(np.deg2rad(a))
        
        # 计算直线端点
        y1 = int(Y - t * cx)
        y2 = int(Y + t * (w - cx))
        
        cv2.line(overlay, (0, y1), (w, y2), color, 1)
        
    # 半透明叠加，避免完全遮挡原图
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis

def main():
    # 1. 准备模型
    print(f"Loading UNet from {UNET_CKPT}...")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    
    if os.path.exists(UNET_CKPT):
        state = torch.load(UNET_CKPT, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    else:
        print("Warning: Checkpoint not found, using initialized weights.")
    model.eval()

    # 2. 准备输入图片 (模拟退化)
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image not found at {IMG_PATH}")
        return

    bgr_raw = cv2.imread(IMG_PATH)
    rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
    
    # Letterbox resize
    rgb_resized, meta = letterbox_rgb_u8(rgb_raw, IMG_SIZE, pad_value=0)
    
    # 合成雨雾 (Input)
    random.seed(42) # 固定随机性以便复现
    rgb_degraded_np = synthesize_rain_fog(rgb_resized) # 返回 float32
    
    # 转 Tensor
    input_tensor = torch.from_numpy(rgb_degraded_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # 3. UNet 推理 (获取 Restored Image)
    print("Running UNet inference...")
    with torch.no_grad():
        # forward 返回: restored, seg, target_dce
        restored_tensor, _, _ = model(input_tensor, enable_restoration=True, enable_segmentation=False)
    
    # 转回 OpenCV 格式 (BGR uint8)
    restored_bgr = tensor_to_numpy(restored_tensor)
    
    # 同时也把 Input 转回来对比
    input_bgr = tensor_to_numpy(input_tensor)

    # 4. Radon 变换检测
    print("Running Radon Transform on Restored Image...")
    # 初始化你的检测器
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    
    # 检测
    final_res, all_candidates, debug_info, _ = detector.detect(restored_bgr)
    
    if final_res is None:
        print("Radon failed to detect any lines.")
        return

    # 5. 可视化结果
    print("Visualizing...")
    
    # A. 绘制所有候选线
    vis_candidates = draw_candidates(restored_bgr, all_candidates, color=(0, 255, 255), alpha=0.4)
    
    # B. 绘制最佳线 (红色加粗)
    best_Y, best_alpha = final_res
    h, w = restored_bgr.shape[:2]
    cx = w / 2
    t = np.tan(np.deg2rad(best_alpha))
    y1 = int(best_Y - t * cx)
    y2 = int(best_Y + t * (w - cx))
    cv2.line(vis_candidates, (0, y1), (w, y2), (0, 0, 255), 3)

    # C. 获取特征图 (Edge Map) - 看看过曝是否弄丢了边缘
    # 我们取 Scale 2 的特征图来看看
    edge_map = debug_info[2]['map'] # scale=2 usually is the best balance
    
    # D. 为了对比，我们也跑一下原始 Input (有雨雾的) 的 Radon
    _, _, debug_info_input, _ = detector.detect(input_bgr)
    edge_map_input = debug_info_input[2]['map']

    # --- 绘图 ---
    plt.figure(figsize=(20, 10))

    # 1. 原始退化输入
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
    plt.title("1. Input (Degraded)")
    plt.axis("off")

    # 2. Input 的特征图 (通常很乱，因为有雨)
    plt.subplot(2, 3, 4)
    plt.imshow(edge_map_input, cmap='gray')
    plt.title("4. Input Edge Map (Noisy?)")
    plt.axis("off")

    # 3. UNet 复原图 (观察是否泛白/色偏)
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))
    plt.title("2. Restored (Note Color Shift)")
    plt.axis("off")

    # 4. Restored 的特征图 (关键！看海天线边缘是否清晰)
    plt.subplot(2, 3, 5)
    plt.imshow(edge_map, cmap='gray')
    plt.title("5. Restored Edge Map (Is Horizon Clear?)")
    plt.axis("off")

    # 5. 最终检测结果 (所有候选+最佳)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(vis_candidates, cv2.COLOR_BGR2RGB))
    plt.title(f"3. Radon Candidates\nBest: alpha={best_alpha:.1f}")
    plt.axis("off")
    
    # 6. Sinogram (正弦图)
    # 取 scale 2 的 sinogram
    sinograms = detector.get_sinograms(restored_bgr) # 需确保你的 detector 有这个 helper，或者直接用 detector._radon_gpu
    # 这里简单处理，如果不方便获取，可以忽略。但上面的 detector.detect 已经返回了 collected_sinograms
    # detector.detect 返回值的第四个是 collected_sinograms list
    # 我们在上面接收了: final_res, all_candidates, debug_info, sinograms_list = ...
    # 假设 dataset_loader_gradient_radon_cnn.py 里的逻辑返回了 sinograms
    
    # 这里为了通用性，我直接画 edge map 即可，sinogram 比较抽象
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, "Check Fig 5.\nIf the white line is clear,\nRadon works fine.", 
             ha='center', va='center', fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()