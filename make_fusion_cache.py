# -*- coding: utf-8 -*-
"""
make_fusion_cache.py (Final C2 Version)
功能：
1. 读取原图 -> (Val/Test固定/Train随机)加雨雾 -> Letterbox -> UNet(C2)复原 -> 得到 Restored & Mask
2. Un-letterbox (去黑边) -> 提取有效区域 (ROI)
3. 传统分支: ROI -> 梯度 -> Radon
4. 语义分支: ROI -> Mask 边缘 -> Radon
5. 融合 -> 保存为 .npy
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch.amp as amp

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import synthesize_rain_fog  # 复用你的退化函数
from gradient_radon import TextureSuppressedMuSCoWERT

# ============================
# 配置
# ============================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
SPLIT_DIR = r"splits_musid"

# 输出目录 (建议加上 _c2 后缀以示区别)
SAVE_ROOT = r"Hashmani's Dataset/FusionCache_C2"

# 权重路径 (使用 C2 最终权重)
RGHNET_CKPT = r"rghnet_best_c2.pth"
DCE_WEIGHTS = r"Epoch99.pth"

# 参数保持一致
IMG_SIZE_UNET = 1024
RESIZE_H = 2240
RESIZE_W = 180

# 边界提取参数
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def letterbox_rgb_u8(image_rgb_u8, dst_size, pad_value=0):
    h, w = image_rgb_u8.shape[:2]
    scale = min(dst_size / float(w), dst_size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=interp)
    canvas = np.full((dst_size, dst_size, 3), pad_value, dtype=np.uint8)
    pad_left = (dst_size - new_w) // 2
    pad_top = (dst_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, dict(scale=scale, pad_left=pad_left, pad_top=pad_top, new_w=new_w, new_h=new_h, orig_w=w, orig_h=h)

def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """归一化并 padding/crop 到统一尺寸"""
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

def calculate_radon_label(x1, y1, x2, y2, img_w, img_h, resize_h, resize_w):
    """计算归一化的 rho, theta 标签 (0~1)"""
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1
    line_angle = np.arctan2(dy, dx)
    
    # 转换到 Radon 的 theta 定义 (法线角度)
    # 直线方程: x cos(theta) + y sin(theta) = rho
    # 法线角度 = 线角度 - 90度
    theta_rad = line_angle - np.pi / 2
    
    # 归一化到 [0, pi) 或 (-pi/2, pi/2) 取决于你的 radon 实现
    # 这里假设我们的一致性逻辑是 0~180度
    while theta_rad < 0: theta_rad += np.pi
    while theta_rad >= np.pi: theta_rad -= np.pi

    # 计算 rho (原点到直线的距离)
    # 用中点计算最稳
    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)

    label_theta = np.degrees(theta_rad) / 180.0

    # 映射 rho 到 pixel space (假设对角线长度为最大范围)
    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    # resize_h 对应对角线长度的映射
    # 中心对应 resize_h / 2
    rho_pixel_pos = rho + original_diag / 2.0 
    
    # 因为我们做了 padding (resize_h > diag)，所以有偏移
    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top
    
    label_rho = final_rho_idx / (resize_h - 1)

    return float(np.clip(label_rho, 0, 1)), float(np.clip(label_theta, 0, 1))

def _load_split_indices(split_dir):
    def _load(name):
        p = os.path.join(split_dir, name)
        return np.load(p).tolist() if os.path.exists(p) else []
    return {
        "train": _load("train_indices.npy"),
        "val":   _load("val_indices.npy"),
        "test":  _load("test_indices.npy"),
    }

def build_cache_for_split(df, indices, split_name, out_dir, seg_model, detector, theta_scan):
    ensure_dir(out_dir)
    print(f"Processing {split_name}: {len(indices)} images...")
    
    # 设置随机性策略
    is_train = (split_name == "train")
    
    for idx in tqdm(indices, ncols=80):
        row = df.iloc[idx]
        img_name = str(row.iloc[0])
        try:
            x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        except: continue

        path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(path):
            # 尝试常见后缀
            for ext in [".JPG", ".jpg", ".png", ".jpeg"]:
                if os.path.exists(path + ext): path += ext; break
        
        if not os.path.exists(path): continue
        bgr = cv2.imread(path)
        if bgr is None: continue
        
        h_orig, w_orig = bgr.shape[:2]
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Letterbox
        rgb_sq, meta = letterbox_rgb_u8(rgb0, IMG_SIZE_UNET, pad_value=0)
        
        # 2. 合成退化 (Degradation)
        # 关键：这里要复刻 train_unet.py 的逻辑
        if not is_train:
            # Val/Test: 固定种子
            state = random.getstate(); np_state = np.random.get_state()
            seed = int(idx) + 100000 if split_name == "val" else int(idx) + 200000
            random.seed(seed); np.random.seed(seed)
            rgb_degraded = synthesize_rain_fog(rgb_sq)
            random.setstate(state); np.random.set_state(np_state)
        else:
            # Train: 随机退化
            # 注意：Cache 只生成一次，所以这里实际上是“冻结”了训练集的某一种随机状态
            # 这对于 CNN 训练是可以接受的，甚至有助于稳定
            rgb_degraded = synthesize_rain_fog(rgb_sq)

        inp = torch.from_numpy(rgb_degraded).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 3. UNet Inference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                restored_t, seg_logits, _ = seg_model(inp, None, True, True)
        
        # 4. Un-Letterbox & Crop (恢复到原图有效区域)
        # 获取 restored (RGB) 和 mask (0/1)
        restored_sq = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        pred_mask_sq = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        pad_top = int(meta["pad_top"])
        new_h = int(meta["new_h"])
        new_w = int(meta["new_w"])
        pad_left = int(meta["pad_left"])
        
        # [关键] 只取中间有效部分，去除 padding 的黑边
        # 这样 Radon 就不会检测到上下黑边了
        roi_restored = restored_sq[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        roi_mask = pred_mask_sq[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        
        # 这里的 roi_restored 已经是去除黑边的了，但尺寸是缩放后的尺寸 (new_w, new_h)
        # 我们需要把它 resize 回原图尺寸 (w_orig, h_orig) 吗？
        # 实际上 detector 内部会处理尺寸。
        # 我们可以直接把 ROI resize 回原图尺寸，这样物理意义最明确
        roi_restored_orig = cv2.resize(roi_restored, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        roi_mask_orig = cv2.resize(roi_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        roi_restored_bgr = cv2.cvtColor(roi_restored_orig, cv2.COLOR_RGB2BGR)

        # 5. 特征提取
        # A) 传统特征 (Gradient Radon)
        try:
            # detect 返回的 sinograms 是一个列表
            _, _, _, trad_sinos = detector.detect(roi_restored_bgr)
        except Exception:
            trad_sinos = []

        processed_trad = []
        for s in trad_sinos[:3]:
            processed_trad.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        # 补齐
        while len(processed_trad) < 3:
            processed_trad.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

        # B) 语义特征 (Segmentation Edge Radon)
        # 提取 mask 边缘
        edges = cv2.Canny((roi_mask_orig * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
        if EDGE_DILATE > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=EDGE_DILATE)
        
        # 对边缘做 Radon
        seg_sino_raw = detector._radon_gpu(edges, theta_scan)
        processed_seg = process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W)

        # 6. 组合与保存
        combined_input = np.stack(processed_trad + [processed_seg], axis=0).astype(np.float32) # (4, H, W)
        
        # 计算 Label (在原图尺寸下计算)
        l_rho, l_theta = calculate_radon_label(x1, y1, x2, y2, w_orig, h_orig, RESIZE_H, RESIZE_W)
        label = np.array([l_rho, l_theta], dtype=np.float32)

        np.save(os.path.join(out_dir, f"{idx}.npy"), {"input": combined_input, "label": label})

def main():
    ensure_dir(SAVE_ROOT)
    splits = _load_split_indices(SPLIT_DIR)
    
    print(f"[Load] Model: {RGHNET_CKPT}")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(RGHNET_CKPT):
        state = torch.load(RGHNET_CKPT, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {RGHNET_CKPT}")
    model.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)
    
    df = pd.read_csv(CSV_PATH, header=None)

    for split_name in ["train", "val", "test"]:
        if len(splits[split_name]) > 0:
            build_cache_for_split(
                df, splits[split_name], split_name, 
                os.path.join(SAVE_ROOT, split_name), 
                model, detector, theta_scan
            )

    print("All done!")

if __name__ == "__main__":
    main()