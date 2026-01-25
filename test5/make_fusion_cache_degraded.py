# -*- coding: utf-8 -*-
"""
make_fusion_cache_degraded.py

Build FusionCache for degraded MU-SID test images.

基于 make_fusion_cache.py 的相同处理流程，
为每种退化类型生成 cache 文件。

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.amp as amp
from tqdm import tqdm

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
# 选择要处理的退化类型，None 表示处理全部
SELECTED_DEGRADATIONS = None  # 或 ["gaussian_noise_15", "low_light_2.0"]
# ============================

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
DEGRADED_IMG_DIR = TEST5_DIR / "degraded_images"
SPLITS_DIR = PROJECT_ROOT / "splits_musid"

CACHE_ROOT = TEST5_DIR / "FusionCache_Degraded"

# MU-SID trained weights
RGHNET_CKPT = str(PROJECT_ROOT / "weights" / "rghnet_best_c2.pth")
DCE_WEIGHTS = str(PROJECT_ROOT / "weights" / "Epoch99.pth")

# Image sizes (与 make_fusion_cache.py 一致)
UNET_IN_W = 1024
UNET_IN_H = 576

# Sinogram 统一尺寸
RESIZE_H = 2240
RESIZE_W = 180

# 后处理参数
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_gt_csv():
    """Load GroundTruth_test.csv as DataFrame."""
    gt_csv = SPLITS_DIR / "GroundTruth_test.csv"
    # CSV 无 header: img_stem, x1, y1, x2, y2, mx, my, theta
    df = pd.read_csv(gt_csv, header=None)
    df.columns = ["img_stem", "x1", "y1", "x2", "y2", "mx", "my", "theta"]
    return df


def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Normalize and pad sinogram to target size."""
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
    """Calculate Radon space label (rho_norm, theta_norm)."""
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1

    line_angle = np.arctan2(dy, dx)
    theta_rad = line_angle - np.pi / 2
    while theta_rad < 0:
        theta_rad += np.pi
    while theta_rad >= np.pi:
        theta_rad -= np.pi

    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)

    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    rho_pixel_pos = rho + original_diag / 2.0
    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top

    label_rho = final_rho_idx / (resize_h - 1)
    label_theta = np.rad2deg(theta_rad) / 180.0

    return float(np.clip(label_rho, 0, 1)), float(np.clip(label_theta, 0, 1))


def post_process_mask_top_connected(mask_np):
    """Post-process segmentation mask."""
    valid = (mask_np != 255)
    sky = ((mask_np == 1) & valid).astype(np.uint8)
    
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE, MORPH_CLOSE))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_TOP] <= TOP_TOUCH_TOL:
            keep[labels == i] = 1
    
    out = mask_np.copy()
    out[(mask_np == 1) & (keep == 0)] = 0
    return out


def build_cache_for_degradation(df, deg_folder, out_dir, model, detector, theta_scan):
    """Build cache for a single degradation type."""
    ensure_dir(out_dir)
    deg_name = deg_folder.name
    
    processed = 0
    skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=deg_name, ncols=80):
        img_stem = str(row["img_stem"])
        
        # 原始坐标 (1920x1080)
        try:
            x1_org, y1_org = float(row["x1"]), float(row["y1"])
            x2_org, y2_org = float(row["x2"]), float(row["y2"])
        except:
            skipped += 1
            continue
        
        # 读取退化图像 (MU-SID 使用大写 .JPG)
        img_path = deg_folder / f"{img_stem}.JPG"
        if not img_path.exists():
            img_path = deg_folder / f"{img_stem}.jpg"
        if not img_path.exists():
            skipped += 1
            continue
        
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue
        
        h_orig, w_orig = bgr.shape[:2]
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Resize 到 UNet 尺寸 (1024x576)
        rgb_unet = cv2.resize(rgb0, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(rgb_unet.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # 2. UNet 推理
        with torch.no_grad():
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE == "cuda")):
                restored_t, seg_logits, _ = model(inp, None, True, True)
        
        restored_np = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
        mask_np = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 3. 后处理
        mask_pp = post_process_mask_top_connected(mask_np)
        
        # 4. 特征提取
        try:
            _, _, _, trad_sinos = detector.detect(restored_bgr)
        except:
            trad_sinos = []
        
        processed_stack = []
        for s in trad_sinos[:3]:
            processed_stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))
        
        # Seg -> Canny -> Radon
        edges = cv2.Canny((mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
        if EDGE_DILATE > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=EDGE_DILATE)
        seg_sino = detector._radon_gpu(edges, theta_scan)
        processed_stack.append(process_sinogram(seg_sino, RESIZE_H, RESIZE_W))
        
        combined_input = np.stack(processed_stack, axis=0).astype(np.float32)
        
        # 5. 计算 Label (坐标缩放到 1024x576)
        scale_x = UNET_IN_W / w_orig
        scale_y = UNET_IN_H / h_orig
        
        x1_s, y1_s = x1_org * scale_x, y1_org * scale_y
        x2_s, y2_s = x2_org * scale_x, y2_org * scale_y
        
        l_rho, l_theta = calculate_radon_label(x1_s, y1_s, x2_s, y2_s, UNET_IN_W, UNET_IN_H, RESIZE_H, RESIZE_W)
        label = np.array([l_rho, l_theta], dtype=np.float32)
        
        # 6. 保存
        np.save(
            os.path.join(out_dir, f"{img_stem}.npy"),
            {
                "input": combined_input,
                "label": label,
                "img_name": f"{img_stem}.jpg",
                "degradation": deg_name,
            },
        )
        processed += 1
    
    return processed, skipped


def main():
    print("=" * 60)
    print("Experiment 5: Build FusionCache for Degraded Images")
    print("=" * 60)
    
    # Load model
    print(f"\n[Device] {DEVICE}")
    print("[Load] UNet (RestorationGuidedHorizonNet)...")
    
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(RGHNET_CKPT, map_location=DEVICE, weights_only=False), strict=False)
    model.eval()
    
    print("[Load] Radon Transform (TextureSuppressedMuSCoWERT)...")
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)
    
    # Load ground truth
    print("[Load] GroundTruth_test.csv...")
    df = load_gt_csv()
    print(f"  -> {len(df)} test images")
    
    # Get degradation folders
    if not DEGRADED_IMG_DIR.exists():
        print(f"[Error] Degraded images not found: {DEGRADED_IMG_DIR}")
        print("Please run generate_degraded_images.py first")
        sys.exit(1)
    
    deg_folders = sorted([d for d in DEGRADED_IMG_DIR.iterdir() if d.is_dir()])
    
    if SELECTED_DEGRADATIONS:
        deg_folders = [d for d in deg_folders if d.name in SELECTED_DEGRADATIONS]
    
    print(f"\n[Process] {len(deg_folders)} degradation types:")
    for d in deg_folders:
        print(f"  - {d.name}")
    
    ensure_dir(CACHE_ROOT)
    
    # Build cache for each degradation
    total_processed = 0
    total_skipped = 0
    
    for deg_folder in deg_folders:
        out_dir = CACHE_ROOT / deg_folder.name
        processed, skipped = build_cache_for_degradation(
            df, deg_folder, str(out_dir), model, detector, theta_scan
        )
        total_processed += processed
        total_skipped += skipped
        print(f"  -> {deg_folder.name}: {processed} cached, {skipped} skipped")
    
    print("\n" + "=" * 60)
    print("[Done] FusionCache for degraded images saved to:")
    print(f"  {CACHE_ROOT}")
    print(f"\nTotal: {total_processed} cached, {total_skipped} skipped")
    print("=" * 60)


if __name__ == "__main__":
    main()
