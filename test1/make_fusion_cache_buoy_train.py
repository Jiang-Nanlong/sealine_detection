# -*- coding: utf-8 -*-
"""
Build FusionCache for Buoy training (Experiment 6: In-Domain Training).

This script generates FusionCache for Buoy train/val/test sets.
Uses MU-SID pretrained UNet weights for feature extraction.

Inputs:
  - test4/Buoy_GroundTruth.csv
  - test4/buoy_frames/
  - test1/splits_buoy/

Outputs:
  - test1/FusionCache_Buoy/{train,val,test}/<idx>.npy

Usage:
  python test1/make_fusion_cache_buoy_train.py
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

from unet_model import RestorationGuidedHorizonNet  # noqa: E402
from gradient_radon import TextureSuppressedMuSCoWERT  # noqa: E402

# ============================
# PyCharm 配置�?
# ============================
# 使用 test4 中已准备�?Buoy 数据
CSV_PATH = str(PROJECT_ROOT / "test4" / "Buoy_GroundTruth.csv")
IMG_DIR = str(PROJECT_ROOT / "test4" / "buoy_frames")
SPLIT_DIR = str(PROJECT_ROOT / "test1" / "splits_buoy")
SAVE_ROOT = str(PROJECT_ROOT / "test1" / "FusionCache_Buoy")

# �?使用 Buoy 本地训练�?UNet 权重 (In-Domain Training)
RGHNET_CKPT = str(PROJECT_ROOT / "test1" / "weights_buoy" / "buoy_rghnet_best_seg_c2.pth")
DCE_WEIGHTS = str(PROJECT_ROOT / "weights" / "Epoch99.pth")
# ============================

# Unified UNet input size
UNET_IN_W = 1024
UNET_IN_H = 576

# Sinogram unified size
RESIZE_H = 2240
RESIZE_W = 180

# Params
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_image(img_dir: str, img_name: str):
    """Read image from frames directory."""
    p = os.path.join(img_dir, img_name)
    if not os.path.exists(p):
        return None
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Normalize and pad/crop sinogram to target size."""
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
    """Compute normalized (rho, theta) label."""
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


def post_process_mask_top_connected(mask_np: np.ndarray) -> np.ndarray:
    """Keep connected sky components touching the top edge."""
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


def _load_split_indices(split_dir: str):
    """Load train/val/test split indices."""
    tr = np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64).tolist()
    va = np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64).tolist()
    te = np.load(os.path.join(split_dir, "test_indices.npy")).astype(np.int64).tolist()
    return {"train": tr, "val": va, "test": te}


def build_cache_for_split(df: pd.DataFrame, indices, out_dir: str, seg_model, detector, theta_scan):
    """Build fusion cache for a split."""
    ensure_dir(out_dir)
    split_name = os.path.basename(out_dir)
    print(f"[Split] {split_name}: {len(indices)} samples")

    if len(indices) == 0:
        print(f"  (empty split, skipping)")
        return

    for idx in tqdm(indices, ncols=80, desc=split_name):
        row = df.iloc[idx]
        img_name = str(row["img_name"])

        try:
            x1_org = float(row["x1"])
            y1_org = float(row["y1"])
            x2_org = float(row["x2"])
            y2_org = float(row["y2"])
        except Exception:
            continue

        bgr = _read_image(IMG_DIR, img_name)
        if bgr is None:
            continue

        h_orig, w_orig = bgr.shape[:2]
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1) Resize to UNet input size
        rgb_unet = cv2.resize(rgb0, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(rgb_unet.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2) UNet inference
        with torch.no_grad():
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE == "cuda")):
                restored_t, seg_logits, _ = seg_model(inp, None, True, True)

        restored_np = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
        mask_np = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # 3) Mask post-process
        mask_pp = post_process_mask_top_connected(mask_np)

        # 4) Feature extraction
        try:
            _, _, _, trad_sinos = detector.detect(restored_bgr)
        except Exception:
            trad_sinos = []

        processed_stack = []
        for s in trad_sinos[:3]:
            processed_stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

        edges = cv2.Canny((mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
        if EDGE_DILATE > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=EDGE_DILATE)

        seg_sino = detector._radon_gpu(edges, theta_scan)
        processed_stack.append(process_sinogram(seg_sino, RESIZE_H, RESIZE_W))

        combined_input = np.stack(processed_stack, axis=0).astype(np.float32)

        # 5) Compute label in UNet space
        scale_x = UNET_IN_W / float(w_orig)
        scale_y = UNET_IN_H / float(h_orig)

        x1_s, y1_s = x1_org * scale_x, y1_org * scale_y
        x2_s, y2_s = x2_org * scale_x, y2_org * scale_y

        l_rho, l_theta = calculate_radon_label(
            x1_s, y1_s, x2_s, y2_s,
            UNET_IN_W, UNET_IN_H,
            RESIZE_H, RESIZE_W,
        )
        label = np.array([l_rho, l_theta], dtype=np.float32)

        np.save(
            os.path.join(out_dir, f"{idx}.npy"),
            {
                "input": combined_input,
                "label": label,
                "img_name": img_name,
                "orig_w": w_orig,
                "orig_h": h_orig,
            },
        )


def main():
    ensure_dir(SAVE_ROOT)

    print("=" * 60)
    print("Build FusionCache for Buoy Training (Experiment 6)")
    print("=" * 60)

    # Check paths
    if not os.path.exists(CSV_PATH):
        print(f"[Error] CSV not found: {CSV_PATH}")
        print("  Please run test4/prepare_buoy_testset.py first.")
        return 1

    if not os.path.isdir(IMG_DIR):
        print(f"[Error] Image directory not found: {IMG_DIR}")
        return 1

    if not os.path.isdir(SPLIT_DIR):
        print(f"[Error] Split directory not found: {SPLIT_DIR}")
        print("  Please run test1/prepare_buoy_trainset.py first.")
        return 1

    splits = _load_split_indices(SPLIT_DIR)

    print(f"[Device] {DEVICE}")
    print(f"[CSV]    {CSV_PATH}")
    print(f"[IMG]    {IMG_DIR}")
    print(f"[SPLIT]  {SPLIT_DIR}")
    print(f"[OUT]    {SAVE_ROOT}")

    # Check weights
    if not os.path.exists(RGHNET_CKPT):
        print(f"[Error] RGHNet weights not found: {RGHNET_CKPT}")
        return 1
    if not os.path.exists(DCE_WEIGHTS):
        print(f"[Error] DCE weights not found: {DCE_WEIGHTS}")
        return 1

    print(f"[Load] RGHNet ckpt: {RGHNET_CKPT}")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)

    state = torch.load(RGHNET_CKPT, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"[CSV] {len(df)} total samples")

    for split in ["train", "val", "test"]:
        build_cache_for_split(df, splits[split], os.path.join(SAVE_ROOT, split), model, detector, theta_scan)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
