# -*- coding: utf-8 -*-
"""
make_fusion_cache.py (Rect 1024x576, No Padding)

生成 4 通道融合缓存（供 OfflineHorizonDataset / CNN 训练使用）：
  - 通道1-3：传统梯度 + Radon（在 UNet 复原图 restored 上做）
  - 通道4 ：分割 mask 的“边界线” -> Radon（在 post_process 后的 mask 上做）

本版本关键点：
1) UNet 输入使用 1024x576 直接 resize（适配所有 1920x1080，完全无 padding）
2) UNet 输出 restored/mask resize 回原图尺寸，再做传统特征与边界 Radon
3) mask 做 robust 后处理：只保留“触顶”的天空连通域（与你 test.py / cache 一致）
4) 输出目录包含 train/val/test 子目录（按 splits_musid/*.npy）

注意：
- 你改了 UNet 输入尺寸/数据分布后：必须重训 UNet、重做 cache、再重训 CNN。
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torch.amp as amp

from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT


# ============================
# 配置（按你工程路径改）
# ============================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"

# 固定划分（与 train_unet.py 一致）
SPLIT_DIR = r"splits_musid"  # train_indices.npy / val_indices.npy / test_indices.npy

# 输出缓存目录（建议新建，不要覆盖旧 letterbox cache）
SAVE_ROOT = r"Hashmani's Dataset/FusionCache_1024x576"  # 会自动生成 train/val/test

# 权重路径（用你训练出来的 best）
RGHNET_CKPT = r"rghnet_best_joint.pth"
DCE_WEIGHTS = r"Epoch99.pth"

# UNet 输入尺寸（无 padding，适配 1920x1080）
UNET_IN_W = 1024
UNET_IN_H = 576

# sinogram 统一尺寸
RESIZE_H = 2240
RESIZE_W = 180

# -------- robust 后处理参数（必须与 test.py 一致） --------
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0

# 边界提取参数（必须与 test.py 一致）
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_image_any_ext(img_dir: str, img_name: str):
    """
    GroundTruth.csv 里的 img_name 可能不带后缀，这里自动尝试常见后缀。
    """
    candidates = []
    base = os.path.join(img_dir, img_name)
    candidates.append(base)
    candidates.append(base + ".jpg")
    candidates.append(base + ".JPG")
    candidates.append(base + ".jpeg")
    candidates.append(base + ".JPEG")
    candidates.append(base + ".png")
    candidates.append(base + ".PNG")

    for p in candidates:
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                return img
    return None


def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """归一化并 padding/crop 到统一尺寸 (H,W)."""
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
    """
    计算归一化的 rho, theta 标签 (0~1)
    逻辑需与你 test.py 的 draw_line_from_rho_theta() 完全一致。
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1
    line_angle = np.arctan2(dy, dx)
    theta_rad = line_angle - np.pi / 2
    while theta_rad < 0:
        theta_rad += np.pi
    while theta_rad >= np.pi:
        theta_rad -= np.pi
    theta_deg = np.rad2deg(theta_rad)
    label_theta = theta_deg / 180.0

    # rho: distance from center to line
    a = dy
    b = -dx
    c = dx * y1 - dy * x1
    rho = -(a * cx + b * cy + c) / (np.sqrt(a * a + b * b) + 1e-12)

    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    rho_pixel_pos = rho + original_diag / 2.0
    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top
    label_rho = final_rho_idx / (resize_h - 1)

    return float(np.clip(label_rho, 0, 1)), float(np.clip(label_theta, 0, 1))


def post_process_mask_top_connected(mask_np: np.ndarray,
                                    sky_id: int = 1,
                                    ignore_id: int = 255,
                                    morph_close: int = MORPH_CLOSE,
                                    top_touch_tol: int = TOP_TOUCH_TOL) -> np.ndarray:
    """
    只保留“接触图像顶边”的天空连通域，去掉海面上的 sky 假阳性岛。
    与 test.py / cache 保持一致。
    mask_np 期望值：0/1（可包含 255 ignore）
    """
    valid = (mask_np != ignore_id)
    sky = ((mask_np == sky_id) & valid).astype(np.uint8)

    if morph_close and morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)

    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num_labels):
        y_top = stats[i, cv2.CC_STAT_TOP]
        if y_top <= top_touch_tol:
            keep[labels == i] = 1

    out = mask_np.copy()
    out[(mask_np == sky_id) & (keep == 0)] = 0
    return out


def _load_split_indices(split_dir: str):
    """
    读取 splits_musid 的 train/val/test indices（行号索引）
    """
    tr = np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64).tolist()
    va = np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64).tolist()
    te = np.load(os.path.join(split_dir, "test_indices.npy")).astype(np.int64).tolist()
    return {"train": tr, "val": va, "test": te}


def build_cache_for_split(df: pd.DataFrame,
                          indices: list,
                          out_dir: str,
                          seg_model,
                          detector,
                          theta_scan: np.ndarray):
    ensure_dir(out_dir)

    for row_idx in tqdm(indices, desc=f"cache->{os.path.basename(out_dir)}", ncols=90):
        row = df.iloc[row_idx]
        img_name = str(row.iloc[0])

        try:
            x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        except Exception:
            continue

        bgr = _read_image_any_ext(IMG_DIR, img_name)
        if bgr is None:
            continue

        h_img, w_img = bgr.shape[:2]
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1) UNet input: 1024x576 resize (NO padding)
        rgb_unet = cv2.resize(rgb0, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(rgb_unet.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2) forward (统一 pipeline：restoration + segmentation)
        with torch.no_grad():
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                restored_t, seg_logits, _ = seg_model(inp, None, enable_restoration=True, enable_segmentation=True)

        # 3) restored/mask resize back to original
        restored_unet_rgb = (
            restored_t[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().astype(np.uint8)  # (576,1024,3)

        pred_mask_unet = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # (576,1024)

        restored_orig_rgb = cv2.resize(restored_unet_rgb, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        pred_mask_orig = cv2.resize(pred_mask_unet, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        restored_orig_bgr = cv2.cvtColor(restored_orig_rgb, cv2.COLOR_RGB2BGR)

        # 4) mask robust post-process (MUST match test.py)
        pred_mask_pp = post_process_mask_top_connected(pred_mask_orig)

        # A) 传统三通道（在复原图上）
        try:
            _, _, _, trad_sinos = detector.detect(restored_orig_bgr)
        except Exception:
            trad_sinos = []

        processed_trad = []
        for s in trad_sinos[:3]:
            processed_trad.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_trad) < 3:
            processed_trad.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

        # B) 分割边界通道（Canny -> Radon），与 test.py 一致
        edges = cv2.Canny((pred_mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
        if EDGE_DILATE and EDGE_DILATE > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=int(EDGE_DILATE))

        seg_sino_raw = detector._radon_gpu(edges, theta_scan)
        processed_seg = process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W)

        combined_input = np.stack(processed_trad + [processed_seg], axis=0).astype(np.float32)

        # label (rho, theta)
        l_rho, l_theta = calculate_radon_label(x1, y1, x2, y2, w_img, h_img, RESIZE_H, RESIZE_W)
        label = np.array([l_rho, l_theta], dtype=np.float32)

        # 保存为 dict（与 dataset_loader_offline.py 兼容）
        np.save(os.path.join(out_dir, f"{row_idx}.npy"), {"input": combined_input, "label": label})


def main():
    ensure_dir(SAVE_ROOT)

    # splits
    splits = _load_split_indices(SPLIT_DIR)
    for k in ("train", "val", "test"):
        print(f"[Split] {k}: {len(splits[k])}")

    # load UNet
    print("[Load] UNet model...")
    seg_model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if not os.path.exists(RGHNET_CKPT):
        raise FileNotFoundError(f"UNet ckpt not found: {RGHNET_CKPT}")
    seg_model.load_state_dict(torch.load(RGHNET_CKPT, map_location=DEVICE), strict=False)
    seg_model.eval()

    # load radon extractor
    print("[Load] Traditional extractor...")
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    # data
    df = pd.read_csv(CSV_PATH, header=None)
    print(f"[Data] rows={len(df)}")

    # build cache
    for split_name in ("train", "val", "test"):
        out_dir = os.path.join(SAVE_ROOT, split_name)
        build_cache_for_split(df, splits[split_name], out_dir, seg_model, detector, theta_scan)

    print(f"Done! Cache saved to: {SAVE_ROOT}")


if __name__ == "__main__":
    main()
