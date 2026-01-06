# -*- coding: utf-8 -*-
"""
make_fusion_cache.py (Letterbox + 1024)

生成 4 通道融合缓存：
  - 通道1-3：传统梯度 + Radon（在 UNet 复原图上做）
  - 通道4 ：语义分割(天空/海面)的“边界线” -> Radon

关键改动：
  1) UNet 输入使用 letterbox（等比缩放 + padding），IMG_SIZE_UNET=1024
  2) UNet 输出 restored/mask 先 unletterbox 回原图尺寸，再做传统特征与边界 Radon
  3) 分割 mask 做 robust 后处理：仅保留“触顶”的天空连通域（移植自 eval_stage_B）
  4) 输出目录自带 train/val/test 三个子目录（按 split_indices/*.npy）

注意：
  - 一旦改了 letterbox/分辨率，你必须重训 UNet、重做 cache、再重训 CNN。
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.amp as amp

from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT


# ============================
# 配置（按你工程路径改）
# ============================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"

SPLIT_DIR = r"Hashmani's Dataset/split_indices"  # train_indices.npy / val_indices.npy / test_indices.npy
SAVE_ROOT = r"Hashmani's Dataset/FusionCache_letterbox1024"  # 会自动生成 train/val/test

# 权重路径
RGHNET_CKPT = r"rghnet_best_joint.pth"
DCE_WEIGHTS = r"Epoch99.pth"

# UNet 输入尺寸（square letterbox）
IMG_SIZE_UNET = 1024

# sinogram 统一尺寸
RESIZE_H = 2240
RESIZE_W = 180

# -------- robust 后处理参数（全局变量，方便你在 PyCharm 里直接改） --------
MORPH_CLOSE = 3      # 0/3/5 常用；过大可能把海面假阳性连上天空
TOP_TOUCH_TOL = 0    # 天空连通域触顶容忍（0 表示必须接触最顶行）

# 边界提取参数
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1      # 0 不膨胀；1/2 轻微加粗边界，Radon 更稳

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def letterbox_rgb_u8(image_rgb_u8: np.ndarray, dst_size: int, pad_value: int = 0):
    """Keep aspect ratio resize to fit in dst_size x dst_size and pad."""
    h, w = image_rgb_u8.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.zeros((dst_size, dst_size, 3), dtype=np.uint8)
        meta = dict(scale=1.0, pad_left=0, pad_top=0, new_w=dst_size, new_h=dst_size, orig_w=w, orig_h=h)
        return canvas, meta

    scale = min(dst_size / float(w), dst_size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = max(1, min(dst_size, new_w))
    new_h = max(1, min(dst_size, new_h))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=interp)

    canvas = np.full((dst_size, dst_size, 3), pad_value, dtype=np.uint8)
    pad_left = (dst_size - new_w) // 2
    pad_top = (dst_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = dict(scale=scale, pad_left=pad_left, pad_top=pad_top, new_w=new_w, new_h=new_h, orig_w=w, orig_h=h)
    return canvas, meta


def unletterbox_rgb_u8(img_sq_rgb_u8: np.ndarray, meta: dict, out_w: int, out_h: int):
    """Crop padding area then resize back to (out_w,out_h)."""
    pl, pt = int(meta["pad_left"]), int(meta["pad_top"])
    nw, nh = int(meta["new_w"]), int(meta["new_h"])
    crop = img_sq_rgb_u8[pt:pt + nh, pl:pl + nw]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def unletterbox_mask_u8(mask_sq_u8: np.ndarray, meta: dict, out_w: int, out_h: int):
    pl, pt = int(meta["pad_left"]), int(meta["pad_top"])
    nw, nh = int(meta["new_w"]), int(meta["new_h"])
    crop = mask_sq_u8[pt:pt + nh, pl:pl + nw]
    if crop.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


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
    """计算归一化的 rho, theta 标签 (0~1)，逻辑需与你 CNN loader 一致。"""
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

    label_theta = np.degrees(theta_rad) / 180.0

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
    """只保留“接触图像顶边”的天空连通域，去掉海面上的 sky 假阳性岛。"""
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
    def _load(name):
        p = os.path.join(split_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split index file not found: {p}")
        arr = np.load(p)
        return [int(x) for x in arr.tolist()]

    return {
        "train": _load("train_indices.npy"),
        "val": _load("val_indices.npy"),
        "test": _load("test_indices.npy"),
    }


def _read_image_any_ext(img_dir: str, img_name: str):
    base = os.path.join(img_dir, str(img_name))
    cand = [base, base + ".JPG", base + ".jpg", base + ".png", base + ".jpeg"]
    for p in cand:
        if os.path.exists(p):
            return cv2.imread(p)
    return None


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

        # 1) UNet letterbox input
        rgb_sq, meta = letterbox_rgb_u8(rgb0, IMG_SIZE_UNET, pad_value=0)
        inp = torch.from_numpy(rgb_sq.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2) forward
        with torch.no_grad():
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                restored_t, seg_logits, _ = seg_model(inp, None)

        # 3) restored back to original size
        restored_sq_rgb = (restored_t[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        restored_orig_rgb = unletterbox_rgb_u8(restored_sq_rgb, meta, out_w=w_img, out_h=h_img)
        restored_orig_bgr = cv2.cvtColor(restored_orig_rgb, cv2.COLOR_RGB2BGR)

        # 4) mask back to original size + robust post-process
        pred_mask_sq = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # 0/1
        pred_mask_orig = unletterbox_mask_u8(pred_mask_sq, meta, out_w=w_img, out_h=h_img)
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

        # B) 语义边界通道（mask 边界 -> Radon）
        edges = cv2.Canny((pred_mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
        if EDGE_DILATE and EDGE_DILATE > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=int(EDGE_DILATE))

        seg_sino_raw = detector._radon_gpu(edges, theta_scan)
        processed_seg = process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W)

        combined_input = np.stack(processed_trad + [processed_seg], axis=0).astype(np.float32)

        # label
        l_rho, l_theta = calculate_radon_label(x1, y1, x2, y2, w_img, h_img, RESIZE_H, RESIZE_W)
        label = np.array([l_rho, l_theta], dtype=np.float32)

        np.save(os.path.join(out_dir, f"{row_idx}.npy"), {"input": combined_input, "label": label})


def main():
    ensure_dir(SAVE_ROOT)

    splits = _load_split_indices(SPLIT_DIR)
    for k in ("train", "val", "test"):
        print(f"[Split] {k}: {len(splits[k])}")

    print(f"[Load] RG-HNet: {RGHNET_CKPT}")
    seg_model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if not os.path.exists(RGHNET_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {RGHNET_CKPT}")
    state = torch.load(RGHNET_CKPT, map_location=DEVICE)
    seg_model.load_state_dict(state, strict=False)
    seg_model.eval()

    print("[Load] Traditional extractor...")
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    df = pd.read_csv(CSV_PATH, header=None)
    print(f"[Data] rows={len(df)}")

    for split_name in ("train", "val", "test"):
        build_cache_for_split(df, splits[split_name], os.path.join(SAVE_ROOT, split_name), seg_model, detector, theta_scan)

    print(f"Done! Cache saved to: {SAVE_ROOT}")


if __name__ == "__main__":
    main()
