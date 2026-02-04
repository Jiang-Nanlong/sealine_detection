# -*- coding: utf-8 -*-
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
# 配置
# ============================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
SPLIT_DIR = r"splits_musid"
SAVE_ROOT = r"Hashmani's Dataset/FusionCache_new_1024x576"

# 权重路径
RGHNET_CKPT = r"weights_new/rghnet_best_c2.pth"
DCE_WEIGHTS = r"weights_new/Epoch99.pth"

# 统一尺寸
UNET_IN_W = 1024
UNET_IN_H = 576

# sinogram 统一尺寸
RESIZE_H = 2240
RESIZE_W = 180

# 参数
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_image_any_ext(img_dir: str, img_stem_or_name: str):
    """Read image by stem or filename.

    MU-SID GroundTruth.csv stores the *stem* (e.g., "DSC_0051_9") without extension,
    while the actual file is typically ".JPG".

    Returns:
      (bgr, resolved_filename)
        - bgr: cv2 image (BGR)
        - resolved_filename: basename with extension (e.g., "DSC_0051_9.JPG")
    """
    name = str(img_stem_or_name)
    lower = name.lower()

    # If the caller already provides an extension, try it first.
    candidates = [name] if (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")) else []

    # Common extensions (MU-SID images are often .JPG)
    candidates += [name + ".JPG", name + ".jpg", name + ".jpeg", name + ".png"]

    for fn in candidates:
        p = os.path.join(img_dir, fn)
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is not None:
                return im, os.path.basename(fn)
    return None, None


def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
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
    # 计算中心偏移
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1

    # 角度计算
    line_angle = np.arctan2(dy, dx)
    theta_rad = line_angle - np.pi / 2
    while theta_rad < 0: theta_rad += np.pi
    while theta_rad >= np.pi: theta_rad -= np.pi

    # Rho 计算
    # 公式：x * cos + y * sin = rho
    # 使用中点计算
    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)

    # 映射到 Pixel Index
    # Radon 变换后的 sinogram 高度对应图像对角线长度
    # 但我们统一 Pad 到了 RESIZE_H (2240)
    # 所以要基于 RESIZE_H 进行归一化

    # 图像的物理对角线长度
    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)

    # rho = 0 对应中心
    rho_pixel_pos = rho + original_diag / 2.0

    # 加上 padding 的偏移量
    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top

    label_rho = final_rho_idx / (resize_h - 1)
    label_theta = np.rad2deg(theta_rad) / 180.0

    return float(np.clip(label_rho, 0, 1)), float(np.clip(label_theta, 0, 1))


def post_process_mask_top_connected(mask_np):
    # 简化的连通域处理
    valid = (mask_np != 255)
    sky = ((mask_np == 1) & valid).astype(np.uint8)
    # 闭运算连接断裂
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE, MORPH_CLOSE))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    # 连通域分析
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_TOP] <= TOP_TOUCH_TOL:
            keep[labels == i] = 1
    out = mask_np.copy()
    out[(mask_np == 1) & (keep == 0)] = 0
    return out


def _load_split_indices(split_dir):
    tr = np.load(os.path.join(split_dir, "train_indices.npy")).astype(np.int64).tolist()
    va = np.load(os.path.join(split_dir, "val_indices.npy")).astype(np.int64).tolist()
    te = np.load(os.path.join(split_dir, "test_indices.npy")).astype(np.int64).tolist()
    return {"train": tr, "val": va, "test": te}


def build_cache_for_split(df, indices, out_dir, seg_model, detector, theta_scan):
    ensure_dir(out_dir)
    print(f"Processing {os.path.basename(out_dir)}: {len(indices)}")

    for idx in tqdm(indices, ncols=80):
        row = df.iloc[idx]
        img_name = str(row.iloc[0])
        try:
            # 原始坐标 (1920x1080)
            x1_org, y1_org = float(row.iloc[1]), float(row.iloc[2])
            x2_org, y2_org = float(row.iloc[3]), float(row.iloc[4])
        except:
            continue

        bgr, img_filename = _read_image_any_ext(IMG_DIR, img_name)
        if bgr is None:
            continue

        h_orig, w_orig = bgr.shape[:2]
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Resize 到 UNet 尺寸 (1024x576)
        rgb_unet = cv2.resize(rgb0, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(rgb_unet.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2. UNet 推理 (C2)
        with torch.no_grad():
            # NOTE: use the actual device type to avoid autocast warnings/errors on CPU.
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE == "cuda")):
                restored_t, seg_logits, _ = seg_model(inp, None, True, True)

        restored_np = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
        mask_np = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # 3. 后处理
        mask_pp = post_process_mask_top_connected(mask_np)

        # 4. 特征提取 (基于 1024x576 的图像)
        try:
            _, _, _, trad_sinos = detector.detect(restored_bgr)
        except:
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

        # 5. [核心修正] 计算 Label 时，必须将坐标缩放到 1024x576
        scale_x = UNET_IN_W / w_orig
        scale_y = UNET_IN_H / h_orig

        x1_s, y1_s = x1_org * scale_x, y1_org * scale_y
        x2_s, y2_s = x2_org * scale_x, y2_org * scale_y

        # 传入 UNet 的尺寸，而不是原图尺寸
        l_rho, l_theta = calculate_radon_label(x1_s, y1_s, x2_s, y2_s, UNET_IN_W, UNET_IN_H, RESIZE_H, RESIZE_W)
        label = np.array([l_rho, l_theta], dtype=np.float32)

        # IMPORTANT: persist the original image filename so downstream scripts
        # (e.g., evaluate_full_pipeline.py) can reliably load the corresponding image.
        # This avoids the brittle fallback "<idx>.jpg".
        np.save(
            os.path.join(out_dir, f"{idx}.npy"),
            {"input": combined_input, "label": label,
             "img_name": str(img_filename) if img_filename else str(img_name) + ".JPG"},
        )


def main():
    ensure_dir(SAVE_ROOT)
    splits = _load_split_indices(SPLIT_DIR)

    print(f"[Load] Model: {RGHNET_CKPT}")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(RGHNET_CKPT, map_location=DEVICE), strict=False)
    model.eval()

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)
    df = pd.read_csv(CSV_PATH, header=None)

    for split in ["train", "val", "test"]:
        build_cache_for_split(df, splits[split], os.path.join(SAVE_ROOT, split), model, detector, theta_scan)


if __name__ == "__main__":
    main()