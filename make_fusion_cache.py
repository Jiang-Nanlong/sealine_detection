# make_fusion_cache.py
import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.amp as amp

from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# =========================
# 全局配置（PyCharm 直接改这里）
# =========================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR  = r"Hashmani's Dataset/MU-SID"

# UNet 权重
RGHNET_CKPT = "rghnet_best_joint.pth"   # 你最终用于生成复原+分割的权重
DCE_WEIGHTS = "Epoch99.pth"

# 输出 cache 根目录（会自动创建 train/val/test 子目录）
SAVE_ROOT = r"Hashmani's Dataset/FusionCache_split"

# 固定划分目录
SPLIT_DIR = r"splits_musid"

# 生成哪些 split 的 cache（建议：先 train+val；最终一次性再生成 test）
SPLITS_TO_BUILD = ["train", "val", "test"]   # 你也可以改成 ["train", "test"]

# 输入输出尺寸
IMG_SIZE_UNET = 384
RESIZE_H = 2240
RESIZE_W = 180

# ===== 语义mask鲁棒后处理 & 边缘参数（你主要调这里做对比实验）=====
MORPH_CLOSE = 3         # 0 禁用；常用 0/3/5（太大可能把错误粘连）
TOP_TOUCH_TOL = 2       # “接触顶边”容忍像素
SKY_ID = 1              # sky=1, sea=0

CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE_ITERS = 1   # 0 不膨胀；1~2 通常够用
# ================================================================

# 性能/稳定性
USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# 若输出已存在是否跳过
SKIP_IF_EXISTS = True
# =========================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def resolve_image_path(img_dir: str, name_in_csv: str):
    """兼容 CSV 中带/不带后缀的情况"""
    base = os.path.join(img_dir, str(name_in_csv))
    candidates = [
        base,
        base + ".JPG",
        base + ".jpg",
        base + ".png",
        base + ".jpeg",
        base + ".JPEG",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def safe_load(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def calculate_radon_label(x1, y1, x2, y2, img_w, img_h, resize_h, resize_w):
    """计算归一化 rho/theta 标签 (0~1)，与 padding 逻辑对齐"""
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

    original_diag = np.sqrt(img_w**2 + img_h**2)
    rho_pixel_pos = rho + original_diag / 2.0

    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top
    label_rho = final_rho_idx / (resize_h - 1)

    return float(np.clip(label_rho, 0, 1)), float(np.clip(label_theta, 0, 1))


def process_sinogram(sino, target_h, target_w):
    """归一化并 padding/crop 到统一尺寸"""
    mi, ma = sino.min(), sino.max()
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

    return container.astype(np.float32)


def post_process_mask_top_connected_simple(mask_np: np.ndarray,
                                           sky_id: int = 1,
                                           morph_close: int = 3,
                                           top_touch_tol: int = 2) -> np.ndarray:
    """
    鲁棒 mask 清理（移植 eval/vis_stage_B 思路）：
      1) 只保留“接触顶边”的 sky 连通域（去掉海面 sky 假阳性岛）
      2) 可选：对保留天空做轻量 close（弥合裂缝）
    """
    if mask_np.ndim != 2:
        raise ValueError("mask_np must be HxW")

    sky = (mask_np == sky_id).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num_labels):
        y_top = stats[i, cv2.CC_STAT_TOP]
        if y_top <= top_touch_tol:
            keep[labels == i] = 1

    if morph_close and morph_close > 0:
        k = int(morph_close)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)

    out = np.zeros_like(mask_np, dtype=np.uint8)
    out[keep == 1] = sky_id
    return out


def load_split_indices(split_dir: str):
    """
    优先读取 train_indices/val_indices/test_indices.npy（你全局划分脚本生成的）
    兼容读取 train_idx/test_idx.npy（你改 CNN 脚本保存的）
    """
    paths_primary = {
        "train": os.path.join(split_dir, "train_indices.npy"),
        "val":   os.path.join(split_dir, "val_indices.npy"),
        "test":  os.path.join(split_dir, "test_indices.npy"),
    }
    paths_alt = {
        "train": os.path.join(split_dir, "train_idx.npy"),
        "test":  os.path.join(split_dir, "test_idx.npy"),
    }

    out = {}
    # primary
    if all(os.path.exists(p) for p in paths_primary.values()):
        out["train"] = np.load(paths_primary["train"]).astype(np.int64).tolist()
        out["val"]   = np.load(paths_primary["val"]).astype(np.int64).tolist()
        out["test"]  = np.load(paths_primary["test"]).astype(np.int64).tolist()
        return out

    # alt
    if os.path.exists(paths_alt["train"]) and os.path.exists(paths_alt["test"]):
        out["train"] = np.load(paths_alt["train"]).astype(np.int64).tolist()
        out["test"]  = np.load(paths_alt["test"]).astype(np.int64).tolist()
        # 没有 val 的情况下就不生成 val
        return out

    raise FileNotFoundError(
        "找不到 split 索引文件。请确认 SPLIT_DIR 下存在：\n"
        f"  {paths_primary['train']} / {paths_primary['val']} / {paths_primary['test']}\n"
        "或至少存在：\n"
        f"  {paths_alt['train']} / {paths_alt['test']}\n"
    )


def main():
    ensure_dir(SAVE_ROOT)

    print(f"[Device] {DEVICE}")
    print(f"[Params] MORPH_CLOSE={MORPH_CLOSE}, TOP_TOUCH_TOL={TOP_TOUCH_TOL}, "
          f"CANNY=({CANNY_LOW},{CANNY_HIGH}), EDGE_DILATE_ITERS={EDGE_DILATE_ITERS}")
    print(f"[Splits] SPLITS_TO_BUILD={SPLITS_TO_BUILD}")

    # 读取 GroundTruth
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, header=None)

    # 读取 split
    splits = load_split_indices(SPLIT_DIR)
    for k, v in splits.items():
        print(f"[Split] {k}: {len(v)}")

    # 加载 UNet
    if not os.path.exists(RGHNET_CKPT):
        raise FileNotFoundError(f"UNet checkpoint not found: {RGHNET_CKPT}")

    print(f"[Load] RG-HNet: {RGHNET_CKPT}")
    seg_model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    seg_model.load_state_dict(safe_load(RGHNET_CKPT, DEVICE), strict=False)
    seg_model.eval()

    # 传统 Radon 提取器（GPU Radon）
    print("[Load] Traditional Extractor (MuSCoWERT + GPU Radon)")
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0., 180., RESIZE_W, endpoint=False)

    # 保存 meta（方便你写实验记录）
    meta = {
        "csv_path": CSV_PATH,
        "img_dir": IMG_DIR,
        "save_root": SAVE_ROOT,
        "ckpt": RGHNET_CKPT,
        "dce_weights": DCE_WEIGHTS,
        "img_size_unet": IMG_SIZE_UNET,
        "resize_h": RESIZE_H,
        "resize_w": RESIZE_W,
        "mask_post": {
            "morph_close": MORPH_CLOSE,
            "top_touch_tol": TOP_TOUCH_TOL,
            "sky_id": SKY_ID,
            "canny_low": CANNY_LOW,
            "canny_high": CANNY_HIGH,
            "edge_dilate_iters": EDGE_DILATE_ITERS,
        },
        "splits_built": SPLITS_TO_BUILD,
        "use_amp": USE_AMP,
        "device": DEVICE,
    }
    with open(os.path.join(SAVE_ROOT, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 开始生成
    with torch.no_grad():
        for split_name in SPLITS_TO_BUILD:
            if split_name not in splits:
                print(f"[Skip] split '{split_name}' not found in SPLIT_DIR. Available: {list(splits.keys())}")
                continue

            out_dir = os.path.join(SAVE_ROOT, split_name)
            ensure_dir(out_dir)

            indices = splits[split_name]
            print(f"\n[Build] {split_name}: {len(indices)} samples -> {out_dir}")

            for idx in tqdm(indices, desc=f"Cache {split_name}"):
                out_path = os.path.join(out_dir, f"{idx}.npy")
                if SKIP_IF_EXISTS and os.path.exists(out_path):
                    continue

                if idx < 0 or idx >= len(df):
                    continue

                # 读 GT 行
                row = df.iloc[idx]
                img_name = str(row.iloc[0])
                try:
                    x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
                except Exception:
                    continue

                img_path = resolve_image_path(IMG_DIR, img_name)
                if img_path is None:
                    continue

                bgr = cv2.imread(img_path)
                if bgr is None:
                    continue
                h_img, w_img = bgr.shape[:2]

                # ====== UNet 前处理 ======
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (IMG_SIZE_UNET, IMG_SIZE_UNET), interpolation=cv2.INTER_AREA)
                inp = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                # ====== UNet 推理：复原 + 分割 ======
                with amp.autocast(device_type=DEVICE_TYPE, enabled=(USE_AMP and DEVICE_TYPE == "cuda")):
                    restored_t, seg_logits, _ = seg_model(inp, None)

                pred_mask = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

                # ====== A) 三通道：复原输出 -> 传统梯度+Radon ======
                restored_rgb = (restored_t[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                restored_rgb_full = cv2.resize(restored_rgb, (w_img, h_img), interpolation=cv2.INTER_AREA)
                restored_bgr_full = cv2.cvtColor(restored_rgb_full, cv2.COLOR_RGB2BGR)

                _, _, _, trad_sinograms = detector.detect(restored_bgr_full)

                processed_trad = []
                for s in trad_sinograms[:3]:
                    processed_trad.append(process_sinogram(s, RESIZE_H, RESIZE_W))
                while len(processed_trad) < 3:
                    processed_trad.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

                # ====== B) 第四通道：分割 -> 边界 -> Radon ======
                pred_mask_full = cv2.resize(pred_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

                # robust mask 清理（只保留顶边连通天空）
                pred_mask_full = post_process_mask_top_connected_simple(
                    pred_mask_full,
                    sky_id=SKY_ID,
                    morph_close=MORPH_CLOSE,
                    top_touch_tol=TOP_TOUCH_TOL
                )

                edges = cv2.Canny((pred_mask_full * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
                if EDGE_DILATE_ITERS and EDGE_DILATE_ITERS > 0:
                    kernel = np.ones((3, 3), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=int(EDGE_DILATE_ITERS))

                seg_sino_raw = detector._radon_gpu(edges, theta_scan)
                processed_seg = process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W)

                # ====== 融合 4 通道 ======
                combined_input = np.stack(processed_trad + [processed_seg], axis=0).astype(np.float32)

                # 标签
                l_rho, l_theta = calculate_radon_label(x1, y1, x2, y2, w_img, h_img, RESIZE_H, RESIZE_W)

                np.save(out_path, {
                    "input": combined_input,
                    "label": np.array([l_rho, l_theta], dtype=np.float32),
                    "idx": int(idx),
                    "img": os.path.basename(img_path),
                    "split": split_name,
                })

    print("\n[DONE] Fusion cache built.")
    print(f"Saved root: {os.path.abspath(SAVE_ROOT)}")
    print("Subfolders:", ", ".join([s for s in SPLITS_TO_BUILD if os.path.isdir(os.path.join(SAVE_ROOT, s))]))


if __name__ == "__main__":
    main()
