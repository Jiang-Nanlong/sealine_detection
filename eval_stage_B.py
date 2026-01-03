# eval_stage_B.py (upgraded)
# - Adds horizon MAE that matches your new visualization logic:
#   * (optional) small morphological opening
#   * keep ONLY sky components connected to the top border
#   * then extract horizon per-column
# - Reports BOTH:
#   * Horizon MAE (raw)  : from raw argmax mask
#   * Horizon MAE (post) : after post-processing (recommended for system-level evaluation)

import os
import glob
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import cv2
from tqdm import tqdm

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import HorizonImageDataset


# ====== 你按自己工程改这里 ======
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
CKPT = "rghnet_stage_b.pth"          # e.g. rghnet_best_seg.pth / rghnet_best_joint.pth / rghnet_stage_c2.pth
DCE_WEIGHTS = "Epoch99.pth"
IMG_SIZE = 384

BATCH_SIZE = 8
VAL_RATIO = 0.2
SEED = 123

# --- Horizon post-process params (推荐默认) ---
POST_OPEN_KSIZE = 3      # 0=关闭；3/5 适合去掉细条噪声
POST_CLOSE_KSIZE = 0     # 默认关闭；天空被严重切碎才考虑 3
POST_TOP_MARGIN = 2      # 允许连到顶边的容差（像素）
POST_MIN_AREA = 50       # 小于该面积的天空连通域直接丢弃

# --- Horizon smoothing params (只影响“线”，不改变mask) ---
SMOOTH_MEDIAN_K = 21     # 1=关闭；建议 11~31
SMOOTH_MAX_JUMP = 20     # 单列最大允许跳变（像素）
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path: str, map_location: str):
    """torch.load without the weights_only warning (with backward-compat)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def count_coverage(csv_path, img_dir):
    import pandas as pd
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].astype(str).tolist()

    exts = ["", ".JPG", ".jpg", ".png", ".jpeg"]
    hit = 0
    miss = []
    for n in names:
        base = os.path.join(img_dir, n)
        ok = any(os.path.exists(base + e) for e in exts)
        if ok:
            hit += 1
        else:
            miss.append(n)

    all_imgs = []
    for p in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
        all_imgs += glob.glob(os.path.join(img_dir, p))

    return len(names), hit, len(miss), len(all_imgs), miss[:10]


def compute_metrics(seg_logits, mask, ignore_index=255):
    """
    seg_logits: [B,2,H,W]
    mask: [B,H,W] long, {0,1,255}
    """
    pred = seg_logits.argmax(dim=1)  # [B,H,W]
    valid = (mask != ignore_index)

    tp = ((pred == 1) & (mask == 1) & valid).sum().item()
    fp = ((pred == 1) & (mask == 0) & valid).sum().item()
    fn = ((pred == 0) & (mask == 1) & valid).sum().item()
    tn = ((pred == 0) & (mask == 0) & valid).sum().item()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": iou, "dice": dice, "acc": acc, "prec": prec, "rec": rec
    }, pred


def post_process_pred_mask(
    pred_np: np.ndarray,
    valid_np: np.ndarray,
    sky_id: int = 1,
    open_ksize: int = POST_OPEN_KSIZE,
    close_ksize: int = POST_CLOSE_KSIZE,
    top_margin: int = POST_TOP_MARGIN,
    min_area: int = POST_MIN_AREA,
) -> np.ndarray:
    """
    Keep ONLY sky components that are connected to the image top border.

    pred_np: [H,W] {0,1}
    valid_np: [H,W] bool (gt != ignore_index)
    """
    H, W = pred_np.shape

    sky = ((pred_np == sky_id) & valid_np).astype(np.uint8)

    # A) opening: remove thin noise (optional)
    if open_ksize and open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, k)

    # B) connected components: keep those touching top
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        y_top = stats[i, cv2.CC_STAT_TOP]
        if area < min_area:
            continue
        if y_top <= top_margin:
            keep[labels == i] = 1

    # C) small closing for holes (optional)
    if close_ksize and close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k)

    out = pred_np.copy()
    out[(pred_np == sky_id) & (keep == 0)] = 0
    return out


def _fill_nan_nearest(y: np.ndarray) -> np.ndarray:
    """Fill NaNs by nearest valid value (for smoothing only)."""
    y2 = y.copy()
    n = len(y2)
    if np.all(np.isnan(y2)):
        return y2

    # forward fill
    last = np.nan
    for i in range(n):
        if np.isnan(y2[i]):
            y2[i] = last
        else:
            last = y2[i]

    # backward fill
    last = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(y2[i]):
            y2[i] = last
        else:
            last = y2[i]

    return y2


def _median_filter_1d(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return y
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i + k])
    return out


def horizon_y_curve(mask_hw: np.ndarray, valid_hw: np.ndarray, sky_id: int = 1) -> Optional[np.ndarray]:
    """Return per-column horizon y (float array with NaNs for invalid columns)."""
    H, W = mask_hw.shape
    col_valid = valid_hw.any(axis=0)

    y = np.full((W,), np.nan, dtype=np.float32)

    for x in range(W):
        if not col_valid[x]:
            continue
        col = mask_hw[:, x]
        vcol = valid_hw[:, x]
        sky = (col == sky_id) & vcol
        if sky.any():
            y[x] = float(np.where(sky)[0].max())

    good = np.where(~np.isnan(y))[0]
    if good.size < 2:
        return None

    # interpolate missing columns
    x_all = np.arange(W)
    y_interp = np.interp(x_all, good, y[good]).astype(np.float32)
    y_interp[~col_valid] = np.nan

    return y_interp


def smooth_horizon_y(y: np.ndarray, median_k: int = SMOOTH_MEDIAN_K, max_jump: int = SMOOTH_MAX_JUMP) -> np.ndarray:
    """Median smoothing + max jump clamp. y can contain NaNs."""
    if median_k <= 1 and (max_jump is None or max_jump <= 0):
        return y

    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return y

    y_work = _fill_nan_nearest(y)

    if median_k and median_k > 1:
        y_work = _median_filter_1d(y_work, median_k)

    if max_jump and max_jump > 0:
        for i in range(1, len(y_work)):
            if np.isnan(y_work[i]) or np.isnan(y_work[i - 1]):
                continue
            if abs(float(y_work[i]) - float(y_work[i - 1])) > max_jump:
                y_work[i] = y_work[i - 1]

    # restore NaNs in invalid columns
    y_work[~ok] = np.nan
    return y_work


def horizon_mae_from_masks(
    pred_hw: np.ndarray,
    gt_hw: np.ndarray,
    ignore_index: int = 255,
    postprocess: bool = False,
    smooth: bool = False,
) -> Optional[float]:
    """Compute MAE between horizon curves derived from pred/gt masks."""
    valid_hw = (gt_hw != ignore_index)

    pred_use = pred_hw
    if postprocess:
        pred_use = post_process_pred_mask(pred_hw, valid_hw)

    y_gt = horizon_y_curve(gt_hw, valid_hw, sky_id=1)
    y_pd = horizon_y_curve(pred_use, valid_hw, sky_id=1)
    if y_gt is None or y_pd is None:
        return None

    if smooth:
        y_gt = smooth_horizon_y(y_gt)
        y_pd = smooth_horizon_y(y_pd)

    ok = ~np.isnan(y_gt) & ~np.isnan(y_pd)
    if ok.sum() == 0:
        return None

    return float(np.mean(np.abs(y_pd[ok] - y_gt[ok])))


def overlay_mask(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    gt_sky = (gt == 1)
    pd_sky = (pred == 1)
    ign = (gt == 255)

    out[gt_sky] = (out[gt_sky] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    out[pd_sky] = (out[pd_sky] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    out[gt_sky & pd_sky] = (out[gt_sky & pd_sky] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    out[ign] = (out[ign] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)
    return out


def tensor_to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def draw_polyline(rgb: np.ndarray, pts, color_rgb, thickness=2):
    if len(pts) < 2:
        return rgb
    img = rgb.copy()
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color_bgr, thickness, lineType=cv2.LINE_AA)
    return img


def horizon_polyline_from_curve(y: np.ndarray):
    if y is None:
        return []
    W = len(y)
    pts = []
    for x in range(W):
        if np.isnan(y[x]):
            continue
        pts.append((int(x), int(y[x])))
    return pts


def run_model_get_seg_logits(model, img_bchw):
    """Compatible with both old/new forward signatures."""
    try:
        out = model(img_bchw, None, enable_restoration=True, enable_segmentation=True)
    except TypeError:
        out = model(img_bchw, None)

    if isinstance(out, (list, tuple)):
        return out[1]
    raise RuntimeError("Model output unexpected.")


def main():
    # 1) 覆盖率检查
    n_csv, hit, miss_n, n_dir, miss_example = count_coverage(CSV_PATH, IMG_DIR)
    print(f"[Coverage] CSV rows={n_csv}, matched images={hit}, missing={miss_n}, files_in_dir={n_dir}")
    if miss_n > 0:
        print(f"[Coverage] first missing names: {miss_example}")

    # 2) 划分 val
    ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode='segmentation')
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(ds, [n_train, n_val], generator=g)
    print(f"[Split] val={len(val_ds)}")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    # 3) 模型加载
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    state = safe_load(CKPT, DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    os.makedirs("eval_vis", exist_ok=True)

    # 4) Eval
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    horizon_raw_list: List[float] = []
    horizon_post_list: List[float] = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(val_loader, desc="Eval Stage B")):
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                seg_logits = run_model_get_seg_logits(model, img)

            m, pred = compute_metrics(seg_logits, mask)
            for k in totals:
                totals[k] += m[k]

            pred_np = pred.detach().cpu().numpy().astype(np.uint8)
            mask_np = mask.detach().cpu().numpy().astype(np.int32)

            # Horizon MAE (raw/post)
            for b in range(pred_np.shape[0]):
                mae_raw = horizon_mae_from_masks(
                    pred_np[b], mask_np[b],
                    postprocess=False,
                    smooth=False,
                )
                if mae_raw is not None:
                    horizon_raw_list.append(mae_raw)

                mae_post = horizon_mae_from_masks(
                    pred_np[b], mask_np[b],
                    postprocess=True,
                    smooth=True,
                )
                if mae_post is not None:
                    horizon_post_list.append(mae_post)

            # 保存少量可视化（raw vs post）
            if i < 10:
                for b in range(min(pred_np.shape[0], 4)):
                    rgb = tensor_to_uint8_rgb(img[b])
                    gt_hw = mask_np[b]
                    raw_hw = pred_np[b]
                    valid_hw = (gt_hw != 255)
                    post_hw = post_process_pred_mask(raw_hw, valid_hw)

                    # build horizon curves for drawing (match MAE post)
                    y_gt = horizon_y_curve(gt_hw, valid_hw)
                    y_pd_raw = horizon_y_curve(raw_hw, valid_hw)
                    y_pd_post = horizon_y_curve(post_hw, valid_hw)

                    y_gt_s = smooth_horizon_y(y_gt) if y_gt is not None else None
                    y_pd_post_s = smooth_horizon_y(y_pd_post) if y_pd_post is not None else None

                    over_raw = overlay_mask(rgb, raw_hw, gt_hw)
                    over_post = overlay_mask(rgb, post_hw, gt_hw)

                    # draw GT + Pred(post) on post overlay
                    if y_gt_s is not None:
                        over_post = draw_polyline(over_post, horizon_polyline_from_curve(y_gt_s), (0, 255, 0), 2)
                    if y_pd_post_s is not None:
                        over_post = draw_polyline(over_post, horizon_polyline_from_curve(y_pd_post_s), (255, 0, 0), 2)

                    cv2.imwrite(
                        os.path.join("eval_vis", f"batch{i}_idx{b}_raw.png"),
                        cv2.cvtColor(over_raw, cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(
                        os.path.join("eval_vis", f"batch{i}_idx{b}_post.png"),
                        cv2.cvtColor(over_post, cv2.COLOR_RGB2BGR)
                    )

    # 5) 汇总
    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    h_raw = float(np.mean(horizon_raw_list)) if horizon_raw_list else None
    h_post = float(np.mean(horizon_post_list)) if horizon_post_list else None

    print("\n===== Stage B Validation =====")
    print(f"PixelAcc={acc:.4f}  IoU(sky=1)={iou:.4f}  Dice={dice:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")

    if h_raw is not None:
        print(f"Horizon MAE raw  (pixels) = {h_raw:.2f}  (no post-process)")
    else:
        print("Horizon MAE raw  (pixels) = N/A")

    if h_post is not None:
        print(f"Horizon MAE post (pixels) = {h_post:.2f}  (top-connected sky + smoothed)")
    else:
        print("Horizon MAE post (pixels) = N/A")

    print("Saved visualization to ./eval_vis (raw/post overlays)")


if __name__ == "__main__":
    main()
