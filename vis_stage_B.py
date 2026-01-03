# vis_stage_B.py
import os
import random
import numpy as np
import cv2
import torch
import torch.amp as amp
from torch.utils.data import random_split

from unet_model import RestorationGuidedHorizonNet
from dataset_loader import HorizonImageDataset

# ===== 你按自己路径改这里 =====
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR  = r"Hashmani's Dataset/MU-SID"
CKPT     = "rghnet_stage_b.pth"   # 也可以换成 rghnet_best_seg.pth / rghnet_best_joint.pth
DCE_WEIGHTS = "Epoch99.pth"
IMG_SIZE = 384

OUT_DIR  = "seg_vis_ransac"
NUM_SAMPLES = 30
VAL_RATIO = 0.2
SEED = 123
# ============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def mask_to_vis(mask: np.ndarray) -> np.ndarray:
    """
    0=sea -> black, 1=sky -> white, 255=ignore -> gray
    """
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 0] = (0, 0, 0)
    out[mask == 1] = (255, 255, 255)
    out[mask == 255] = (160, 160, 160)
    return out


def keep_top_connected_sky(binary_sky: np.ndarray, top_eps: int = 2) -> np.ndarray:
    """
    binary_sky: 0/1
    只保留“接触到图像顶边”的 sky 连通域
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_sky.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(binary_sky, dtype=np.uint8)
    for i in range(1, num_labels):
        y_top = stats[i, cv2.CC_STAT_TOP]
        if y_top <= top_eps:
            kept[labels == i] = 1
    return kept


def post_process_mask(mask_np: np.ndarray,
                      morph_close_ksize: int = 5,
                      top_eps: int = 2) -> np.ndarray:
    """
    关键思路：
    1) 先 top-connected 过滤（避免 close 把海面白块粘到天上）
    2) 再对“保留下来的天空”做轻量 close，弥合雨线裂缝
    """
    H, W = mask_np.shape
    binary_sky = (mask_np == 1).astype(np.uint8)

    # A) 先保留 top-connected sky
    top_sky = keep_top_connected_sky(binary_sky, top_eps=top_eps)

    # B) 再对 top_sky 做轻量 close（可选）
    if morph_close_ksize and morph_close_ksize > 0:
        k = int(morph_close_ksize)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        top_sky = cv2.morphologyEx(top_sky, cv2.MORPH_CLOSE, kernel)

    # 重建 mask：sky=1 其他=0（忽略255通常GT才有，pred里一般没有）
    out = np.zeros((H, W), dtype=np.uint8)
    out[top_sky == 1] = 1
    return out


def overlay_mask(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    颜色约定（半透明）：
    GT sky: green
    Pred sky: red
    overlap: yellow
    ignore: white-ish
    """
    out = rgb.copy()
    gt_sky = (gt == 1)
    pd_sky = (pred == 1)
    ign = (gt == 255)

    out[gt_sky] = (out[gt_sky] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    out[pd_sky] = (out[pd_sky] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    out[gt_sky & pd_sky] = (out[gt_sky & pd_sky] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    out[ign] = (out[ign] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)
    return out


def horizon_points_from_mask(mask: np.ndarray, ignore_index: int = 255):
    """
    逐列抽取“最靠下的 sky 像素”作为候选点 (x, y)
    """
    H, W = mask.shape
    xs, ys = [], []
    for x in range(W):
        col = mask[:, x]
        valid = (col != ignore_index)
        sky_idx = np.where((col == 1) & valid)[0]
        if sky_idx.size == 0:
            continue
        ys.append(int(sky_idx.max()))
        xs.append(int(x))
    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


def ransac_fit_line(xs: np.ndarray,
                    ys: np.ndarray,
                    n_iter: int = 300,
                    thresh: float = 2.5,
                    min_inliers: int = 80,
                    seed: int = 0):
    """
    拟合 y = a*x + b
    返回 (a, b, inlier_mask)；失败则返回 None
    """
    if xs.size < 2:
        return None
    rng = np.random.default_rng(seed)

    best_inliers = None
    best_cnt = 0
    N = xs.size

    for _ in range(n_iter):
        i1, i2 = rng.choice(N, size=2, replace=False)
        x1, x2 = xs[i1], xs[i2]
        if x1 == x2:
            continue
        y1, y2 = ys[i1], ys[i2]
        a = (y2 - y1) / float(x2 - x1)
        b = y1 - a * x1

        y_pred = a * xs + b
        resid = np.abs(ys - y_pred)
        inliers = resid < thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers

    if best_inliers is None or best_cnt < max(min_inliers, int(0.5 * N)):
        return None

    # 用 inliers 做一次最小二乘精修
    xi = xs[best_inliers].astype(np.float32)
    yi = ys[best_inliers].astype(np.float32)
    a, b = np.polyfit(xi, yi, deg=1)
    return float(a), float(b), best_inliers


def horizon_line_ransac(mask: np.ndarray,
                        ignore_index: int = 255,
                        seed: int = 0):
    """
    返回用于画线的 pts [(x,y), ...]
    """
    H, W = mask.shape
    xs, ys = horizon_points_from_mask(mask, ignore_index=ignore_index)
    if xs.size < 2:
        return []

    fit = ransac_fit_line(xs, ys, seed=seed)
    if fit is None:
        # 退化：直接用原始点连线（至少能画出来）
        return [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]

    a, b, _ = fit
    pts = []
    for x in range(W):
        y = int(round(a * x + b))
        y = max(0, min(H - 1, y))
        pts.append((x, y))
    return pts


def horizon_polyline(mask: np.ndarray, ignore_index: int = 255):
    """
    原始逐列连线（不鲁棒），主要用于 GT 或 fallback
    """
    xs, ys = horizon_points_from_mask(mask, ignore_index=ignore_index)
    return [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]


def draw_polyline(rgb: np.ndarray, pts, color_rgb, thickness=2):
    if len(pts) < 2:
        return rgb
    img = rgb.copy()
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color_bgr, thickness, lineType=cv2.LINE_AA)
    return img


def run_model_get_seg(model, img_bchw):
    try:
        out = model(img_bchw, None, enable_restoration=True, enable_segmentation=True)
    except TypeError:
        out = model(img_bchw, None)

    if isinstance(out, (list, tuple)):
        return out[1]  # seg_logits
    raise RuntimeError("Model output unexpected.")


def make_mosaic(rgb, gt_mask_vis, overlay, pred_mask_vis):
    """
    2x2 拼图：
    [rgb | gt]
    [ovl | pred]
    """
    H, W, _ = rgb.shape
    top = np.concatenate([rgb, gt_mask_vis], axis=1)
    bot = np.concatenate([overlay, pred_mask_vis], axis=1)
    return np.concatenate([top, bot], axis=0)


def put_label(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="segmentation")
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(ds, [n_train, n_val], generator=g)

    random.seed(SEED)
    picks = random.sample(range(len(val_ds)), k=min(NUM_SAMPLES, len(val_ds)))

    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if not os.path.exists(CKPT):
        print(f"[Error] Checkpoint not found: {CKPT}")
        return
    sd = safe_load(CKPT, DEVICE)
    model.load_state_dict(sd, strict=False)
    print(f"[OK] Loaded weights from {CKPT}")

    model.eval()
    print(f"[Running] Saving results to: {os.path.abspath(OUT_DIR)}")

    with torch.no_grad():
        for i, idx in enumerate(picks):
            img_chw, gt_hw = val_ds[idx]
            rgb = tensor_to_uint8_rgb(img_chw)
            gt_hw_np = gt_hw.numpy().astype(np.int32)

            img_bchw = img_chw.unsqueeze(0).to(DEVICE)
            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                seg_logits = run_model_get_seg(model, img_bchw)
                pred_raw = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

            # --- mask 清理（强烈建议）---
            pred_pp = post_process_mask(pred_raw, morph_close_ksize=5, top_eps=2)

            # --- 画线：GT 用 polyline，Pred 用 RANSAC ---
            pts_gt = horizon_polyline(gt_hw_np)
            pts_pd = horizon_line_ransac(pred_pp, seed=SEED + i)

            overlay = overlay_mask(rgb, pred_pp, gt_hw_np)
            overlay = draw_polyline(overlay, pts_gt, color_rgb=(0, 255, 0), thickness=2)   # GT green
            overlay = draw_polyline(overlay, pts_pd, color_rgb=(255, 0, 0), thickness=2)   # Pred red

            gt_vis = mask_to_vis(gt_hw_np.astype(np.uint8))
            pred_vis = mask_to_vis(pred_pp.astype(np.uint8))

            mosaic = make_mosaic(rgb, gt_vis, overlay, pred_vis)
            put_label(mosaic, "RGB", 10, 25)
            put_label(mosaic, "GT Mask", rgb.shape[1] + 10, 25)
            put_label(mosaic, "Overlay (GT=Green, Pred=Red)", 10, rgb.shape[0] + 25)
            put_label(mosaic, "Pred Mask (Post)", rgb.shape[1] + 10, rgb.shape[0] + 25)

            out_path = os.path.join(OUT_DIR, f"{i:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
