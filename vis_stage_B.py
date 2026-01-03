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
CKPT     = "rghnet_best_seg.pth"   # 建议：best_seg / best_joint / stage_c2
DCE_WEIGHTS = "Epoch99.pth"
IMG_SIZE = 384

OUT_DIR  = "seg_vis_fixed"
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


def post_process_pred_mask(pred_np: np.ndarray,
                           sky_id: int = 1,
                           open_ksize: int = 3,
                           close_ksize: int = 0,
                           top_margin: int = 2,
                           min_area: int = 50) -> np.ndarray:
    """
    pred_np: [H,W] 0/1 (or multi-class but sky_id=1)
    目标：
      - 去掉海面里零星 sky 小岛（导致 horizon 断崖）
      - 保留“与图像顶边连通”的天空区域（真实天空几乎必然接触 y=0）
    """

    H, W = pred_np.shape
    sky = (pred_np == sky_id).astype(np.uint8)

    # A) opening：去细条噪声/雨线误检（可选）
    if open_ksize and open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, k)

    # B) 连通域：只保留接触顶边的天空（核心）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)

    keep = np.zeros_like(sky)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        y_top = stats[i, cv2.CC_STAT_TOP]
        if area < min_area:
            continue
        if y_top <= top_margin:
            keep[labels == i] = 1

    # C) closing：只对“保留下来的天空”做小闭运算填小洞（可选，默认关）
    if close_ksize and close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k)

    out = pred_np.copy()
    out[(pred_np == sky_id) & (keep == 0)] = 0
    return out


def overlay_mask(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    gt_sky = (gt == 1)
    pd_sky = (pred == 1)
    ign = (gt == 255)

    out[gt_sky] = (out[gt_sky] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)       # GT: 绿
    out[pd_sky] = (out[pd_sky] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)       # Pred: 红
    out[gt_sky & pd_sky] = (out[gt_sky & pd_sky] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)  # 重合: 黄
    out[ign] = (out[ign] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)
    return out


def median_filter_1d(y: np.ndarray, k: int = 21) -> np.ndarray:
    """简单 1D 中值滤波（不依赖 scipy），k 必须是奇数"""
    if k <= 1:
        return y
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i+k])
    return out


def horizon_polyline(mask: np.ndarray,
                     ignore_index: int = 255,
                     smooth: bool = True,
                     median_k: int = 21,
                     max_jump: int = 20):
    """
    从 mask 取地平线：
      - 每列取“天空最靠下的像素 y”
      - 可选：中值滤波 + 限跳变（强力消灭断崖折线）
    """
    H, W = mask.shape
    ys = np.full((W,), -1, dtype=np.int32)

    for x in range(W):
        col = mask[:, x]
        valid = (col != ignore_index)
        if valid.sum() == 0:
            continue
        sky = (col == 1) & valid
        if sky.sum() == 0:
            continue
        ys[x] = int(np.where(sky)[0].max())

    # 如果有缺失列，用邻近值补一下
    ok = ys >= 0
    if ok.sum() < 2:
        return []
    xs_ok = np.where(ok)[0]
    ys_ok = ys[ok].astype(np.float32)
    ys_interp = np.interp(np.arange(W), xs_ok, ys_ok).astype(np.int32)

    if smooth:
        ys_f = median_filter_1d(ys_interp.astype(np.float32), k=median_k).astype(np.int32)
        # 限跳变：防止单列异常点突然大幅下坠
        for x in range(1, W):
            if abs(int(ys_f[x]) - int(ys_f[x-1])) > max_jump:
                ys_f[x] = ys_f[x-1]
        ys_use = ys_f
    else:
        ys_use = ys_interp

    pts = [(x, int(ys_use[x])) for x in range(W)]
    return pts


def draw_polyline(rgb: np.ndarray, pts, color_rgb, thickness=2):
    if len(pts) < 2:
        return rgb
    img = rgb.copy()
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color_bgr, thickness, lineType=cv2.LINE_AA)
    return img


def run_model_get_seg(model, img_bchw):
    # 兼容你前面多阶段 forward
    try:
        out = model(img_bchw, None, enable_restoration=True, enable_segmentation=True)
    except TypeError:
        out = model(img_bchw, None)

    if isinstance(out, (list, tuple)):
        return out[1]  # seg_logits
    raise RuntimeError("Model output unexpected.")


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
    if os.path.exists(CKPT):
        sd = safe_load(CKPT, DEVICE)
        model.load_state_dict(sd, strict=False)
        print(f"[OK] Loaded weights from {CKPT}")
    else:
        print(f"[Error] Checkpoint not found: {CKPT}")
        return

    model.eval()
    print(f"[Running] Saving results to: {os.path.abspath(OUT_DIR)}")

    with torch.no_grad():
        for i, idx in enumerate(picks):
            img_chw, gt_hw = val_ds[idx]
            img_bchw = img_chw.unsqueeze(0).to(DEVICE)
            gt_hw_np = gt_hw.numpy().astype(np.int32)

            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                seg_logits = run_model_get_seg(model, img_bchw)
                pred_raw = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

            # ✅ 后处理：去孤岛 + 稳线
            pred_hw = post_process_pred_mask(
                pred_raw,
                open_ksize=3,     # 建议 3；雨特别重可试 5
                close_ksize=0,    # 默认关；真有“天空被切碎”才试 3
                top_margin=2,
                min_area=50
            )

            rgb = tensor_to_uint8_rgb(img_chw)
            over = overlay_mask(rgb, pred_hw, gt_hw_np)

            pts_gt = horizon_polyline(gt_hw_np, smooth=True, median_k=21, max_jump=20)
            pts_pd = horizon_polyline(pred_hw,  smooth=True, median_k=21, max_jump=20)

            over2 = draw_polyline(over, pts_gt, color_rgb=(0, 255, 0), thickness=2)   # GT: 绿线
            over2 = draw_polyline(over2, pts_pd, color_rgb=(255, 0, 0), thickness=2)  # Pred: 红线

            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_overlay.png"), cv2.cvtColor(over2, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_pred.png"), (pred_hw * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_gt.png"), (np.clip(gt_hw_np, 0, 1) * 255).astype(np.uint8))

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
