# vis_stage_b.py
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
CKPT     = "rghnet_stage_b.pth"
DCE_WEIGHTS = "Epoch99.pth"
IMG_SIZE = 384

OUT_DIR  = "seg_vis"
NUM_SAMPLES = 30        # 输出多少张图
VAL_RATIO = 0.2         # 从数据集中切出 20% 当“展示/验证集”
SEED = 123
# ============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path: str, map_location: str):
    """避免 torch.load weights_only 的 FutureWarning，兼容旧 torch"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] float(0-1) -> [H,W,3] uint8 RGB"""
    img = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def overlay_mask(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    pred/gt: [H,W] int
    约定：gt里 255 是 ignore
    sky=1 用颜色叠加显示：
      GT=绿, Pred=红, 重叠=黄, Ignore=白
    """
    out = rgb.copy()

    gt_sky = (gt == 1)
    pd_sky = (pred == 1)
    ign = (gt == 255)

    # GT green
    out[gt_sky] = (out[gt_sky] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    # Pred red
    out[pd_sky] = (out[pd_sky] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    # overlap yellow
    out[gt_sky & pd_sky] = (out[gt_sky & pd_sky] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    # ignore white-ish
    out[ign] = (out[ign] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)

    return out


def horizon_polyline(mask: np.ndarray, ignore_index=255):
    """
    从 mask 估计地平线折线：对每一列找 sky(1) 的最后一个像素 y
    返回 pts: [(x,y), ...]，如果某列没有有效 sky，则跳过
    """
    H, W = mask.shape
    pts = []
    for x in range(W):
        col = mask[:, x]
        valid = (col != ignore_index)
        if valid.sum() == 0:
            continue
        sky = (col == 1) & valid
        if sky.sum() == 0:
            continue
        y = int(np.where(sky)[0].max())
        pts.append((x, y))
    return pts


def draw_polyline(rgb: np.ndarray, pts, color_rgb, thickness=2):
    if len(pts) < 2:
        return rgb
    img = rgb.copy()
    # cv2 是 BGR，所以要倒一下
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i+1], color_bgr, thickness, lineType=cv2.LINE_AA)
    return img


def run_model_get_seg(model, img_bchw, target_clean_or_none=None):
    """
    兼容你不同版本 forward：
    1) model(img, None, enable_restoration=True, enable_segmentation=True) -> (restored, seg, something)
    2) model(img, None) -> (restored, seg) 或 (restored, seg, something)
    """
    try:
        out = model(img_bchw, target_clean_or_none,
                    enable_restoration=True, enable_segmentation=True)
    except TypeError:
        out = model(img_bchw, target_clean_or_none)

    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            seg = out[1]
            return seg
    raise RuntimeError("Model forward output is unexpected; cannot find seg logits.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Dataset (segmentation 模式会返回 (clean_img, mask))
    ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="segmentation")
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(ds, [n_train, n_val], generator=g)

    # 2) 随机挑样本
    random.seed(SEED)
    picks = random.sample(range(len(val_ds)), k=min(NUM_SAMPLES, len(val_ds)))

    # 3) Load model
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    sd = safe_load(CKPT, DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"[OK] Saving {len(picks)} visualizations to: {os.path.abspath(OUT_DIR)}")

    # 4) Inference + save
    with torch.no_grad():
        for i, idx in enumerate(picks):
            img_chw, gt_hw = val_ds[idx]  # img: [3,H,W], gt: [H,W]
            img_bchw = img_chw.unsqueeze(0).to(DEVICE)
            gt_hw_np = gt_hw.numpy().astype(np.int32)

            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                seg_logits = run_model_get_seg(model, img_bchw, None)  # [1,2,H,W]
                pred_hw = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

            rgb = tensor_to_uint8_rgb(img_chw)
            over = overlay_mask(rgb, pred_hw, gt_hw_np)

            # 画地平线折线：GT(绿) / Pred(红)
            pts_gt = horizon_polyline(gt_hw_np)
            pts_pd = horizon_polyline(pred_hw)
            over2 = draw_polyline(over, pts_gt, color_rgb=(0, 255, 0), thickness=2)
            over2 = draw_polyline(over2, pts_pd, color_rgb=(255, 0, 0), thickness=2)

            # 保存：overlay / pred / gt
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_overlay.png"),
                        cv2.cvtColor(over2, cv2.COLOR_RGB2BGR))

            pred_vis = (pred_hw * 255).astype(np.uint8)  # 0/255
            gt_vis = gt_hw_np.copy().astype(np.uint8)
            gt_vis[gt_vis == 1] = 255
            gt_vis[gt_vis == 255] = 127  # ignore band 用灰显示

            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_pred.png"), pred_vis)
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_gt.png"), gt_vis)
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_img.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print("[DONE] Open ./seg_vis to view results.")


if __name__ == "__main__":
    main()
