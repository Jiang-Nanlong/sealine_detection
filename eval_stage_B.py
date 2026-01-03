# eval_stage_b.py
import os
import glob
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
CKPT = "rghnet_stage_b.pth"
DCE_WEIGHTS = "Epoch99.pth"
IMG_SIZE = 384

BATCH_SIZE = 8
VAL_RATIO = 0.2
SEED = 123
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path, map_location):
    """避免 torch.load 的 weights_only warning（兼容老版本 PyTorch）"""
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

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": iou, "dice": dice, "acc": acc, "prec": prec, "rec": rec}, pred


def horizon_mae(pred_hw, mask_hw, ignore_index=255):
    """
    pred_hw/mask_hw: [H,W] numpy int arrays
    通过“每列 sky(1) 的最后一个像素”估计地平线 y
    """
    H, W = mask_hw.shape
    valid_cols = 0
    err_sum = 0.0

    for x in range(W):
        col_gt = mask_hw[:, x]
        col_pd = pred_hw[:, x]
        valid = (col_gt != ignore_index)
        if valid.sum() == 0:
            continue

        gt_sky = (col_gt == 1) & valid
        pd_sky = (col_pd == 1) & valid
        if gt_sky.sum() == 0 or pd_sky.sum() == 0:
            continue

        # numpy 写法：找 sky 像素的最后一个索引
        y_gt = int(np.where(gt_sky)[0].max())
        y_pd = int(np.where(pd_sky)[0].max())

        err_sum += abs(y_pd - y_gt)
        valid_cols += 1

    if valid_cols == 0:
        return None
    return err_sum / valid_cols


def save_vis(img_chw, pred_hw, mask_hw, out_path):
    """
    img_chw: [3,H,W] float 0-1 RGB
    叠加：GT=绿，Pred=红，重叠=黄，Ignore=白
    """
    img = (img_chw.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    overlay = img.copy()

    gt = (mask_hw == 1)
    pd = (pred_hw == 1)
    ign = (mask_hw == 255)

    overlay[gt] = (overlay[gt] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[pd] = (overlay[pd] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    overlay[gt & pd] = (overlay[gt & pd] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    overlay[ign] = (overlay[ign] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)

    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


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
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE == "cuda")
    )

    # 3) 模型加载
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    state = safe_load(CKPT, DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    os.makedirs("eval_vis", exist_ok=True)

    # 4) Eval
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    horizon_errs = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(val_loader, desc="Eval Stage B")):
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            # 兼容 forward 签名（你现在的版本返回 restored_img, seg_logits, target_enhanced）
            out = model(img, None)  # target_clean=None
            seg_logits = out[1]

            m, pred = compute_metrics(seg_logits, mask)
            for k in totals:
                totals[k] += m[k]

            pred_np = pred.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for b in range(pred_np.shape[0]):
                mae = horizon_mae(pred_np[b], mask_np[b])
                if mae is not None:
                    horizon_errs.append(mae)

            # 保存少量可视化
            if i < 10:
                for b in range(min(pred_np.shape[0], 4)):
                    save_vis(
                        img[b].cpu(), pred_np[b], mask_np[b],
                        os.path.join("eval_vis", f"batch{i}_idx{b}.png")
                    )

    # 5) 汇总
    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    h_mae = float(np.mean(horizon_errs)) if horizon_errs else None

    print("\n===== Stage B Validation =====")
    print(f"PixelAcc={acc:.4f}  IoU(sky=1)={iou:.4f}  Dice={dice:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    if h_mae is not None:
        print(f"Horizon MAE (pixels) = {h_mae:.2f}  (lower is better)")
    print("Saved visualization to ./eval_vis")


if __name__ == "__main__":
    main()
