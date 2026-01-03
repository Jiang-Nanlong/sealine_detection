# vis_stage_B.py  (robust horizon visualization)
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

# 你想看哪个阶段的分割效果就换哪个权重
CKPT = "rghnet_stage_b.pth"          # 或 rghnet_best_seg.pth / rghnet_stage_c2.pth 等
DCE_WEIGHTS = "Epoch99.pth"

IMG_SIZE = 384
OUT_DIR  = "seg_vis_robust"
NUM_SAMPLES = 30
VAL_RATIO = 0.2
SEED = 123
# ============================

SKY_ID = 1
IGNORE_ID = 255

# --- 后处理参数（只作用于 Pred mask） ---
# 建议先把 MORPH_CLOSE 设为 0 或 3，避免“闭运算把假阳性连大”
MORPH_CLOSE = 3         # 0=不做闭运算；3/5=温和；7=可能把假阳性连大
TOP_TOUCH_TOL = 2       # 连通域必须接触顶边的容差像素

# --- RANSAC 参数（决定红线稳不稳） ---
RANSAC_ITERS = 300
RANSAC_THRESH = 3       # inlier 判定阈值（像素）
PREFILTER_PCTL = 85     # 预过滤：只保留 y <= pctl 的点（把掉海里的大 y 先剔除）
MIN_POINTS = 80         # 点太少就别拟合（直接退化成原始线）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path: str, map_location: str):
    """torch.load without FutureWarning (weights_only=True) when supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def post_process_mask_top_connected(mask_np: np.ndarray,
                                    sky_id=SKY_ID,
                                    ignore_id=IGNORE_ID,
                                    morph_close=MORPH_CLOSE,
                                    top_touch_tol=TOP_TOUCH_TOL) -> np.ndarray:
    """
    只保留“接触图像顶边”的天空连通域，去掉海面上的 sky 假阳性岛。
    注意：morph_close 过大可能把假阳性连上天空导致保留，建议 0/3/5。
    """
    H, W = mask_np.shape

    # 只对有效区域做处理：ignore 保持 ignore
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
    # 原来被预测成 sky，但不在“顶边连通天空”里的，一律改成 sea(0)
    out[(mask_np == sky_id) & (keep == 0)] = 0
    return out


def overlay_mask(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """GT=绿, Pred=红, overlap=黄, ignore=白"""
    out = rgb.copy()
    gt_sky = (gt == SKY_ID)
    pd_sky = (pred == SKY_ID)
    ign = (gt == IGNORE_ID)

    out[gt_sky] = (out[gt_sky] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    out[pd_sky] = (out[pd_sky] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    out[gt_sky & pd_sky] = (out[gt_sky & pd_sky] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    out[ign] = (out[ign] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)
    return out


def draw_polyline(rgb: np.ndarray, pts, color_rgb, thickness=2) -> np.ndarray:
    if pts is None or len(pts) < 2:
        return rgb
    img = rgb.copy()
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color_bgr, thickness, lineType=cv2.LINE_AA)
    return img


def horizon_points_from_mask(mask: np.ndarray, sky_id=SKY_ID, ignore_id=IGNORE_ID):
    """
    每列提取一个候选点：该列 sky 的最大 y（最靠下的 sky 像素）
    """
    H, W = mask.shape
    pts = []
    for x in range(W):
        col = mask[:, x]
        valid = (col != ignore_id)
        if valid.sum() == 0:
            continue
        sky = (col == sky_id) & valid
        if sky.sum() == 0:
            continue
        y = int(np.where(sky)[0].max())
        pts.append((x, y))
    return pts


def ransac_fit_line(points, iters=RANSAC_ITERS, thresh=RANSAC_THRESH):
    """
    RANSAC 拟合 y = a*x + b
    points: list[(x,y)]
    return: (a,b,inlier_mask)
    """
    if points is None or len(points) < 2:
        return None, None, None

    pts = np.array(points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    n = len(points)
    best_inliers = None
    best_cnt = 0
    best_a, best_b = None, None

    # 随机采样两点定义直线
    for _ in range(iters):
        i1, i2 = np.random.choice(n, 2, replace=False)
        x1, y1 = xs[i1], ys[i1]
        x2, y2 = xs[i2], ys[i2]
        if abs(x2 - x1) < 1e-6:
            continue
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        y_pred = a * xs + b
        err = np.abs(ys - y_pred)
        inliers = err <= thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
            best_a, best_b = float(a), float(b)

    if best_inliers is None or best_cnt < 2:
        return None, None, None

    # 用 inliers 做一次最小二乘再拟合（更稳）
    xs_in = xs[best_inliers]
    ys_in = ys[best_inliers]
    A = np.vstack([xs_in, np.ones_like(xs_in)]).T
    a_ls, b_ls = np.linalg.lstsq(A, ys_in, rcond=None)[0]
    return float(a_ls), float(b_ls), best_inliers


def robust_horizon_polyline(mask: np.ndarray,
                            sky_id=SKY_ID,
                            ignore_id=IGNORE_ID,
                            prefilter_pctl=PREFILTER_PCTL,
                            min_points=MIN_POINTS):
    """
    输出稳定的地平线 polyline（用于 Pred）
    1) 提点
    2) 预过滤（剔除 y 很大的离群列，通常是海面 sky leak）
    3) RANSAC 拟合直线 -> 生成整宽度 polyline
    """
    H, W = mask.shape
    pts = horizon_points_from_mask(mask, sky_id=sky_id, ignore_id=ignore_id)
    if len(pts) < min_points:
        # 点太少：退化成原始连线（至少还能画出来）
        return pts, {"mode": "raw", "n": len(pts)}

    pts_np = np.array(pts, dtype=np.int32)
    ys = pts_np[:, 1].astype(np.float32)

    # 预过滤：只保留 y <= percentile 的点（大 y 通常是海面假阳性拖下去）
    cut = np.percentile(ys, prefilter_pctl)
    keep_mask = (ys <= cut)
    pts_f = pts_np[keep_mask]
    if len(pts_f) < min_points // 2:
        pts_f = pts_np  # 过滤太狠就不滤

    pts_f_list = [(int(x), int(y)) for x, y in pts_f.tolist()]

    a, b, inliers = ransac_fit_line(pts_f_list)
    if a is None:
        return pts, {"mode": "raw_fallback", "n": len(pts)}

    # 用拟合线生成整幅宽度的点
    line_pts = []
    for x in range(W):
        y = int(round(a * x + b))
        y = max(0, min(H - 1, y))
        line_pts.append((x, y))

    info = {
        "mode": "ransac",
        "n_raw": len(pts),
        "n_fit": len(pts_f_list),
        "a": a,
        "b": b,
    }
    return line_pts, info


def run_model_get_seg_logits(model, img_bchw):
    # 兼容你 unet_model 里两种 forward 签名
    try:
        out = model(img_bchw, None, enable_restoration=True, enable_segmentation=True)
    except TypeError:
        out = model(img_bchw, None)

    if isinstance(out, (list, tuple)):
        return out[1]  # seg_logits
    raise RuntimeError("Model output unexpected.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    ds = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode="segmentation")
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(ds, [n_train, n_val], generator=g)

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
            gt_np = gt_hw.numpy().astype(np.int32)

            with amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == "cuda")):
                seg_logits = run_model_get_seg_logits(model, img_bchw)
                pred_raw = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

            # Pred 后处理（去海面孤岛 + 可选温和闭运算）
            pred_pp = post_process_mask_top_connected(
                pred_raw,
                sky_id=SKY_ID,
                ignore_id=IGNORE_ID,
                morph_close=MORPH_CLOSE,
                top_touch_tol=TOP_TOUCH_TOL,
            )

            rgb = tensor_to_uint8_rgb(img_chw)

            # 覆盖图（Pred 用后处理后的 mask）
            over = overlay_mask(rgb, pred_pp, gt_np)

            # GT 线（原始列扫描，保留标注形状）
            pts_gt = horizon_points_from_mask(gt_np, sky_id=SKY_ID, ignore_id=IGNORE_ID)

            # Pred 线（RANSAC 稳定版）
            pts_pd, info = robust_horizon_polyline(pred_pp)

            # 画线：GT=绿，Pred=红
            over2 = draw_polyline(over, pts_gt, color_rgb=(0, 255, 0), thickness=2)
            over2 = draw_polyline(over2, pts_pd, color_rgb=(255, 0, 0), thickness=2)

            # 保存
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_overlay.png"),
                        cv2.cvtColor(over2, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_pred_raw.png"),
                        (pred_raw * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(OUT_DIR, f"{i:03d}_pred_pp.png"),
                        (pred_pp * 255).astype(np.uint8))

            if i < 5:
                print(f"[{i:03d}] pred_line={info.get('mode')}  n_raw={info.get('n_raw', info.get('n', -1))}")

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
