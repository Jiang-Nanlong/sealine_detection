# evaluate_fusion_cnn_with_outliers.py
# ------------------------------------------------------------
# Evaluate Fusion-CNN on FusionCache and export:
#   - per-sample metrics CSV
#   - outlier list TXT
#   - outlier visualizations (GT vs Pred lines)
#
# IMPORTANT:
# This script is aligned with the CURRENT FusionCache label encoding
# produced by make_fusion_cache.py in this repo:
#   label[0] = rho_norm in [0,1] where rho_index = rho_norm * RESIZE_H (padded height)
#   label[1] = theta_norm in [0,1] where theta_rad = theta_norm * pi  (theta period=180°)
#
# The returned (rho, theta) are in the UNet-image coordinate system (1024x576):
# origin at image center, compatible with drawing on UNet output images.
# ------------------------------------------------------------

import os
import math
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import cv2

from cnn_model import HorizonResNet


# =========================
# User config (PyCharm friendly)
# =========================
SPLIT = "test"  # "train" / "val" / "test"
WEIGHTS_PATH = r"splits_musid/best_fusion_cnn_1024x576.pth"
CACHE_ROOT = r"Hashmani's Dataset/FusionCache_1024x576"
CACHE_DIR = os.path.join(CACHE_ROOT, SPLIT)

# Where to find original images to draw outliers.
# If your cache stores only basename (e.g., DSC_0001.jpg), set this to the dataset root.
ORIGINAL_IMG_ROOT = r"Hashmani's Dataset"  # <-- change to your image root if needed

OUT_DIR = "eval_outputs"
OUTLIER_DIR = os.path.join(OUT_DIR, f"outliers_{SPLIT}")
CSV_PATH = os.path.join(OUT_DIR, f"eval_{SPLIT}.csv")
OUTLIER_TXT = os.path.join(OUT_DIR, f"outliers_{SPLIT}.txt")

# Outlier selection
TOPK_OUTLIERS = 50
THRESH_RHO_ORIG_PX = 20.0
THRESH_EDGEY_ORIG_PX = 20.0
THRESH_THETA_DEG = 2.0
THRESH_LINE_DIST_ORIG_PX = 20.0
THRESH_CONF_MAX = 0.15  # also treat very low confidence as outlier (if conf head exists)

# Visualization
DRAW_THICKNESS = 3
FONT_SCALE = 0.55

# Original resolution scaling (approx) for reporting pixel errors in the original image
ORIG_W_APPROX = 1920.0
ORIG_H_APPROX = 1080.0

# FusionCache / UNet geometry
UNET_W = 1024
UNET_H = 576

# FusionCache label geometry (padded height for Radon rho axis)
RESIZE_W = 1024
RESIZE_H = 2240
PAD_TOP = 832

BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Label de-normalization (must match make_fusion_cache.py)
# =========================
@dataclass
class DenormConfig:
    resize_w: int = RESIZE_W
    resize_h: int = RESIZE_H
    unet_w: int = UNET_W
    unet_h: int = UNET_H
    pad_top: int = PAD_TOP

    @property
    def x0(self) -> float:
        return self.resize_w / 2.0

    @property
    def y0(self) -> float:
        # padded origin center in resize_h space
        return self.pad_top + self.unet_h / 2.0


def denorm_rho_theta(rho_norm: float, theta_norm: float, cfg: DenormConfig) -> Tuple[float, float]:
    """Convert normalized (rho_norm, theta_norm) to (rho_px, theta_deg) in UNet-image coordinates."""
    theta_rad = float(theta_norm) * math.pi
    theta_deg = theta_rad * 180.0 / math.pi

    # rho_norm corresponds to an absolute rho-index (y) in padded RESIZE_H space.
    y = float(rho_norm) * float(cfg.resize_h)

    # Convert to centered rho in the coordinate system whose origin is the UNet center.
    # Because both the point coordinates and origin shift by pad_top, this rho is
    # directly usable in UNet-space computations with origin at (unet_w/2, unet_h/2).
    rho = y - float(cfg.y0)
    return rho, theta_deg


def angular_diff_deg(a_deg: float, b_deg: float, period: float = 180.0) -> float:
    """Minimum absolute angular difference with wrap-around (period=180° for Radon)."""
    d = abs(a_deg - b_deg) % period
    return min(d, period - d)


# =========================
# Line geometry / metrics
# =========================
def _theta_ab(theta_deg: float) -> Tuple[float, float]:
    th = math.radians(theta_deg)
    return math.cos(th), math.sin(th)  # a,b


def line_y_at_x(rho: float, theta_deg: float, x: float, w: int, h: int) -> float:
    """Return y (in pixels) where the line (rho, theta) crosses x, in UNet image coordinates."""
    a, b = _theta_ab(theta_deg)
    x0 = w / 2.0
    y0 = h / 2.0

    # rho = a*(x-x0) + b*(y-y0)
    if abs(b) < 1e-6:
        # nearly vertical line: no single y
        return float("nan")
    return (rho - a * (x - x0)) / b + y0


def point_to_line_distance(x: float, y: float, rho: float, theta_deg: float, w: int, h: int) -> float:
    a, b = _theta_ab(theta_deg)
    x0 = w / 2.0
    y0 = h / 2.0
    return abs(a * (x - x0) + b * (y - y0) - rho)


def mean_point_to_line_distance(
    rho_pred: float,
    theta_pred_deg: float,
    rho_gt: float,
    theta_gt_deg: float,
    w: int,
    h: int,
    n_samples: int = 200,
) -> float:
    """Mean distance from sampled points on GT line to the predicted line."""
    xs = np.linspace(0.0, float(w - 1), n_samples, dtype=np.float32)
    ys = np.array([line_y_at_x(rho_gt, theta_gt_deg, float(x), w, h) for x in xs], dtype=np.float32)

    ok = np.isfinite(ys)
    if not np.any(ok):
        return float("nan")

    xs = xs[ok]
    ys = ys[ok]
    d = [point_to_line_distance(float(x), float(y), rho_pred, theta_pred_deg, w, h) for x, y in zip(xs, ys)]
    return float(np.mean(d)) if len(d) else float("nan")


def edge_y_error(
    rho_pred: float,
    theta_pred_deg: float,
    rho_gt: float,
    theta_gt_deg: float,
    w: int,
    h: int,
) -> float:
    """Avg |y_pred - y_gt| at left and right image edges."""
    yL_p = line_y_at_x(rho_pred, theta_pred_deg, 0.0, w, h)
    yR_p = line_y_at_x(rho_pred, theta_pred_deg, float(w - 1), w, h)
    yL_g = line_y_at_x(rho_gt, theta_gt_deg, 0.0, w, h)
    yR_g = line_y_at_x(rho_gt, theta_gt_deg, float(w - 1), w, h)

    vals = []
    if np.isfinite(yL_p) and np.isfinite(yL_g):
        vals.append(abs(float(yL_p) - float(yL_g)))
    if np.isfinite(yR_p) and np.isfinite(yR_g):
        vals.append(abs(float(yR_p) - float(yR_g)))

    if not vals:
        return float("nan")
    return float(np.mean(vals))


# =========================
# Dataset
# =========================
class FusionCacheDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")), key=lambda p: int(Path(p).stem))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy found under: {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        idx = int(Path(path).stem)
        d = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(d["input"]).float()
        y = torch.from_numpy(d["label"]).float()
        img_name = d.get("img_name") or d.get("image_name") or d.get("name") or ""
        if not img_name:
            # fallback: store idx only
            img_name = f"{idx}.jpg"
        return x, y, img_name, f"{idx}.npy"


# =========================
# Model loading (with optional confidence head)
# =========================
def load_model() -> HorizonResNet:
    model = HorizonResNet(in_channels=4).to(DEVICE)

    try:
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    # support both pure state_dict and dict checkpoints
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


# =========================
# Visualization helpers
# =========================
def _safe_int(v: float, lo: int, hi: int) -> int:
    if not np.isfinite(v):
        return int((lo + hi) // 2)
    return int(max(lo, min(hi, round(float(v)))))


def draw_lines_and_text(
    img: np.ndarray,
    rho_gt: float,
    theta_gt: float,
    rho_pr: float,
    theta_pr: float,
    text_lines: List[str],
) -> np.ndarray:
    out = img.copy()

    yL_g = line_y_at_x(rho_gt, theta_gt, 0.0, UNET_W, UNET_H)
    yR_g = line_y_at_x(rho_gt, theta_gt, float(UNET_W - 1), UNET_W, UNET_H)
    yL_p = line_y_at_x(rho_pr, theta_pr, 0.0, UNET_W, UNET_H)
    yR_p = line_y_at_x(rho_pr, theta_pr, float(UNET_W - 1), UNET_W, UNET_H)

    p1g = (0, _safe_int(yL_g, -UNET_H, 2 * UNET_H))
    p2g = (UNET_W - 1, _safe_int(yR_g, -UNET_H, 2 * UNET_H))
    p1p = (0, _safe_int(yL_p, -UNET_H, 2 * UNET_H))
    p2p = (UNET_W - 1, _safe_int(yR_p, -UNET_H, 2 * UNET_H))

    # GT: green, Pred: red
    cv2.line(out, p1g, p2g, (0, 255, 0), DRAW_THICKNESS)
    cv2.line(out, p1p, p2p, (0, 0, 255), DRAW_THICKNESS)

    y0 = 22
    for t in text_lines:
        cv2.putText(out, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
        y0 += 20

    return out


def find_image_path(img_name: str) -> Optional[str]:
    # absolute
    if os.path.isabs(img_name) and os.path.exists(img_name):
        return img_name

    # common direct joins
    cand = [
        os.path.join(ORIGINAL_IMG_ROOT, img_name),
        os.path.join(ORIGINAL_IMG_ROOT, SPLIT, img_name),
        os.path.join(ORIGINAL_IMG_ROOT, "test", img_name),
        os.path.join(ORIGINAL_IMG_ROOT, "val", img_name),
        os.path.join(ORIGINAL_IMG_ROOT, "train", img_name),
    ]
    for p in cand:
        if os.path.exists(p):
            return p

    # fallback: recursive search (only used for outliers)
    hits = glob.glob(os.path.join(ORIGINAL_IMG_ROOT, "**", img_name), recursive=True)
    if hits:
        return hits[0]
    return None


def load_original_image(img_name: str) -> np.ndarray:
    p = find_image_path(img_name)
    if p is None:
        # blank placeholder
        return np.zeros((UNET_H, UNET_W, 3), dtype=np.uint8)

    img = cv2.imread(p)
    if img is None:
        return np.zeros((UNET_H, UNET_W, 3), dtype=np.uint8)

    img = cv2.resize(img, (UNET_W, UNET_H), interpolation=cv2.INTER_AREA)
    return img


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUTLIER_DIR, exist_ok=True)

    cfg = DenormConfig()
    scale_to_orig = ORIG_W_APPROX / float(UNET_W)

    ds = FusionCacheDataset(CACHE_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = load_model()

    rows: List[Dict] = []

    with torch.no_grad():
        for xb, yb, img_names, npy_names in dl:
            xb = xb.to(DEVICE, non_blocking=True)

            # Support optional confidence head
            conf = None
            try:
                pred, conf = model(xb, return_conf=True)  # type: ignore
            except TypeError:
                pred = model(xb)

            pred_np = pred.detach().cpu().numpy()
            yb_np = yb.numpy()
            conf_np = conf.detach().cpu().numpy().reshape(-1) if conf is not None else None

            for i in range(len(img_names)):
                rho_gt, theta_gt = denorm_rho_theta(float(yb_np[i, 0]), float(yb_np[i, 1]), cfg)
                rho_pr, theta_pr = denorm_rho_theta(float(pred_np[i, 0]), float(pred_np[i, 1]), cfg)

                rho_err = abs(rho_pr - rho_gt)
                theta_err = angular_diff_deg(theta_pr, theta_gt, period=180.0)

                line_dist = mean_point_to_line_distance(rho_pr, theta_pr, rho_gt, theta_gt, UNET_W, UNET_H)
                ey = edge_y_error(rho_pr, theta_pr, rho_gt, theta_gt, UNET_W, UNET_H)

                row = {
                    "img_name": str(img_names[i]),
                    "npy": str(npy_names[i]),
                    "rho_gt": float(rho_gt),
                    "theta_gt": float(theta_gt),
                    "rho_pred": float(rho_pr),
                    "theta_pred": float(theta_pr),
                    "rho_err": float(rho_err),
                    "theta_err_deg": float(theta_err),
                    "line_dist": float(line_dist),
                    "edge_y": float(ey),
                    "rho_err_orig": float(rho_err * scale_to_orig),
                    "line_dist_orig": float(line_dist * scale_to_orig),
                    "edge_y_orig": float(ey * scale_to_orig),
                }

                if conf_np is not None:
                    row["conf"] = float(_sigmoid(conf_np[i]))

                rows.append(row)

    # ---------- CSV ----------
    fieldnames = list(rows[0].keys()) if rows else []
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---------- outliers ----------
    def is_outlier(r: Dict) -> bool:
        if r["rho_err_orig"] >= THRESH_RHO_ORIG_PX:
            return True
        if r["edge_y_orig"] >= THRESH_EDGEY_ORIG_PX:
            return True
        if r["theta_err_deg"] >= THRESH_THETA_DEG:
            return True
        if r["line_dist_orig"] >= THRESH_LINE_DIST_ORIG_PX:
            return True
        if "conf" in r and float(r["conf"]) <= THRESH_CONF_MAX:
            return True
        return False

    out_thr = [r for r in rows if is_outlier(r)]
    out_top = sorted(rows, key=lambda d: d["edge_y_orig"], reverse=True)[:TOPK_OUTLIERS]

    out_map = {(r["img_name"], r["npy"]): r for r in (out_thr + out_top)}
    outliers = sorted(list(out_map.values()), key=lambda d: d["edge_y_orig"], reverse=True)

    with open(OUTLIER_TXT, "w", encoding="utf-8") as f:
        for k, r in enumerate(outliers):
            conf_str = f"\tconf={r['conf']:.3f}" if "conf" in r else ""
            f.write(
                f"{k}\t{r['img_name']}\t"
                f"theta={r['theta_err_deg']:.3f}\t"
                f"rho~{r['rho_err_orig']:.2f}\t"
                f"lineDist~{r['line_dist_orig']:.2f}\t"
                f"edgeY~{r['edge_y_orig']:.2f}{conf_str}\n"
            )

    # Save outlier visualizations
    for r in outliers:
        img = load_original_image(r["img_name"])
        text = [
            Path(r["img_name"]).name,
            f"theta_err={r['theta_err_deg']:.3f} deg",
            f"rho_err~{r['rho_err_orig']:.2f}px (orig approx)",
            f"lineDist~{r['line_dist_orig']:.2f}px",
            f"edgeY~{r['edge_y_orig']:.2f}px",
        ]
        if "conf" in r:
            text.append(f"conf={r['conf']:.3f}")

        vis = draw_lines_and_text(img, r["rho_gt"], r["theta_gt"], r["rho_pred"], r["theta_pred"], text)
        out_path = os.path.join(OUTLIER_DIR, Path(r["img_name"]).stem + ".png")
        cv2.imwrite(out_path, vis)

    # ---------- summary ----------
    def _arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=np.float32)

    def stat(x: np.ndarray) -> Tuple[float, float, float, float, float]:
        return float(np.mean(x)), float(np.median(x)), float(np.percentile(x, 90)), float(np.percentile(x, 95)), float(np.max(x))

    rho = _arr("rho_err")
    rhoo = _arr("rho_err_orig")
    th = _arr("theta_err_deg")
    ld = _arr("line_dist")
    ldo = _arr("line_dist_orig")
    ey = _arr("edge_y")
    eyo = _arr("edge_y_orig")

    print("========== Fusion-CNN Evaluation (with outliers) ==========")
    print(f"Split: {SPLIT} | N={len(rows)} | Device={DEVICE}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Cache:   {CACHE_DIR}\n")

    m, md, p90, p95, mx = stat(rho)
    print(f"Rho abs error (px, UNet {UNET_W}x{UNET_H}): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")
    m, md, p90, p95, mx = stat(rhoo)
    print(f"Rho abs error (px, original-scale approx): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")

    m, md, p90, p95, mx = stat(th)
    print(f"Theta error (deg, wrap-aware, period=180): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")

    m, md, p90, p95, mx = stat(ld)
    print(f"Mean point->line distance (px, UNet): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")
    m, md, p90, p95, mx = stat(ldo)
    print(f"Mean point->line distance (px, original-scale approx): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")

    m, md, p90, p95, mx = stat(ey)
    print(f"Edge-Y error (px, UNet): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}")
    m, md, p90, p95, mx = stat(eyo)
    print(f"Edge-Y error (px, original-scale approx): mean={m:.4f}, median={md:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={mx:.4f}\n")

    theta1 = float((th <= 1.0).mean() * 100)
    theta2 = float((th <= 2.0).mean() * 100)
    theta5 = float((th <= 5.0).mean() * 100)
    rho5 = float((rhoo <= 5.0).mean() * 100)
    rho10 = float((rhoo <= 10.0).mean() * 100)
    rho20 = float((rhoo <= 20.0).mean() * 100)
    ld5 = float((ldo <= 5.0).mean() * 100)
    ld10 = float((ldo <= 10.0).mean() * 100)
    ld20 = float((ldo <= 20.0).mean() * 100)
    ey5 = float((eyo <= 5.0).mean() * 100)
    ey10 = float((eyo <= 10.0).mean() * 100)
    ey20 = float((eyo <= 20.0).mean() * 100)

    print("---- Threshold stats (helpful for论文表格) ----")
    print(f"Theta <=1°: {theta1:.2f}% | <=2°: {theta2:.2f}% | <=5°: {theta5:.2f}%")
    print(f"Rho(orig) <=5px: {rho5:.2f}% | <=10px: {rho10:.2f}% | <=20px: {rho20:.2f}%")
    print(f"LineDist(orig) <=5px: {ld5:.2f}% | <=10px: {ld10:.2f}% | <=20px: {ld20:.2f}%")
    print(f"EdgeY(orig) <=5px: {ey5:.2f}% | <=10px: {ey10:.2f}% | <=20px: {ey20:.2f}%\n")

    print(f"[Saved] per-sample metrics -> {CSV_PATH}")
    print(f"[Outliers] selected={len(outliers)} (threshold-or-topK)")
    print(f"[Outliers] saving visualization to: {OUTLIER_DIR}")
    print(f"[Saved] outlier list -> {OUTLIER_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()
