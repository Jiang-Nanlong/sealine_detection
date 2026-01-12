# -*- coding: utf-8 -*-
"""
evaluate_fusion_cnn_with_outliers.py
====================================

用途（适合 PyCharm 直接运行，不需要命令行传参）：
  1) 在指定 split（默认 test）上评估最终 Fusion-CNN 权重：
       - rho 平均绝对误差（像素）：UNet 尺度（默认 1024x576）与原图尺度（默认 1920x1080）
       - theta 平均角度误差（度）：按 0~180° 环形距离计算（wrap-around）
       - 更贴近“海天线”的指标：GT 线段上的点到预测线的平均垂直距离（px）
       - 额外更直观：左右边界 y 值误差（px）

  2) 自动找出“误差异常大”的样本，并导出可视化图片：
       - 同一张图上画 GT 海天线（绿）+ 预测海天线（红）
       - 文件名尽量与原图一致，但输出到单独目录，不覆盖原图

注意：
  - 这里评估的是“整个框架+CNN”的结果（因为 FusionCache 已包含 UNet 输出+Radon 特征）。
  - 如果你改过 UNet 或 cache 生成逻辑，需要重新运行 make_fusion_cache.py 才能反映到这里。

------------------------------------------------------------
你只需要改动下面【配置区】即可运行。
"""

import os
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cnn_model import HorizonResNet

# =========================
# 配置区（按你本机路径修改）
# =========================
WEIGHTS_PATH = r"splits_musid/best_fusion_cnn_1024x576.pth"

CACHE_ROOT = r"Hashmani's Dataset/FusionCache_1024x576"
SPLIT = "test"  # "train" / "val" / "test"

# GroundTruth.csv：第一列是图片名；后四列是两点坐标（x1,y1,x2,y2），坐标基于原图(通常1920x1080)
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"

# UNet 输入尺寸（与你 make_fusion_cache.py 一致）
UNET_IN_W = 1024
UNET_IN_H = 576

# cache 里 sinogram 高度（与你 make_fusion_cache.py 一致）
RESIZE_H = 2240
ANGLE_BINS = 180

# 运行参数
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出目录（不要和原图同路径）
OUT_DIR = r"eval_outputs"
SAVE_CSV = True
CSV_NAME = f"eval_{SPLIT}.csv"

# Outlier 输出策略：
#   - 先按阈值筛选（任意一项超过阈值就算 outlier）
#   - 然后再补充 topK（按 line_dist_orig 从大到小）
OUTLIER_TOPK = 50
THRESH_THETA_DEG = 5.0          # 角度误差 > 5° 认为异常
THRESH_RHO_PX_ORIG = 20.0       # 原图尺度 rho 误差 > 20px 认为异常
THRESH_LINE_DIST_PX_ORIG = 20.0 # 原图尺度 平均点到线距离 > 20px 认为异常
THRESH_EDGEY_PX_ORIG = 20.0     # 原图尺度 左右边界 y 误差 > 20px 认为异常
LINE_SAMPLE_N = 60              # 计算 point->line 距离时在 GT 线段采样点数

# 画线样式（OpenCV BGR）
COLOR_GT = (0, 255, 0)    # Green
COLOR_PR = (0, 0, 255)    # Red
THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2


# =========================
# 数据集：读取 FusionCache
# =========================
class SplitCacheDataset(Dataset):
    """
    Cache 文件格式（make_fusion_cache.py 保存）：
      {idx}.npy 是 dict：
        - "input": (4, 2240, 180) float32
        - "label": (2,) [rho_norm, theta_norm] float32
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.files = sorted(
            [f for f in os.listdir(cache_dir) if f.endswith(".npy")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        fn = self.files[i]
        idx = int(os.path.splitext(fn)[0])
        obj = np.load(os.path.join(self.cache_dir, fn), allow_pickle=True).item()
        x = torch.from_numpy(obj["input"]).float()
        y = torch.from_numpy(obj["label"]).float()
        return x, y, idx


# =========================
# 工具：读图（兼容无扩展名）
# =========================
def read_image_any_ext(img_dir: str, img_name: str):
    candidates = [img_name]
    root, ext = os.path.splitext(img_name)
    if ext == "":
        candidates += [root + e for e in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
    for c in candidates:
        p = os.path.join(img_dir, c)
        if os.path.exists(p):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is not None:
                return bgr, c
    return None, None


# =========================
# 反归一化：rho/theta
# =========================
@dataclass
class DenormConfig:
    unet_w: int = UNET_IN_W
    unet_h: int = UNET_IN_H
    resize_h: int = RESIZE_H
    angle_range_deg: float = 180.0


def denorm_rho_theta(rho_norm: np.ndarray, theta_norm: np.ndarray, cfg: DenormConfig):
    """
    与 test.py::get_line_ends / make_fusion_cache.py::calculate_radon_label 对齐：
      rho_real = rho_norm*(resize_h-1) - pad_top - diag/2
      theta_deg = theta_norm*180
    """
    w, h = cfg.unet_w, cfg.unet_h
    diag = math.sqrt(w*w + h*h)
    pad_top = (cfg.resize_h - diag) / 2.0
    rho_real = rho_norm * (cfg.resize_h - 1.0) - pad_top - (diag / 2.0)
    theta_deg = (theta_norm * cfg.angle_range_deg) % cfg.angle_range_deg
    return rho_real, theta_deg


def angular_diff_deg(a: np.ndarray, b: np.ndarray, period: float = 180.0) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


# =========================
# 几何：线段端点、距离、边界 y 误差
# =========================
def line_intersections_in_image(rho: float, theta_deg: float, w: int, h: int, eps: float = 1e-8) -> List[Tuple[float, float]]:
    """
    线方程（以图像中心为原点）：
      (x-cx)*cos + (y-cy)*sin = rho
    返回与图像边界的交点（在图像内部）。
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    pts: List[Tuple[float, float]] = []

    # x = 0
    if abs(sin_t) > eps:
        y = cy + (rho - ((0 - cx) * cos_t)) / sin_t
        if 0.0 <= y <= (h - 1.0):
            pts.append((0.0, float(y)))

    # x = w-1
    if abs(sin_t) > eps:
        x = w - 1.0
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if 0.0 <= y <= (h - 1.0):
            pts.append((x, float(y)))

    # y = 0
    if abs(cos_t) > eps:
        y = 0.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if 0.0 <= x <= (w - 1.0):
            pts.append((float(x), y))

    # y = h-1
    if abs(cos_t) > eps:
        y = h - 1.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if 0.0 <= x <= (w - 1.0):
            pts.append((float(x), y))

    # 去重（可能有重复点）
    uniq = []
    for p in pts:
        if all((abs(p[0]-q[0]) > 1e-6 or abs(p[1]-q[1]) > 1e-6) for q in uniq):
            uniq.append(p)
    return uniq


def farthest_pair(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    best = (pts[0], pts[1])
    best_d2 = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx*dx + dy*dy
            if d2 > best_d2:
                best_d2 = d2
                best = (pts[i], pts[j])
    return best


def endpoints_in_unet(rho: float, theta_deg: float, w: int, h: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = line_intersections_in_image(rho, theta_deg, w, h)
    if len(pts) >= 2:
        return farthest_pair(pts)
    # fallback：用左右边界 y
    y0, y1 = y_at_x(rho, theta_deg, 0.0, w, h), y_at_x(rho, theta_deg, w-1.0, w, h)
    y0 = float(np.clip(y0, 0, h-1)) if np.isfinite(y0) else h/2.0
    y1 = float(np.clip(y1, 0, h-1)) if np.isfinite(y1) else h/2.0
    return (0.0, y0), (w-1.0, y1)


def y_at_x(rho: float, theta_deg: float, x: float, w: int, h: int, eps: float = 1e-8) -> float:
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    if abs(sin_t) < eps:
        return float("nan")
    return cy + (rho - ((x - cx) * cos_t)) / sin_t


def edge_y_error_unet(rho_p: float, th_p: float, rho_g: float, th_g: float, w: int, h: int) -> float:
    """
    更直观的海天线误差：比较左右边界处 y 值的差异（UNet 尺度）。
    """
    ypl = y_at_x(rho_p, th_p, 0.0, w, h)
    ypr = y_at_x(rho_p, th_p, w-1.0, w, h)
    ygl = y_at_x(rho_g, th_g, 0.0, w, h)
    ygr = y_at_x(rho_g, th_g, w-1.0, w, h)
    vals = []
    for a, b in [(ypl, ygl), (ypr, ygr)]:
        if np.isfinite(a) and np.isfinite(b):
            vals.append(abs(a - b))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def mean_point_to_line_distance_unet(
    rho_pred: float,
    theta_pred_deg: float,
    rho_gt: float,
    theta_gt_deg: float,
    w: int,
    h: int,
    n_samples: int = 60,
) -> float:
    """在 GT 线上（落在图像内部的部分）采样点，计算点到预测线的平均垂直距离（UNet 尺度像素）。

    说明：
    - 旧实现是先求 GT 与图像边框的交点作为线段端点；当 GT 线因归一化/解码原因落在图外时，会触发裁剪回退，
      导致采样点不再是真正的 GT 线点，从而出现“rho/theta 很小但 lineDist 巨大”的不一致。
    - 新实现直接在 x=0..w-1 上采样并筛选落在 [0,h-1] 内的点；若 GT 线在整幅图内都不可见，则返回 NaN（后续统计会跳过）。
    """
    xs = np.linspace(0.0, w - 1.0, n_samples, dtype=np.float64)
    ys = np.array([y_at_x(rho_gt, theta_gt_deg, float(x), w, h) for x in xs], dtype=np.float64)

    valid = np.isfinite(ys) & (ys >= 0.0) & (ys <= (h - 1.0))
    if int(valid.sum()) < 2:
        return float('nan')

    xs = xs[valid]
    ys = ys[valid]

    theta_p = math.radians(theta_pred_deg)
    cos_p, sin_p = math.cos(theta_p), math.sin(theta_p)
    cx, cy = w / 2.0, h / 2.0

    x_c = xs - cx
    y_c = ys - cy
    d = np.abs(x_c * cos_p + y_c * sin_p - rho_pred)  # cos^2+sin^2=1
    return float(np.mean(d))


# =========================
# 汇总工具
# =========================
def summarize(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUT_DIR)
    vis_dir = os.path.join(OUT_DIR, f"outliers_{SPLIT}")
    ensure_dir(vis_dir)

    # 1) 读取 split cache
    split_dir = os.path.join(CACHE_ROOT, SPLIT)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split cache dir not found: {split_dir}")
    ds = SplitCacheDataset(split_dir)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=(DEVICE.startswith("cuda")))

    # 2) 读 GroundTruth.csv，用 idx 映射到文件名
    df = np.loadtxt(CSV_PATH, delimiter=",", dtype=str)
    # df: each row [img_name, x1, y1, x2, y2]
    # 注意：np.loadtxt 对逗号分隔和空格很敏感；如果你 CSV 有奇怪格式，建议改用 pandas.read_csv

    # 3) 加载模型
    model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=ANGLE_BINS).to(DEVICE)
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    cfg = DenormConfig(unet_w=UNET_IN_W, unet_h=UNET_IN_H, resize_h=RESIZE_H)

    # 4) 评估
    rows: List[Dict] = []
    rho_err_unet, rho_err_orig = [], []
    theta_err_deg = []
    line_dist_unet, line_dist_orig = [], []
    edgey_unet, edgey_orig = [], []

    with torch.no_grad():
        for xb, yb, idxb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            pred = model(xb)  # [B,2] in [0,1]
            pred_np = pred.detach().cpu().numpy()
            gt_np = yb.detach().cpu().numpy()
            idx_np = idxb.detach().cpu().numpy().astype(np.int64)

            rho_p, th_p = denorm_rho_theta(pred_np[:, 0], pred_np[:, 1], cfg)
            rho_g, th_g = denorm_rho_theta(gt_np[:, 0], gt_np[:, 1], cfg)

            e_theta = angular_diff_deg(th_p, th_g, period=180.0)
            e_rho_unet = np.abs(rho_p - rho_g)

            # 逐样本更复杂指标
            for i in range(len(idx_np)):
                idx = int(idx_np[i])

                # 读取原图尺寸（用于把 UNet 空间点缩放回原图）
                img_name = str(df[idx, 0])
                bgr, resolved_name = read_image_any_ext(IMG_DIR, img_name)
                if bgr is None:
                    # 没图也可以做“标签层面”的误差统计，但可视化会缺失
                    h0, w0 = 1080, 1920
                    resolved_name = img_name
                else:
                    h0, w0 = bgr.shape[:2]

                sx = w0 / float(UNET_IN_W)
                sy = h0 / float(UNET_IN_H)

                # line distance（UNet 空间）
                ld_u = mean_point_to_line_distance_unet(
                    rho_pred=float(rho_p[i]),
                    theta_pred_deg=float(th_p[i]),
                    rho_gt=float(rho_g[i]),
                    theta_gt_deg=float(th_g[i]),
                    w=UNET_IN_W,
                    h=UNET_IN_H,
                    n_samples=LINE_SAMPLE_N,
                )
                # edge y error（UNet 空间）
                ey_u = edge_y_error_unet(
                    rho_p=float(rho_p[i]),
                    th_p=float(th_p[i]),
                    rho_g=float(rho_g[i]),
                    th_g=float(th_g[i]),
                    w=UNET_IN_W,
                    h=UNET_IN_H,
                )

                # 转原图尺度（点缩放：非均匀也支持）
                # 对于“距离类指标”，如果 sx≈sy，可以用平均比例；否则用几何平均做近似
                s_dist = math.sqrt(sx * sy)

                rho_err_u = float(e_rho_unet[i])
                rho_err_o = float(rho_err_u * s_dist)  # 近似尺度变换（海天线数据通常等比缩放）

                ld_o = float(ld_u * s_dist)
                ey_o = float(ey_u * s_dist) if np.isfinite(ey_u) else float("nan")

                rho_err_unet.append(rho_err_u)
                rho_err_orig.append(rho_err_o)
                theta_err_deg.append(float(e_theta[i]))
                line_dist_unet.append(float(ld_u))
                line_dist_orig.append(float(ld_o))
                edgey_unet.append(float(ey_u))
                edgey_orig.append(float(ey_o))

                rows.append({
                    "idx": idx,
                    "img_name": resolved_name,
                    "rho_gt_norm": float(gt_np[i, 0]),
                    "theta_gt_norm": float(gt_np[i, 1]),
                    "rho_pred_norm": float(pred_np[i, 0]),
                    "theta_pred_norm": float(pred_np[i, 1]),
                    "rho_err_px_unet": rho_err_u,
                    "rho_err_px_orig": rho_err_o,
                    "theta_err_deg": float(e_theta[i]),
                    "line_dist_px_unet": float(ld_u),
                    "line_dist_px_orig": float(ld_o),
                    "edgey_err_px_unet": float(ey_u),
                    "edgey_err_px_orig": float(ey_o),
                })

    # 5) 打印汇总
    rho_err_unet = np.asarray(rho_err_unet, dtype=np.float64)
    rho_err_orig = np.asarray(rho_err_orig, dtype=np.float64)
    theta_err_deg = np.asarray(theta_err_deg, dtype=np.float64)
    line_dist_unet = np.asarray(line_dist_unet, dtype=np.float64)
    line_dist_orig = np.asarray(line_dist_orig, dtype=np.float64)
    edgey_unet = np.asarray(edgey_unet, dtype=np.float64)
    edgey_orig = np.asarray(edgey_orig, dtype=np.float64)

    n_line_unet_nan = int(np.sum(~np.isfinite(line_dist_unet)))
    n_line_orig_nan = int(np.sum(~np.isfinite(line_dist_orig)))

    print("========== Fusion-CNN Evaluation ==========")
    print(f"Split: {SPLIT} | N={len(ds)} | Device={DEVICE}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Cache:   {split_dir}")
    print("")

    def pr(name, arr):
        s = summarize(arr)
        print(f"{name}: mean={s['mean']:.4f}, median={s['median']:.4f}, p90={s['p90']:.4f}, p95={s['p95']:.4f}, max={s['max']:.4f}")

    pr(f"Rho abs error (px, UNet {UNET_IN_W}x{UNET_IN_H})", rho_err_unet)
    pr("Rho abs error (px, original-scale approx)", rho_err_orig)
    pr("Theta error (deg, wrap-aware, period=180)", theta_err_deg)
    pr("Mean point->line distance (px, UNet)", line_dist_unet)
    pr("Mean point->line distance (px, original-scale approx)", line_dist_orig)
    if n_line_unet_nan > 0 or n_line_orig_nan > 0:
        print(f"[Note] lineDist skipped NaN samples: UNet={n_line_unet_nan}, orig={n_line_orig_nan}")
    pr("Edge-Y error (px, UNet)", edgey_unet)
    pr("Edge-Y error (px, original-scale approx)", edgey_orig)
    print("")

    def pct_le(arr, thr):
        arr = arr[np.isfinite(arr)]
        return 100.0 * float(np.mean(arr <= thr)) if arr.size else float("nan")

    print("---- Threshold stats (helpful for论文表格) ----")
    print(f"Theta <=1°: {pct_le(theta_err_deg,1):.2f}% | <=2°: {pct_le(theta_err_deg,2):.2f}% | <=5°: {pct_le(theta_err_deg,5):.2f}%")
    print(f"Rho(orig) <=5px: {pct_le(rho_err_orig,5):.2f}% | <=10px: {pct_le(rho_err_orig,10):.2f}% | <=20px: {pct_le(rho_err_orig,20):.2f}%")
    print(f"LineDist(orig) <=5px: {pct_le(line_dist_orig,5):.2f}% | <=10px: {pct_le(line_dist_orig,10):.2f}% | <=20px: {pct_le(line_dist_orig,20):.2f}%")
    print(f"EdgeY(orig) <=5px: {pct_le(edgey_orig,5):.2f}% | <=10px: {pct_le(edgey_orig,10):.2f}% | <=20px: {pct_le(edgey_orig,20):.2f}%")

    # 6) 保存 CSV
    if SAVE_CSV:
        out_csv = os.path.join(OUT_DIR, CSV_NAME)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] per-sample metrics -> {out_csv}")

    # 7) 选 outliers：阈值 + topK
    # 先阈值
    outlier_idx = set()
    for r in rows:
        if (r["theta_err_deg"] > THRESH_THETA_DEG or
            r["rho_err_px_orig"] > THRESH_RHO_PX_ORIG or
            r["line_dist_px_orig"] > THRESH_LINE_DIST_PX_ORIG or
            (np.isfinite(r["edgey_err_px_orig"]) and r["edgey_err_px_orig"] > THRESH_EDGEY_PX_ORIG)):
            outlier_idx.add(r["idx"])

    # 再 topK：按 line_dist_px_orig
    rows_sorted = sorted(rows, key=lambda d: d["line_dist_px_orig"], reverse=True)
    for r in rows_sorted[:min(OUTLIER_TOPK, len(rows_sorted))]:
        outlier_idx.add(r["idx"])

    outlier_rows = [r for r in rows_sorted if r["idx"] in outlier_idx]
    print(f"\n[Outliers] selected={len(outlier_rows)} (threshold-or-topK)")
    print(f"[Outliers] saving visualization to: {vis_dir}")

    # 8) 可视化保存
    for r in outlier_rows:
        idx = r["idx"]
        img_name = r["img_name"]
        bgr, resolved_name = read_image_any_ext(IMG_DIR, img_name)
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]

        # 反归一化得到 UNet 空间 rho/theta
        rho_g, th_g = denorm_rho_theta(np.array([r["rho_gt_norm"]]), np.array([r["theta_gt_norm"]]), cfg)
        rho_p, th_p = denorm_rho_theta(np.array([r["rho_pred_norm"]]), np.array([r["theta_pred_norm"]]), cfg)
        rho_g, th_g = float(rho_g[0]), float(th_g[0])
        rho_p, th_p = float(rho_p[0]), float(th_p[0])

        # UNet 空间端点（在 1024x576 内）
        (gx0, gy0), (gx1, gy1) = endpoints_in_unet(rho_g, th_g, UNET_IN_W, UNET_IN_H)
        (px0, py0), (px1, py1) = endpoints_in_unet(rho_p, th_p, UNET_IN_W, UNET_IN_H)

        # 缩放到原图
        sx = w0 / float(UNET_IN_W)
        sy = h0 / float(UNET_IN_H)
        g0 = (int(round(gx0 * sx)), int(round(gy0 * sy)))
        g1 = (int(round(gx1 * sx)), int(round(gy1 * sy)))
        p0 = (int(round(px0 * sx)), int(round(py0 * sy)))
        p1 = (int(round(px1 * sx)), int(round(py1 * sy)))

        vis = bgr.copy()
        cv2.line(vis, g0, g1, COLOR_GT, THICKNESS)
        cv2.line(vis, p0, p1, COLOR_PR, THICKNESS)

        # 文本：误差
        text1 = f"idx={idx}  theta_err={r['theta_err_deg']:.2f}deg  rho_err~{r['rho_err_px_orig']:.1f}px"
        text2 = f"lineDist~{r['line_dist_px_orig']:.1f}px  edgeY~{r['edgey_err_px_orig']:.1f}px"
        cv2.putText(vis, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(vis, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(vis, "GT: green   Pred: red", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

        # 保存：尽量用原文件名（resolved_name），输出到独立目录
        out_path = os.path.join(vis_dir, resolved_name if resolved_name else os.path.basename(img_name))
        # 如果没有扩展名，强制 .jpg
        root, ext = os.path.splitext(out_path)
        if ext == "":
            out_path = root + ".jpg"
        ensure_dir(os.path.dirname(out_path))
        cv2.imwrite(out_path, vis)

    # 9) 另存一个 outlier 列表
    out_txt = os.path.join(OUT_DIR, f"outliers_{SPLIT}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for r in outlier_rows:
            f.write(f"{r['idx']}\t{r['img_name']}\t"
                    f"theta={r['theta_err_deg']:.3f}\t"
                    f"rho~{r['rho_err_px_orig']:.2f}\t"
                    f"lineDist~{r['line_dist_px_orig']:.2f}\t"
                    f"edgeY~{r['edgey_err_px_orig']:.2f}\n")
    print(f"[Saved] outlier list -> {out_txt}")
    print("Done.")


if __name__ == "__main__":
    main()
