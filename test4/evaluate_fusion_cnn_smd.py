"""
Evaluate Fusion-CNN on SMD FusionCache (Experiment A: zero-shot generalization).

This is a SMD-specialized wrapper around evaluate_fusion_cnn.py with:
  - sensible defaults pointing to ./test4/FusionCache_SMD_1024x576
  - per-domain breakdown: NIR / VIS_Onboard / VIS_Onshore (inferred from cached img_name)

Assumptions:
  - You already ran:
      python test4/prepare_smd_testset.py
      python test4/make_fusion_cache_smd.py
  - Cached .npy contains keys: input, label, img_name

Run (from project root):
  python test4/evaluate_fusion_cnn_smd.py \
    --weights splits_musid/best_fusion_cnn_1024x576.pth \
    --cache_root test4/FusionCache_SMD_1024x576 \
    --split test \
    --out_csv test4/eval_smd_test_per_sample.csv
"""

import argparse
import os
import sys
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入 cnn_model，如果直接运行脚本找不到模块，尝试添加父目录到 sys.path
try:
    from cnn_model import HorizonResNet
except ImportError:
    # 如果当前脚本位于 test4 下，而 cnn_model 在根目录，需要将根目录加入路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    from cnn_model import HorizonResNet


# -------------------------
# Cache dataset
# -------------------------
class SplitCacheDataset(Dataset):
    """Loads cache files produced by make_fusion_cache_smd.py.

    Each {idx}.npy is a dict with keys:
      - input: (4, 2240, 180) float32
      - label: (2,) [rho_norm, theta_norm] float32
      - img_name: str (e.g., 'NIR__MVI_1478_NIR__000123.jpg')
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        # 确保目录存在，否则列出文件会报错
        if not os.path.isdir(cache_dir):
            # 这里留空或抛出异常均可，外层已做检查
            self.files = []
        else:
            self.files = sorted(
                [f for f in os.listdir(cache_dir) if f.endswith(".npy")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        fn = self.files[i]
        idx = int(os.path.splitext(fn)[0])
        path = os.path.join(self.cache_dir, fn)
        data = np.load(path, allow_pickle=True).item()
        x = torch.from_numpy(data["input"]).float()
        y = torch.from_numpy(data["label"]).float()
        img_name = str(data.get("img_name", ""))
        return x, y, idx, img_name


# -------------------------
# Param denormalization
# -------------------------
@dataclass
class DenormConfig:
    unet_w: int = 1024
    unet_h: int = 576
    resize_h: int = 2240
    angle_range_deg: float = 180.0
    orig_w: int = 1920
    orig_h: int = 1080


def denorm_rho_theta(rho_norm: np.ndarray, theta_norm: np.ndarray, cfg: DenormConfig) -> Tuple[np.ndarray, np.ndarray]:
    w, h = cfg.unet_w, cfg.unet_h
    diag = math.sqrt(w * w + h * h)
    pad_top = (cfg.resize_h - diag) / 2.0

    final_rho_idx = rho_norm * (cfg.resize_h - 1.0)
    rho_real = final_rho_idx - pad_top - (diag / 2.0)

    theta_deg = (theta_norm * cfg.angle_range_deg) % cfg.angle_range_deg
    return rho_real, theta_deg


def angular_diff_deg(a: np.ndarray, b: np.ndarray, period: float = 180.0) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


# -------------------------
# Line distance metric
# -------------------------
def line_intersections_in_image(
    rho: float,
    theta_deg: float,
    w: int,
    h: int,
    eps: float = 1e-8,
) -> List[Tuple[float, float]]:
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    pts: List[Tuple[float, float]] = []

    # x = 0
    if abs(sin_t) > eps:
        y = cy + (rho - ((0 - cx) * cos_t)) / sin_t
        if -1 <= y <= h:
            pts.append((0.0, float(y)))

    # x = w-1
    if abs(sin_t) > eps:
        x = w - 1.0
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if -1 <= y <= h:
            pts.append((x, float(y)))

    # y = 0
    if abs(cos_t) > eps:
        y = 0.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if -1 <= x <= w:
            pts.append((float(x), y))

    # y = h-1
    if abs(cos_t) > eps:
        y = h - 1.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if -1 <= x <= w:
            pts.append((float(x), y))

    pts2 = []
    for x, y in pts:
        if 0.0 <= x <= (w - 1.0) and 0.0 <= y <= (h - 1.0):
            pts2.append((x, y))
    return pts2


def farthest_pair(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    best = (pts[0], pts[1])
    best_d2 = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                best = (pts[i], pts[j])
    return best


def mean_point_to_line_distance(
    rho_pred: float,
    theta_pred_deg: float,
    rho_gt: float,
    theta_gt_deg: float,
    w: int,
    h: int,
    n_samples: int = 50,
) -> float:
    pts = line_intersections_in_image(rho_gt, theta_gt_deg, w, h)
    if len(pts) >= 2:
        p0, p1 = farthest_pair(pts)
    else:
        theta = math.radians(theta_gt_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = w / 2.0, h / 2.0
        x0, x1 = 0.0, w - 1.0
        if abs(sin_t) < 1e-8:
            y0 = y1 = cy
        else:
            y0 = cy + (rho_gt - ((x0 - cx) * cos_t)) / sin_t
            y1 = cy + (rho_gt - ((x1 - cx) * cos_t)) / sin_t
        y0 = float(np.clip(y0, 0, h - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        p0, p1 = (x0, y0), (x1, y1)

    xs = np.linspace(p0[0], p1[0], n_samples)
    ys = np.linspace(p0[1], p1[1], n_samples)

    theta_p = math.radians(theta_pred_deg)
    cos_p, sin_p = math.cos(theta_p), math.sin(theta_p)
    cx, cy = w / 2.0, h / 2.0

    x_c = xs - cx
    y_c = ys - cy
    d = np.abs(x_c * cos_p + y_c * sin_p - rho_pred)
    return float(np.mean(d))


# -------------------------
# Reporting helpers
# -------------------------
def summarize(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return f"{name}: N=0"
    return (
        f"{name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}, max={arr.max():.4f}"
    )


def domain_from_img_name(img_name: str) -> str:
    # created by test4/prepare_smd_testset.py: <DOMAIN>__<video_stem>__<frame>.jpg
    for d in ("NIR", "VIS_Onboard", "VIS_Onshore"):
        if img_name.startswith(d + "__"):
            return d
    # fallback
    return "Unknown"


def pct_le(arr: np.ndarray, thr: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


def main():
    # -------------------------------------------------------------
    # [FIX] 动态获取路径，解决 FileNotFoundError 问题
    # -------------------------------------------------------------
    # 获取当前脚本所在目录 (例如 .../sealine_detection/test4)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (例如 .../sealine_detection)
    project_root = os.path.dirname(script_dir)

    # 构造绝对路径的默认值
    default_cache_dir = os.path.join(script_dir, "FusionCache_SMD_1024x576")
    default_weights_path = os.path.join(project_root, "splits_musid", "best_fusion_cnn_1024x576.pth")
    default_out_csv = os.path.join(script_dir, "eval_smd_test_per_sample.csv")

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=default_weights_path)
    ap.add_argument("--cache_root", type=str, default=default_cache_dir)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default=default_out_csv)

    # denorm config
    ap.add_argument("--unet_w", type=int, default=1024)
    ap.add_argument("--unet_h", type=int, default=576)
    ap.add_argument("--resize_h", type=int, default=2240)
    ap.add_argument("--orig_w", type=int, default=1920)
    ap.add_argument("--orig_h", type=int, default=1080)
    ap.add_argument("--line_samples", type=int, default=50)

    args = ap.parse_args()

    cfg = DenormConfig(
        unet_w=args.unet_w,
        unet_h=args.unet_h,
        resize_h=args.resize_h,
        orig_w=args.orig_w,
        orig_h=args.orig_h,
    )
    scale = cfg.orig_w / cfg.unet_w

    split_dir = os.path.join(args.cache_root, args.split)
    if not os.path.isdir(split_dir):
        # 如果找不到 split 目录，打印调试信息
        print(f"[Error] Directory not found: {split_dir}")
        print(f"  - args.cache_root: {args.cache_root}")
        print(f"  - script_dir: {script_dir}")
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    ds = SplitCacheDataset(split_dir)
    if len(ds) == 0:
        print(f"[Warning] No .npy files found in {split_dir}")
        return

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    # 检查权重文件是否存在
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    model = HorizonResNet(in_channels=4, img_h=cfg.resize_h, img_w=180).to(args.device)
    ckpt = torch.load(args.weights, map_location=args.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    # global arrays
    g_rho_err = []
    g_rho_err_orig = []
    g_theta_err = []
    g_line_dist = []

    # per-domain
    per: Dict[str, Dict[str, list]] = {
        "NIR": {"rho": [], "rho_o": [], "theta": [], "line": []},
        "VIS_Onboard": {"rho": [], "rho_o": [], "theta": [], "line": []},
        "VIS_Onshore": {"rho": [], "rho_o": [], "theta": [], "line": []},
        "Unknown": {"rho": [], "rho_o": [], "theta": [], "line": []},
    }

    rows = []

    with torch.no_grad():
        for xb, yb, idxb, names in dl:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)

            pred = model(xb)
            pred_np = pred.detach().cpu().numpy()
            gt_np = yb.detach().cpu().numpy()
            idx_np = idxb.detach().cpu().numpy()
            names = list(names)

            rho_p, th_p = denorm_rho_theta(pred_np[:, 0], pred_np[:, 1], cfg)
            rho_g, th_g = denorm_rho_theta(gt_np[:, 0], gt_np[:, 1], cfg)

            e_rho = np.abs(rho_p - rho_g)
            e_theta = angular_diff_deg(th_p, th_g, period=180.0)

            for i in range(len(idx_np)):
                ld = mean_point_to_line_distance(
                    rho_pred=float(rho_p[i]),
                    theta_pred_deg=float(th_p[i]),
                    rho_gt=float(rho_g[i]),
                    theta_gt_deg=float(th_g[i]),
                    w=cfg.unet_w,
                    h=cfg.unet_h,
                    n_samples=args.line_samples,
                )

                dom = domain_from_img_name(names[i])

                # global
                g_rho_err.append(float(e_rho[i]))
                g_rho_err_orig.append(float(e_rho[i] * scale))
                g_theta_err.append(float(e_theta[i]))
                g_line_dist.append(float(ld))

                # per-domain
                per.setdefault(dom, {"rho": [], "rho_o": [], "theta": [], "line": []})
                per[dom]["rho"].append(float(e_rho[i]))
                per[dom]["rho_o"].append(float(e_rho[i] * scale))
                per[dom]["theta"].append(float(e_theta[i]))
                per[dom]["line"].append(float(ld))

                if args.out_csv:
                    rows.append(
                        {
                            "idx": int(idx_np[i]),
                            "img_name": names[i],
                            "domain": dom,
                            "rho_gt_norm": float(gt_np[i, 0]),
                            "theta_gt_norm": float(gt_np[i, 1]),
                            "rho_pred_norm": float(pred_np[i, 0]),
                            "theta_pred_norm": float(pred_np[i, 1]),
                            "rho_err_px_unet": float(e_rho[i]),
                            "rho_err_px_orig": float(e_rho[i] * scale),
                            "theta_err_deg": float(e_theta[i]),
                            "line_dist_px_unet": float(ld),
                        }
                    )

    # to arrays
    g_rho_err = np.asarray(g_rho_err, dtype=np.float64)
    g_rho_err_orig = np.asarray(g_rho_err_orig, dtype=np.float64)
    g_theta_err = np.asarray(g_theta_err, dtype=np.float64)
    g_line_dist = np.asarray(g_line_dist, dtype=np.float64)

    print("========== SMD Evaluation (Fusion-CNN) ==========")
    print(f"Split: {args.split} | N={len(ds)}")
    print(f"Weights: {args.weights}")
    print(f"Cache:   {split_dir}")
    print("")

    print("[Overall]")
    print(summarize("Rho abs error (px, UNet space)", g_rho_err))
    print(summarize(f"Rho abs error (px, original ~{cfg.orig_w}x{cfg.orig_h})", g_rho_err_orig))
    print(summarize("Theta error (deg, wrap-aware)", g_theta_err))
    print(summarize("Mean point->line distance (px, UNet space)", g_line_dist))
    print("---- Overall thresholds ----")
    print(f"theta <= 1°:  {pct_le(g_theta_err, 1):.2f}% | <=2°: {pct_le(g_theta_err, 2):.2f}% | <=5°: {pct_le(g_theta_err, 5):.2f}%")
    print(f"rho_orig <= 5px: {pct_le(g_rho_err_orig, 5):.2f}% | <=10px: {pct_le(g_rho_err_orig, 10):.2f}% | <=20px: {pct_le(g_rho_err_orig, 20):.2f}%")
    print(f"line_dist <= 5px: {pct_le(g_line_dist, 5):.2f}% | <=10px: {pct_le(g_line_dist, 10):.2f}% | <=20px: {pct_le(g_line_dist, 20):.2f}%")

    # per-domain report
    print("\n[Per-domain breakdown]")
    for dom in ["NIR", "VIS_Onboard", "VIS_Onshore", "Unknown"]:
        arr_rho = np.asarray(per.get(dom, {}).get("rho", []), dtype=np.float64)
        arr_rho_o = np.asarray(per.get(dom, {}).get("rho_o", []), dtype=np.float64)
        arr_theta = np.asarray(per.get(dom, {}).get("theta", []), dtype=np.float64)
        arr_line = np.asarray(per.get(dom, {}).get("line", []), dtype=np.float64)

        if arr_rho.size == 0:
            continue
        print(f"\n--- {dom} | N={arr_rho.size} ---")
        print(summarize("Rho abs error (px, UNet space)", arr_rho))
        print(summarize(f"Rho abs error (px, original ~{cfg.orig_w}x{cfg.orig_h})", arr_rho_o))
        print(summarize("Theta error (deg)", arr_theta))
        print(summarize("Mean point->line distance (px, UNet space)", arr_line))
        print(f"theta<=2°: {pct_le(arr_theta, 2):.2f}% | rho_orig<=10px: {pct_le(arr_rho_o, 10):.2f}% | line_dist<=10px: {pct_le(arr_line, 10):.2f}%")

    if args.out_csv and rows:
        out_path = args.out_csv
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] per-sample metrics -> {out_path}")


if __name__ == "__main__":
    main()