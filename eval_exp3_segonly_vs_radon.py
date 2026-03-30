# -*- coding: utf-8 -*-
"""
eval_exp3_segonly_vs_radon.py

实验3：UNet seg-only（快速版） vs UNet + Radon + CNN（完整流水线）
在 MU-SID 测试集（268张）上对比 VE/AE 精度和推理速度。

用法：
    python eval_exp3_segonly_vs_radon.py                         # 跑全部268张
    python eval_exp3_segonly_vs_radon.py --quick 10              # 只跑前10张
    python eval_exp3_segonly_vs_radon.py --device cpu             # CPU 模式
    python eval_exp3_segonly_vs_radon.py --img-root /path/to/MU-SID  # 自定义图片路径
"""

import os
import sys
import csv
import math
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

# ---- 项目路径 ----
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "method2_unet_radon"))

import torch
import torch.nn.functional as F

# 导入项目模块
from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT
from cnn_model import HorizonResNet


# ======================================================================
#  辅助函数
# ======================================================================

def safe_load_state_dict(ckpt_path, device="cpu"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_gt(csv_path):
    """加载 GT CSV，返回 list of dict。"""
    gt_list = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            stem = row[0].strip()
            x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            gt_list.append({
                "stem": stem,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return gt_list


def compute_gt_metrics(gt):
    """从 GT 端点计算 y_center 和 angle（原图1920x1080坐标系）。"""
    x1, y1, x2, y2 = gt["x1"], gt["y1"], gt["x2"], gt["y2"]
    W = 1920
    xc = (W - 1) / 2.0
    # y_at_center 通过线性插值
    if abs(x2 - x1) > 1e-6:
        slope = (y2 - y1) / (x2 - x1)
        y_center = y1 + slope * (xc - x1)
    else:
        y_center = (y1 + y2) / 2.0
    # angle
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return y_center, angle


def compute_pred_metrics(pred_line):
    """从预测线端点计算 y_center 和 angle（原图1920x1080坐标系）。"""
    x1, y1, x2, y2 = pred_line
    W = 1920
    xc = (W - 1) / 2.0
    if abs(x2 - x1) > 1e-6:
        slope = (y2 - y1) / (x2 - x1)
        y_center = y1 + slope * (xc - x1)
    else:
        y_center = (y1 + y2) / 2.0
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return y_center, angle


def wrap_angle_diff(a, b):
    """角度差 wrap 到 [-90, 90]。"""
    d = a - b
    while d > 90:
        d -= 180
    while d < -90:
        d += 180
    return d


def find_image_path(img_root, stem):
    """尝试 .jpg 和 .JPG 两种扩展名。"""
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        p = os.path.join(img_root, stem + ext)
        if os.path.isfile(p):
            return p
    return None


# ======================================================================
#  seg-only 海天线提取（来自 app_fast/inference_engine_fast.py）
# ======================================================================

def horizon_from_segmask(seg_mask, device_str="cpu"):
    """
    从分割 mask 中提取海天线。
    seg_mask: numpy (H, W)，sky=1/sea=0
    返回 (0, y1, w-1, y2) 在 seg_mask 空间的坐标，或 None。
    """
    h, w = seg_mask.shape
    seg_tensor = torch.from_numpy(seg_mask.astype(np.float32)).to(device_str)

    num_samples = 32
    cols = torch.linspace(0, w - 1, num_samples, device=seg_tensor.device).long()
    sampled = seg_tensor[:, cols]  # (H, 32)
    diff = sampled[1:, :] - sampled[:-1, :]
    has_transition = (diff != 0).float()
    first_trans = has_transition.argmax(dim=0)
    any_trans = has_transition.any(dim=0)

    cols_np = cols.cpu().numpy().astype(np.float64)
    first_trans_np = first_trans.cpu().numpy().astype(np.float64)
    any_trans_np = any_trans.cpu().numpy()
    valid = any_trans_np.astype(bool)

    if valid.sum() < 4:
        return None

    x_pts, y_pts = cols_np[valid], first_trans_np[valid]
    coeffs = np.polyfit(x_pts, y_pts, 1)
    slope, intercept = coeffs
    residuals = np.abs(y_pts - (slope * x_pts + intercept))
    inlier = residuals < 5.0
    if inlier.sum() >= 2:
        coeffs = np.polyfit(x_pts[inlier], y_pts[inlier], 1)
        slope, intercept = coeffs

    y1 = slope * 0 + intercept
    y2 = slope * (w - 1) + intercept
    return (0, y1, w - 1, y2)


# ======================================================================
#  完整流水线辅助
# ======================================================================

def pad_sinogram(sino, target_h, target_w):
    """零填充居中（和 make_fusion_cache / inference_engine 一致）。"""
    sino = sino.astype(np.float32)
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6:
        sino = (sino - mi) / (ma - mi)
    else:
        sino = np.zeros_like(sino, dtype=np.float32)
    h_curr = sino.shape[0]
    container = np.zeros((target_h, target_w), dtype=np.float32)
    start_h = (target_h - h_curr) // 2
    if h_curr <= target_h:
        container[start_h:start_h + h_curr, :] = sino
    else:
        crop_start = (h_curr - target_h) // 2
        container[:, :] = sino[crop_start:crop_start + target_h, :]
    return container


def postprocess_mask(mask_np):
    """连通域后处理：只保留触顶的 sky 区域。"""
    sky = (mask_np == 1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_TOP] <= 0:
            keep[labels == i] = 1
    out = mask_np.copy()
    out[(mask_np == 1) & (keep == 0)] = 0
    return out


def rho_theta_to_line(rho_norm, theta_norm, img_w, img_h):
    """将归一化的 (rho, theta) 转为图像坐标线段端点。"""
    diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
    radon_h = 2240
    pad = (radon_h - diagonal) / 2
    rho_px = rho_norm * (radon_h - 1) - pad - diagonal / 2
    theta_deg = theta_norm * 180.0
    theta_rad = math.radians(theta_deg)

    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    cx, cy = img_w / 2, img_h / 2
    if abs(sin_t) > 1e-6:
        y_at_0 = (rho_px - (0 - cx) * cos_t) / sin_t + cy
        y_at_w = (rho_px - (img_w - cx) * cos_t) / sin_t + cy
        return (0, y_at_0, img_w, y_at_w)
    return None


# ======================================================================
#  主评估流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="实验3：seg-only vs 完整流水线 对比评估")
    parser.add_argument("--device", default="cuda", help="推理设备 (default: cuda)")
    parser.add_argument("--quick", type=int, default=None, help="只跑前 N 张图做快速验证")
    parser.add_argument("--img-root", default="/home/jetson/Documents/Hashmani_Dataset/MU-SID",
                        help="MU-SID 图像目录")
    parser.add_argument("--gt-csv", default=None, help="GT CSV 路径")
    parser.add_argument("--unet-weights", default=None, help="UNet 权重路径")
    parser.add_argument("--cnn-weights", default=None, help="CNN 权重路径")
    args = parser.parse_args()

    project_root = _SCRIPT_DIR

    # 默认路径
    gt_csv = args.gt_csv or str(project_root / "splits_musid" / "GroundTruth_test.csv")
    unet_weights = args.unet_weights or str(project_root / "weights" / "rghnet_best_c2.pth")
    cnn_weights = args.cnn_weights or str(project_root / "weights" / "best_fusion_cnn_4ch.pth")
    img_root = args.img_root

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- 加载 GT ----
    gt_list = load_gt(gt_csv)
    print(f"[INFO] GT 样本数: {len(gt_list)}")

    if args.quick is not None:
        gt_list = gt_list[:args.quick]
        print(f"[INFO] Quick 模式: 只评估前 {args.quick} 张")

    # ---- 加载 UNet（seg-only 用 512x288，完整流水线用 1024x576） ----
    print("[INFO] 加载 UNet ...")
    unet = RestorationGuidedHorizonNet(
        num_classes=2,
        dce_weights_path=None,
        require_dce=False,
    ).to(device)
    state = safe_load_state_dict(unet_weights, str(device))
    unet.load_state_dict(state, strict=False)
    unet.dce_net = None  # 关掉 Zero-DCE
    unet.eval()
    unet.half()  # FP16 (与部署一致)
    print("[INFO] UNet 加载完成 (FP16)")

    # ---- 加载 CNN ----
    print("[INFO] 加载 CNN ...")
    cnn = HorizonResNet(in_channels=4).to(device)
    state = safe_load_state_dict(cnn_weights, str(device))
    cnn.load_state_dict(state, strict=False)
    cnn.eval()
    print("[INFO] CNN 加载完成")

    # ---- 加载 Radon ----
    print("[INFO] 初始化 Radon 变换 ...")
    import matplotlib
    matplotlib.use("Agg")
    radon_engine = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

    # ---- 分辨率常量 ----
    SEGONLY_W, SEGONLY_H = 512, 288
    FULL_W, FULL_H = 1024, 576
    ORIG_W, ORIG_H = 1920, 1080
    RADON_H, RADON_W = 2240, 180

    # ---- Warmup (消除 CUDA/cudnn 首次运行开销) ----
    print("[INFO] Warmup ...")
    with torch.no_grad():
        dummy_small = torch.randn(1, 3, SEGONLY_H, SEGONLY_W, device=device, dtype=torch.float16)
        dummy_full = torch.randn(1, 3, FULL_H, FULL_W, device=device, dtype=torch.float16)
        for _ in range(3):
            unet(dummy_small, target=None, enable_restoration=False, enable_segmentation=True)
            unet(dummy_full, target=None, enable_restoration=True, enable_segmentation=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    print("[INFO] Warmup 完成")

    # ---- 评估 ----
    results = []  # 逐样本结果
    segonly_times = []
    full_times = []
    n_segonly_fail = 0
    n_full_fail = 0

    total = len(gt_list)
    print(f"\n[INFO] 开始评估 {total} 张图像 ...\n")

    for idx, gt in enumerate(gt_list):
        stem = gt["stem"]
        img_path = find_image_path(img_root, stem)
        if img_path is None:
            print(f"  [{idx+1}/{total}] {stem} - 图像文件未找到，跳过")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [{idx+1}/{total}] {stem} - 图像读取失败，跳过")
            continue

        # GT 指标
        gt_y_center, gt_angle = compute_gt_metrics(gt)

        # ================================================================
        #  (a) seg-only 快速版
        # ================================================================
        img_small = cv2.resize(img_bgr, (SEGONLY_W, SEGONLY_H))
        img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        tensor_small = torch.from_numpy(img_rgb_small).permute(2, 0, 1).unsqueeze(0).to(
            device=device, dtype=torch.float16) * (1.0 / 255.0)

        t0 = time.time()
        with torch.no_grad():
            _, seg_logits_small, _ = unet(
                tensor_small, target=None,
                enable_restoration=False, enable_segmentation=True
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

        seg_mask_small = torch.softmax(seg_logits_small, dim=1).argmax(dim=1)[0].cpu().numpy()
        line_seg = horizon_from_segmask(seg_mask_small, str(device))
        t_segonly = (time.time() - t0) * 1000
        segonly_times.append(t_segonly)

        # 缩放到原图坐标
        ve_segonly = np.nan
        ae_segonly = np.nan
        if line_seg is not None:
            sx = ORIG_W / SEGONLY_W
            sy = ORIG_H / SEGONLY_H
            pred_orig = (line_seg[0] * sx, line_seg[1] * sy, line_seg[2] * sx, line_seg[3] * sy)
            pred_y_center, pred_angle = compute_pred_metrics(pred_orig)
            ve_segonly = abs(pred_y_center - gt_y_center)
            ae_segonly = abs(wrap_angle_diff(pred_angle, gt_angle))
        else:
            n_segonly_fail += 1

        # ================================================================
        #  (b) 完整流水线：UNet(1024x576) + Radon + CNN
        # ================================================================
        img_full = cv2.resize(img_bgr, (FULL_W, FULL_H))
        img_rgb_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        tensor_full = torch.from_numpy(img_rgb_full).permute(2, 0, 1).unsqueeze(0).to(
            device=device, dtype=torch.float16) * (1.0 / 255.0)

        t0 = time.time()
        with torch.no_grad():
            restored, seg_logits_full, _ = unet(
                tensor_full, target=None,
                enable_restoration=True, enable_segmentation=True
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

        seg_mask_full = torch.softmax(seg_logits_full, dim=1).argmax(dim=1)[0].cpu().numpy()

        # Stage B: Gradient-Radon
        restored_np = (restored[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

        ve_full = np.nan
        ae_full = np.nan
        try:
            _, _, _, sinograms = radon_engine.detect(restored_bgr)

            radon_features = np.zeros((3, RADON_H, RADON_W), dtype=np.float32)
            for i, sino in enumerate(sinograms):
                if sino is not None and sino.size > 0:
                    radon_features[i] = pad_sinogram(sino, RADON_H, RADON_W)

            # 第4通道：seg mask 边缘 Radon
            seg_mask_pp = postprocess_mask(seg_mask_full)
            seg_edge = cv2.Canny((seg_mask_pp * 255).astype(np.uint8), 50, 150)
            k = np.ones((3, 3), np.uint8)
            seg_edge = cv2.dilate(seg_edge, k, iterations=1)
            seg_sino = radon_engine._radon_gpu(seg_edge.astype(np.float32), radon_engine.theta)
            radon_features_seg = pad_sinogram(seg_sino, RADON_H, RADON_W)

            cnn_input = np.concatenate([radon_features, radon_features_seg[np.newaxis]], axis=0)  # (4, 2240, 180)
            cnn_tensor = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device)

            # Stage C: CNN
            with torch.no_grad():
                pred = cnn(cnn_tensor, return_conf=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

            rho_norm, theta_norm = pred[0].cpu().numpy()
            horizon_line = rho_theta_to_line(rho_norm, theta_norm, FULL_W, FULL_H)

            if horizon_line is not None:
                sx = ORIG_W / FULL_W
                sy = ORIG_H / FULL_H
                pred_orig = (horizon_line[0] * sx, horizon_line[1] * sy,
                             horizon_line[2] * sx, horizon_line[3] * sy)
                pred_y_center, pred_angle = compute_pred_metrics(pred_orig)
                ve_full = abs(pred_y_center - gt_y_center)
                ae_full = abs(wrap_angle_diff(pred_angle, gt_angle))
            else:
                n_full_fail += 1
        except Exception as e:
            print(f"  [{idx+1}/{total}] {stem} - 完整流水线异常: {e}")
            n_full_fail += 1

        t_full = (time.time() - t0) * 1000
        full_times.append(t_full)

        results.append({
            "stem": stem,
            "ve_segonly": ve_segonly,
            "ae_segonly": ae_segonly,
            "ve_full": ve_full,
            "ae_full": ae_full,
            "time_segonly_ms": t_segonly,
            "time_full_ms": t_full,
        })

        # 进度打印
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] {stem}  "
                  f"VE_seg={ve_segonly:.1f}px  AE_seg={ae_segonly:.2f}°  "
                  f"VE_full={ve_full:.1f}px  AE_full={ae_full:.2f}°  "
                  f"t_seg={t_segonly:.0f}ms  t_full={t_full:.0f}ms")

    # ======================================================================
    #  汇总统计
    # ======================================================================
    n_eval = len(results)
    if n_eval == 0:
        print("\n[ERROR] 没有成功评估任何样本！请检查图像路径。")
        return

    ve_seg_arr = np.array([r["ve_segonly"] for r in results])
    ae_seg_arr = np.array([r["ae_segonly"] for r in results])
    ve_full_arr = np.array([r["ve_full"] for r in results])
    ae_full_arr = np.array([r["ae_full"] for r in results])

    # 只统计有效样本
    ve_seg_valid = ve_seg_arr[~np.isnan(ve_seg_arr)]
    ae_seg_valid = ae_seg_arr[~np.isnan(ae_seg_arr)]
    ve_full_valid = ve_full_arr[~np.isnan(ve_full_arr)]
    ae_full_valid = ae_full_arr[~np.isnan(ae_full_arr)]

    def stats(arr):
        if len(arr) == 0:
            return 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr))

    ve_seg_mean, sve_seg = stats(ve_seg_valid)
    ae_seg_mean, sae_seg = stats(ae_seg_valid)
    ve_full_mean, sve_full = stats(ve_full_valid)
    ae_full_mean, sae_full = stats(ae_full_valid)

    time_seg_mean = float(np.mean(segonly_times)) if segonly_times else 0
    time_full_mean = float(np.mean(full_times)) if full_times else 0

    # 命中率 (VE < 阈值)
    def hit_rate(arr, thresh):
        if len(arr) == 0:
            return 0.0
        return float(np.sum(arr < thresh)) / len(arr) * 100

    hr_seg_10 = hit_rate(ve_seg_valid, 10)
    hr_seg_20 = hit_rate(ve_seg_valid, 20)
    hr_full_10 = hit_rate(ve_full_valid, 10)
    hr_full_20 = hit_rate(ve_full_valid, 20)

    # ---- 打印 ----
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("  实验3：UNet seg-only vs UNet + Radon + CNN 完整流水线")
    report_lines.append("=" * 72)
    report_lines.append(f"  评估样本数: {n_eval}")
    report_lines.append(f"  seg-only 检测失败: {n_segonly_fail}  |  完整流水线检测失败: {n_full_fail}")
    report_lines.append("")
    report_lines.append(f"  {'指标':<20s}  {'seg-only':>12s}  {'完整流水线':>12s}")
    report_lines.append(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    report_lines.append(f"  {'VE_mean (px)':<20s}  {ve_seg_mean:>12.2f}  {ve_full_mean:>12.2f}")
    report_lines.append(f"  {'SVE (px)':<20s}  {sve_seg:>12.2f}  {sve_full:>12.2f}")
    report_lines.append(f"  {'AE_mean (°)':<20s}  {ae_seg_mean:>12.2f}  {ae_full_mean:>12.2f}")
    report_lines.append(f"  {'SA (°)':<20s}  {sae_seg:>12.2f}  {sae_full:>12.2f}")
    report_lines.append(f"  {'推理时间 (ms)':<20s}  {time_seg_mean:>12.1f}  {time_full_mean:>12.1f}")
    report_lines.append(f"  {'FPS':<20s}  {1000/time_seg_mean if time_seg_mean>0 else 0:>12.1f}  {1000/time_full_mean if time_full_mean>0 else 0:>12.1f}")
    report_lines.append(f"  {'命中率 VE<10px (%)':<20s}  {hr_seg_10:>12.1f}  {hr_full_10:>12.1f}")
    report_lines.append(f"  {'命中率 VE<20px (%)':<20s}  {hr_seg_20:>12.1f}  {hr_full_20:>12.1f}")
    report_lines.append("=" * 72)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # ---- 保存结果 ----
    results_path = str(project_root / "eval_exp3_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n[INFO] 结果已保存到 {results_path}")

    # ---- 保存逐样本 CSV ----
    csv_path = str(project_root / "eval_exp3_per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "ve_segonly", "ae_segonly", "ve_full", "ae_full",
                         "time_segonly_ms", "time_full_ms"])
        for r in results:
            writer.writerow([
                r["stem"],
                f"{r['ve_segonly']:.4f}" if not np.isnan(r["ve_segonly"]) else "NaN",
                f"{r['ae_segonly']:.4f}" if not np.isnan(r["ae_segonly"]) else "NaN",
                f"{r['ve_full']:.4f}" if not np.isnan(r["ve_full"]) else "NaN",
                f"{r['ae_full']:.4f}" if not np.isnan(r["ae_full"]) else "NaN",
                f"{r['time_segonly_ms']:.1f}",
                f"{r['time_full_ms']:.1f}",
            ])
    print(f"[INFO] 逐样本结果已保存到 {csv_path}")


if __name__ == "__main__":
    main()
