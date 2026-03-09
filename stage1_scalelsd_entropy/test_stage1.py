#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stage1.py — Stage-1 测试入口

功能:
  1. 加载训练好的 best_stage1.pth 权重
  2. 在 test split 上逐样本评估
  3. 输出指标: rho error (像素), theta error (度), line distance
  4. 按阈值统计准确率

用法:
  python -m stage1_scalelsd_entropy.test_stage1 \
      --config stage1_scalelsd_entropy/configs/musid_entropy_stage1.yaml \
      --weights stage1_scalelsd_entropy/weights/best_stage1.pth

验证成功标志:
  - 输出包含 rho_err_px, theta_err_deg 等指标
  - 输出阈值统计: <=1°, <=2°, <=5° 等
"""

import os
import sys
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stage1_scalelsd_entropy.data.musid_dataset import MUSIDEntropyDataset
from stage1_scalelsd_entropy.models.scalelsd_entropy_wrapper import ScaleLSDEntropyWrapper


# ==================================================================
# Config
# ==================================================================
def load_yaml_config(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")


# ==================================================================
# 指标计算
# ==================================================================
def radon_label_to_rho_theta_physical(rho_norm, theta_norm, sinogram_h=2240, img_w=1024, img_h=576):
    """
    将归一化 (rho_norm, theta_norm) 转换回物理量。

    Returns:
        rho_pixel: 距图像中心的法距 (像素)
        theta_deg: 法向角 (度)
    """
    # theta
    theta_deg = theta_norm * 180.0

    # rho
    diag = np.sqrt(img_w ** 2 + img_h ** 2)
    pad = (sinogram_h - diag) / 2.0
    rho_idx = rho_norm * (sinogram_h - 1)
    rho_pixel = rho_idx - pad - diag / 2.0

    return rho_pixel, theta_deg


def compute_line_distance(pred_rho_px, pred_theta_deg, gt_rho_px, gt_theta_deg, img_w=1024, img_h=576):
    """
    计算预测线与 GT 线之间在图像左右端点处的平均垂直距离。
    """
    # 将 (rho, theta) 转换为图像上的两个端点
    def rho_theta_to_endpoints(rho, theta_deg, w, h):
        theta_rad = np.radians(theta_deg)
        cx, cy = w / 2.0, h / 2.0
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        # 线方程: x*cos(t) + y*sin(t) = rho (相对中心)
        # y = (rho - (x-cx)*cos(t)) / sin(t) + cy
        if abs(sin_t) > 1e-6:
            y_left = (rho - (0 - cx) * cos_t) / sin_t + cy
            y_right = (rho - (w - 1 - cx) * cos_t) / sin_t + cy
        else:
            # 近乎垂直线
            y_left = cy
            y_right = cy
        return y_left, y_right

    py_l, py_r = rho_theta_to_endpoints(pred_rho_px, pred_theta_deg, img_w, img_h)
    gy_l, gy_r = rho_theta_to_endpoints(gt_rho_px, gt_theta_deg, img_w, img_h)

    dist = (abs(py_l - gy_l) + abs(py_r - gy_r)) / 2.0
    return dist


# ==================================================================
# Main
# ==================================================================
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Stage-1 Test: evaluate horizon regression")
    parser.add_argument("--config", type=str,
                        default="stage1_scalelsd_entropy/configs/musid_entropy_stage1.yaml")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (overrides config)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--mode", type=str, default=None, choices=["baseline", "entropy"])
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]

    mode = args.mode if args.mode else model_cfg["mode"]
    weights_path = args.weights if args.weights else out_cfg["best_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Stage-1 Test — mode={mode}, split={args.split}")
    print(f"Weights: {weights_path}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- Dataset ----
    csv_key = f"csv_{args.split}"
    csv_path = data_cfg[csv_key]

    ds = MUSIDEntropyDataset(
        csv_path=csv_path,
        img_dir=data_cfg["img_dir"],
        entropy_dir=data_cfg["entropy_dir"],
        img_h=data_cfg["img_h"],
        img_w=data_cfg["img_w"],
        orig_h=data_cfg["orig_h"],
        orig_w=data_cfg["orig_w"],
        sinogram_h=data_cfg["sinogram_h"],
        augment=False,
        label_mode="radon",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Samples: {len(ds)}")

    # ---- Model ----
    model = ScaleLSDEntropyWrapper(
        mode=mode,
        entropy_branch_ch=model_cfg["entropy_branch_ch"],
        backbone_l2_ch=model_cfg["backbone_l2_ch"],
    ).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # ---- Inference ----
    img_w = data_cfg["img_w"]
    img_h = data_cfg["img_h"]
    sinogram_h = data_cfg["sinogram_h"]

    results = []
    for img, ent, label, stems in loader:
        img = img.to(device)
        ent = ent.to(device) if mode == "entropy" else None
        label = label.numpy()[0]  # [2]
        stem = stems[0]

        pred = model(img, entropy_map=ent).cpu().numpy()[0]  # [2]

        # 转物理量
        pred_rho_px, pred_theta_deg = radon_label_to_rho_theta_physical(
            pred[0], pred[1], sinogram_h, img_w, img_h)
        gt_rho_px, gt_theta_deg = radon_label_to_rho_theta_physical(
            label[0], label[1], sinogram_h, img_w, img_h)

        rho_err = abs(pred_rho_px - gt_rho_px)
        theta_err = abs(pred_theta_deg - gt_theta_deg)
        # wrap theta error to [0, 90]
        if theta_err > 90:
            theta_err = 180 - theta_err

        line_dist = compute_line_distance(pred_rho_px, pred_theta_deg,
                                          gt_rho_px, gt_theta_deg, img_w, img_h)

        results.append({
            "stem": stem,
            "rho_err_px": float(rho_err),
            "theta_err_deg": float(theta_err),
            "line_dist_px": float(line_dist),
        })

    # ---- Aggregate metrics ----
    rho_errs = np.array([r["rho_err_px"] for r in results])
    theta_errs = np.array([r["theta_err_deg"] for r in results])
    line_dists = np.array([r["line_dist_px"] for r in results])

    print("\n" + "=" * 60)
    print(f"Results ({len(results)} samples, mode={mode}):")
    print(f"  Rho error   — mean={rho_errs.mean():.2f}px  median={np.median(rho_errs):.2f}px  max={rho_errs.max():.2f}px")
    print(f"  Theta error — mean={theta_errs.mean():.3f}°  median={np.median(theta_errs):.3f}°  max={theta_errs.max():.3f}°")
    print(f"  Line dist   — mean={line_dists.mean():.2f}px  median={np.median(line_dists):.2f}px  max={line_dists.max():.2f}px")

    # 阈值统计
    print("\n  Theta accuracy:")
    for th in [1, 2, 5, 10]:
        pct = (theta_errs <= th).sum() / len(theta_errs) * 100
        print(f"    <= {th}° : {pct:.1f}%")

    print("\n  Line distance accuracy:")
    for th in [5, 10, 20, 50]:
        pct = (line_dists <= th).sum() / len(line_dists) * 100
        print(f"    <= {th}px : {pct:.1f}%")

    # ---- Save results ----
    out_path = os.path.join(out_cfg["weights_dir"], f"test_results_{mode}_{args.split}.json")
    payload = {
        "mode": mode,
        "split": args.split,
        "n_samples": len(results),
        "metrics": {
            "rho_err_mean": float(rho_errs.mean()),
            "rho_err_median": float(np.median(rho_errs)),
            "theta_err_mean": float(theta_errs.mean()),
            "theta_err_median": float(np.median(theta_errs)),
            "line_dist_mean": float(line_dists.mean()),
            "line_dist_median": float(np.median(line_dists)),
        },
        "per_sample": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
