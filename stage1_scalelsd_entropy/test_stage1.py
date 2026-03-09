#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stage1.py — Stage-1 测试入口（真实 ScaleLSD + entropy injection）

功能:
  1. 加载训练好的 checkpoint
  2. 在 test split 上逐样本推理（调用 ScaleLSD.forward_test）
  3. 保存每张图的线段检测结果（lines_pred, lines_score, juncs_pred 等）
  4. 将结果汇总保存为 JSON，供后续评估和可视化使用

输出:
  - JSON 文件：每张图的预测线段、端点、得分
  - 可选：可视化叠加图（在图上画预测线段）

用法:
  在 PyCharm 中直接修改顶部全局变量后运行。
"""

import os
import sys
import json

# 将 scalelsd 仓库目录加入 sys.path，使 from scalelsd.ssl.* 可用
_SCALELSD_REPO = os.path.join(os.path.dirname(__file__), "..", "scalelsd")
if os.path.isdir(_SCALELSD_REPO) and _SCALELSD_REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCALELSD_REPO))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1_scalelsd_entropy.data.musid_entropy_dataset import MUSIDEntropyDataset

# ============================================================
# 全局配置（在 PyCharm 中直接修改后运行）
# ============================================================

# 模式："baseline"（原始 ScaleLSD）or "entropy"（ScaleLSDWithEntropy）
MODE = "entropy"

# 数据路径
IMG_DIR     = "Hashmani's Dataset/MU-SID"
ENTROPY_DIR = "Hashmani's Dataset/MU-SID_entropy_blue"
CSV_TEST    = "splits_musid/GroundTruth_test.csv"

# 图像尺寸
IMG_H = 576
IMG_W = 1024

# 模型权重路径
WEIGHTS_PATH = "stage1_scalelsd_entropy/weights/best_stage1_entropy.pth"

# 推理参数
BATCH_SIZE  = 1       # forward_test 内部使用 LSD，batch>1 可能导致问题
NUM_WORKERS = 0
USE_LSD     = True    # 是否使用 LSD 辅助方向预测
USE_NMS     = True    # 是否对 junction heatmap 做 NMS

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出
SAVE_JSON = "stage1_scalelsd_entropy/weights/test_results.json"
VIS_DIR   = "stage1_scalelsd_entropy/weights/test_vis"    # 可视化输出目录（设为 "" 则不可视化）
VIS_TOP_K = 50        # 每张图最多画 VIS_TOP_K 条得分最高的线段



# ============================================================
# Collate: MUSIDEntropyDataset dict → test 格式
# ============================================================
def collate_test(batch):
    """
    测试用 collate：从 MUSIDEntropyDataset dict 中提取
    images, entropy_maps, gt_lines [B,4], stems。
    """
    images = torch.stack([s["image"] for s in batch], dim=0)
    entropy_maps = torch.stack([s["entropy_map"] for s in batch], dim=0)
    gt_lines = []
    stems = []
    for s in batch:
        ep = s["annotation"]["resized_endpoints"]  # np [2, 2]
        gt_line = np.array([ep[0, 0], ep[0, 1], ep[1, 0], ep[1, 1]], dtype=np.float32)
        gt_lines.append(torch.from_numpy(gt_line))
        stems.append(s["annotation"]["stem"])
    gt_lines = torch.stack(gt_lines, dim=0)  # [B, 4]
    return images, entropy_maps, gt_lines, stems


# ============================================================
# 模型构建
# ============================================================
def build_model(mode, weights_path, device):
    if mode == "baseline":
        from scalelsd.ssl.models.detector import ScaleLSD
        model = ScaleLSD(gray_scale=True)
    elif mode == "entropy":
        from stage1_scalelsd_entropy.models.scalelsd_with_entropy import ScaleLSDWithEntropy
        model = ScaleLSDWithEntropy(gray_scale=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        if "model_state" in state:
            state = state["model_state"]
        elif "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print(f"[WARN] weights not found: {weights_path}")

    return model.to(device)


# ============================================================
# 海天线候选选择
# ============================================================
def select_horizon_candidate(lines, scores, img_h, img_w,
                             max_horizon_dev_deg=15.0, min_length_ratio=0.2):
    """
    从预测线段中选出最可能的海天线候选。

    启发式规则：
      1. 优先选择接近水平的线段（相对水平线的偏差角较小）
      2. 线段长度 >= min_length_ratio * img_w
      3. 在满足条件的线段中按得分排序，取最高分
      4. 如果没有满足条件的候选，则退化为取最长线段

    Args:
        lines: [N, 4] — x1,y1,x2,y2
        scores: [N]
        img_h, img_w: 图像尺寸
        max_horizon_dev_deg: 相对水平线的最大允许偏差角，单位度
        min_length_ratio: 线段最短长度占图像宽度比例

    Returns:
        best_line: [4] or None
        best_score: float or None
        best_idx: int or None
    """
    import numpy as np

    if len(lines) == 0:
        return None, None, None

    lines_np = np.array(lines)
    scores_np = np.array(scores)

    dx = lines_np[:, 2] - lines_np[:, 0]
    dy = lines_np[:, 3] - lines_np[:, 1]
    raw_angles = np.degrees(np.arctan2(dy, dx))
    raw_angles = np.abs(raw_angles)

    # 距离“水平线”的偏差角：0 表示水平，90 表示竖直
    horizon_dev = np.minimum(raw_angles, 180.0 - raw_angles)

    lengths = np.sqrt(dx ** 2 + dy ** 2)
    min_len = min_length_ratio * img_w
    mask = (horizon_dev <= max_horizon_dev_deg) & (lengths >= min_len)

    if not mask.any():
        best_idx = int(np.argmax(lengths))
        return lines_np[best_idx], float(scores_np[best_idx]), best_idx

    valid_scores = scores_np.copy()
    valid_scores[~mask] = -1
    best_idx = int(np.argmax(valid_scores))
    return lines_np[best_idx], float(scores_np[best_idx]), best_idx


# ============================================================
# 评估指标：端点距离
# ============================================================
def compute_endpoint_error(pred_line, gt_line, img_w, img_h):
    """
    计算预测线与 GT 线在图像左右边界处的 y 坐标差异均值。

    pred_line, gt_line: [x1, y1, x2, y2]
    """
    def line_y_at_x(line, x_query):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        if abs(dx) < 1e-6:
            return (y1 + y2) / 2.0
        t = (x_query - x1) / dx
        return y1 + t * (y2 - y1)

    pred_y_left = line_y_at_x(pred_line, 0)
    pred_y_right = line_y_at_x(pred_line, img_w - 1)
    gt_y_left = line_y_at_x(gt_line, 0)
    gt_y_right = line_y_at_x(gt_line, img_w - 1)

    err_left = abs(pred_y_left - gt_y_left)
    err_right = abs(pred_y_right - gt_y_right)
    return (err_left + err_right) / 2.0


# ============================================================
# 可视化
# ============================================================
def visualize_predictions(gray_img, pred_lines, pred_scores, gt_line,
                          horizon_line, save_path, top_k=50):
    """
    画预测线段和 GT 线到灰度图上并保存。
    """
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]

    # 画全部预测线段（蓝色，细线）
    if len(pred_lines) > 0:
        sorted_idx = np.argsort(pred_scores)[::-1][:top_k]
        for i in sorted_idx:
            x1, y1, x2, y2 = pred_lines[i].astype(int)
            cv2.line(vis, (x1, y1), (x2, y2), (255, 128, 0), 1)

    # 画 GT 海天线（绿色，粗线）
    gx1, gy1, gx2, gy2 = gt_line.astype(int)
    cv2.line(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

    # 画选出的海天线候选（红色，粗线）
    if horizon_line is not None:
        hx1, hy1, hx2, hy2 = horizon_line.astype(int)
        cv2.line(vis, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)

    cv2.imwrite(save_path, vis)


# ============================================================
# Main
# ============================================================
@torch.no_grad()
def main():
    device = DEVICE
    mode = MODE

    print("=" * 60)
    print(f"Stage-1 Test — mode={mode}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- Dataset ----
    ds = MUSIDEntropyDataset(
        csv_file=CSV_TEST, img_dir=IMG_DIR, entropy_dir=ENTROPY_DIR,
        img_size=(IMG_H, IMG_W), gray_scale=True,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_test)
    print(f"Test samples: {len(ds)}")

    # ---- Model ----
    model = build_model(mode, WEIGHTS_PATH, device)
    model.eval()
    stride = model.stride
    print(f"Model stride: {stride}")

    if mode == "entropy":
        print(f"alpha = {model.backbone.alpha.item():.6f}")

    # ---- 可视化目录 ----
    do_vis = VIS_DIR and len(VIS_DIR) > 0
    if do_vis:
        os.makedirs(VIS_DIR, exist_ok=True)

    # ---- Inference ----
    results = []
    endpoint_errors = []

    for img_tensor, ent_tensor, gt_line_np, stem in tqdm(loader, desc="test", ncols=90):
        img_tensor = img_tensor.to(device)
        ent_map = ent_tensor.to(device) if mode == "entropy" else None

        # forward_test annotations
        ann = {
            "width": IMG_W,
            "height": IMG_H,
            "use_lsd": USE_LSD,
            "use_nms": USE_NMS,
        }

        if mode == "entropy":
            output_list, _ = model.forward_test(
                img_tensor, annotations=ann, entropy_map=ent_map)
        else:
            output_list, _ = model.forward_test(img_tensor, annotations=ann)

        # 处理每个 batch 样本
        for bi in range(len(output_list)):
            out = output_list[bi]
            cur_stem = stem[bi] if isinstance(stem, (list, tuple)) else stem

            lines_pred = out["lines_pred"].cpu().numpy()     # [N, 4]
            lines_score = out["lines_score"].cpu().numpy()   # [N]
            juncs_pred = out["juncs_pred"].cpu().numpy()      # [M, 2]
            juncs_score = out["juncs_score"].cpu().numpy()    # [M]

            gt_line = gt_line_np[bi].numpy() if torch.is_tensor(gt_line_np) else gt_line_np[bi]

            # 选出海天线候选
            horizon_line, horizon_score, horizon_idx = select_horizon_candidate(
                lines_pred, lines_score, IMG_H, IMG_W,
            )

            # 计算端点误差
            ep_err = None
            if horizon_line is not None:
                ep_err = compute_endpoint_error(horizon_line, gt_line, IMG_W, IMG_H)
                endpoint_errors.append(ep_err)

            rec = {
                "stem": cur_stem,
                "num_lines": int(len(lines_pred)),
                "num_junctions": int(len(juncs_pred)),
                "horizon_line": horizon_line.tolist() if horizon_line is not None else None,
                "horizon_score": horizon_score,
                "endpoint_error_px": ep_err,
                "gt_line": gt_line.tolist(),
                "lines_pred": lines_pred.tolist(),
                "lines_score": lines_score.tolist(),
            }
            results.append(rec)

            # 可视化
            if do_vis:
                gray_for_vis = (img_tensor[bi, 0].cpu().numpy() * 255).astype(np.uint8)
                vis_path = os.path.join(VIS_DIR, f"{cur_stem}.jpg")
                visualize_predictions(
                    gray_for_vis, lines_pred, lines_score, gt_line,
                    horizon_line, vis_path, top_k=VIS_TOP_K,
                )

    # ---- 汇总指标 ----
    print("\n" + "=" * 60)
    print(f"Results ({len(results)} samples, mode={mode}):")
    print(f"  Avg lines per image: {np.mean([r['num_lines'] for r in results]):.1f}")
    print(f"  Avg junctions per image: {np.mean([r['num_junctions'] for r in results]):.1f}")

    if endpoint_errors:
        ep_arr = np.array(endpoint_errors)
        print(f"\n  Horizon endpoint error (px) on {len(ep_arr)} samples with valid candidate:")
        print(f"    mean={ep_arr.mean():.2f}  median={np.median(ep_arr):.2f}  "
              f"max={ep_arr.max():.2f}")
        for th in [5, 10, 20, 50]:
            pct = (ep_arr <= th).sum() / len(ep_arr) * 100
            print(f"    <= {th}px : {pct:.1f}%")
    else:
        print("\n  No valid horizon candidates found.")

    n_no_candidate = sum(1 for r in results if r["horizon_line"] is None)
    print(f"\n  Samples with no horizon candidate: {n_no_candidate}/{len(results)}")

    # ---- 保存 JSON ----
    os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
    payload = {
        "mode": mode,
        "weights": WEIGHTS_PATH,
        "n_samples": len(results),
        "img_size": [IMG_H, IMG_W],
        "per_sample": results,
    }
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {SAVE_JSON}")
    if do_vis:
        print(f"Visualizations saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
