"""
train_reranker.py — 第二阶段 reranker 训练脚本

基于 export_reranker_data.py 导出的候选线特征 CSV，训练轻量 MLP reranker，
对每张图的候选线重新打分，选出最优海岸线。

输入：export_reranker_data.py 导出的 train / val CSV
输出：best_model.pth / last_model.pth / feature_config.json / 训练日志 / 验证报告

用法：
  在 PyCharm 中修改顶部全局变量后直接运行本文件。
"""

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 项目路径设置
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent           # stage2_reranker/
_PROJECT_ROOT = _SCRIPT_DIR.parent                      # sealine_detection/

# ============================================================
# 顶部全局变量配置 — 在 PyCharm 中直接修改
# ============================================================

# ---- 输入输出路径 ----
EXPORT_DIR = "stage2_reranker/exports/entropy__best_checkpoint"
TRAIN_CSV = ""      # 留空则自动拼接 EXPORT_DIR/train_reranker_candidates.csv
VAL_CSV = ""        # 留空则自动拼接 EXPORT_DIR/val_reranker_candidates.csv
OUTPUT_DIR = "stage2_reranker/reranker_output/entropy__best_checkpoint"

# ---- 训练特征列（与 export_reranker_data.py CSV_COLUMNS 对齐）----
FEATURE_COLUMNS = [
    "det_score",
    "length",
    "length_norm",
    "horizon_dev_deg",
    "x_span",
    "x_span_norm",
    "y_mid",
    "y_mid_norm",
    "ent_upper",
    "ent_lower",
    "delta_ent",
    "ent_consistency",
    "grad_normal",
    "grad_tangent",
    "grad_ratio",
    "grad_consistency",
]

# 辅助列（不纳入训练特征，但需要读取用于评估）
AUX_COLUMNS = [
    "split", "image_id", "image_stem", "image_path", "candidate_rank",
    "endpoint_err", "is_best_match", "label",
    "pass_angle_gate", "pass_length_gate", "heuristic_score",
    "det_score",
]

# ---- 模型超参 ----
HIDDEN_DIMS = [64, 32]           # MLP 隐藏层维度
DROPOUT = 0.3                    # Dropout 比率

# ---- 训练超参 ----
BATCH_SIZE = 256
NUM_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
DEVICE = "cuda"

# ---- 软目标参数 ----
# target_score = exp(-endpoint_err / TAU)
TAU = 10.0                       # 温度参数：越小则对 endpoint_err 越敏感

# ---- 图像级权重参数 ----
# best_err = min(endpoint_err)  in each image
# image_weight = 1.0           if best_err <= T_GOOD
#              = linear decay   if T_GOOD < best_err < T_DROP
#              = 0.0           if best_err >= T_DROP
T_GOOD = 10.0
T_DROP = 20.0

# ---- 回归损失类型 ----
REG_LOSS_TYPE = "mse"            # "mse" 或 "smoothl1"

# ---- Best model 选择 ----
#   "image_mean_err"  — 以验证集 image-level mean endpoint error 最小为 best
#   "val_loss"        — 以验证集 loss 最小为 best（fallback）
BEST_METRIC = "image_mean_err"


# ============================================================
# 路径工具
# ============================================================
def _resolve(p):
    """相对路径基于项目根目录解析为绝对路径。"""
    if not p:
        return p
    pp = Path(p)
    if not pp.is_absolute():
        pp = _PROJECT_ROOT / pp
    return str(pp)


def _get_csv_path(csv_arg, export_dir, split_name):
    if csv_arg:
        return _resolve(csv_arg)
    return os.path.join(_resolve(export_dir), f"{split_name}_reranker_candidates.csv")


# ============================================================
# 1. 数据加载与清洗
# ============================================================
def load_csv(filepath):
    """
    读取候选线 CSV，返回 list of dict。

    字段类型：
      - 数值列自动转 float（含 nan / inf 处理）
      - 其他列保留 str
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    fv = float(v)
                    # 将 inf 替换为 nan，后续统一处理
                    if not math.isfinite(fv):
                        fv = float("nan")
                    parsed[k] = fv
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    print(f"  已加载 CSV: {filepath}  ({len(rows)} 行)")
    return rows


def extract_features_and_meta(rows, feature_cols):
    """
    从 rows 中提取特征矩阵 X 和辅助信息 meta。

    Returns:
        X     : ndarray [N, D] float32
        meta  : list of dict（每行的辅助信息，含 endpoint_err 等）
    """
    N = len(rows)
    D = len(feature_cols)
    X = np.zeros((N, D), dtype=np.float32)
    meta = []

    for i, row in enumerate(rows):
        for j, col in enumerate(feature_cols):
            val = row.get(col, float("nan"))
            if isinstance(val, str):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = float("nan")
            X[i, j] = val

        # endpoint_err
        err_val = row.get("endpoint_err", float("nan"))
        if isinstance(err_val, str):
            try:
                err_val = float(err_val)
            except (ValueError, TypeError):
                err_val = float("nan")

        label_val = row.get("label", 0)
        if isinstance(label_val, str):
            try:
                label_val = float(label_val)
            except (ValueError, TypeError):
                label_val = 0.0

        meta.append({
            "image_stem": str(row.get("image_stem", "")),
            "image_id": row.get("image_id", i),
            "candidate_rank": row.get("candidate_rank", i),
            "endpoint_err": err_val,
            "is_best_match": row.get("is_best_match", 0),
            "det_score": row.get("det_score", 0.0),
            "heuristic_score": row.get("heuristic_score", float("nan")),
            "label": float(label_val),
        })

    return X, meta


def compute_soft_targets(meta, tau):
    """
    基于 endpoint_err 生成连续目标分数：target_score = exp(-err / tau)。

    Returns:
        ndarray [N] float32  — 范围 (0, 1]
    """
    N = len(meta)
    targets = np.zeros(N, dtype=np.float32)
    for i, m in enumerate(meta):
        err = m["endpoint_err"]
        if math.isfinite(err) and err >= 0:
            targets[i] = math.exp(-err / tau)
        else:
            targets[i] = 0.0
    return targets


def compute_image_weights(meta, t_good, t_drop):
    """
    基于每张图候选池中最小 endpoint_err 计算图像级权重。

    规则：
      best_err <= t_good       → 1.0
      t_good < best_err < t_drop → 线性衰减
      best_err >= t_drop       → 0.0

    Returns:
        weights  : ndarray [N] float32  — 每条候选线继承其所属图像的权重
        img_stats: dict  — 图像级权重统计
    """
    # 按图分组，找每图最小 err
    image_best_err = defaultdict(lambda: float("inf"))
    image_indices = defaultdict(list)
    for i, m in enumerate(meta):
        stem = m["image_stem"]
        err = m["endpoint_err"]
        image_indices[stem].append(i)
        if math.isfinite(err):
            image_best_err[stem] = min(image_best_err[stem], err)

    # 计算每图权重
    image_weight_map = {}
    for stem in image_indices:
        best_err = image_best_err[stem]
        if not math.isfinite(best_err):
            image_weight_map[stem] = 0.0
        elif best_err <= t_good:
            image_weight_map[stem] = 1.0
        elif best_err >= t_drop:
            image_weight_map[stem] = 0.0
        else:
            image_weight_map[stem] = (t_drop - best_err) / (t_drop - t_good)

    # 展开到每条候选线
    N = len(meta)
    weights = np.zeros(N, dtype=np.float32)
    for stem, indices in image_indices.items():
        w = image_weight_map[stem]
        for idx in indices:
            weights[idx] = w

    # 统计
    all_w = list(image_weight_map.values())
    n_total = len(all_w)
    n_used = sum(1 for w in all_w if w > 0)
    n_dropped = n_total - n_used
    img_stats = {
        "num_images_total": n_total,
        "num_images_used_for_training": n_used,
        "num_images_dropped_by_weight": n_dropped,
        "mean_image_weight": float(np.mean(all_w)) if all_w else 0.0,
        "median_image_weight": float(np.median(all_w)) if all_w else 0.0,
    }

    return weights, img_stats


def clean_features(X):
    """
    对特征矩阵做稳健清洗：
      1. nan → 列中位数填充（如果全 nan 则填 0）
      2. clip 到 ±1e6 范围
    """
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[:] = 0.0
        elif nan_mask.any():
            median_val = float(np.nanmedian(col))
            col[nan_mask] = median_val
        X[:, j] = col
    X = np.clip(X, -1e6, 1e6)
    return X


# ============================================================
# 2. 特征标准化（仅用 train 统计）
# ============================================================
class FeatureScaler:
    """简单的 z-score 标准化器，仅用 train 集的 mean/std。"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        # 避免除零
        self.std[self.std < 1e-8] = 1.0

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        scaler = cls()
        scaler.mean = np.array(d["mean"], dtype=np.float32)
        scaler.std = np.array(d["std"], dtype=np.float32)
        return scaler


# ============================================================
# 3. Dataset
# ============================================================
class CandidateDataset(Dataset):
    """
    表格数据 Dataset。

    每条候选线返回：
      features      : Tensor [D]
      target_score  : Tensor []  — soft target (exp(-err/tau))
      image_weight  : Tensor []  — 图像级权重
      idx           : int        — 全局索引（用于 meta 反查）
    """

    def __init__(self, X, target_scores, image_weights, meta=None):
        self.X = torch.from_numpy(X)
        self.target_scores = torch.from_numpy(target_scores)
        self.image_weights = torch.from_numpy(image_weights)
        self.meta = meta

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.target_scores[idx], self.image_weights[idx], idx


# ============================================================
# 4. MLP 模型
# ============================================================
class RerankerMLP(nn.Module):
    """
    轻量 MLP 打分模型。

    输入：特征向量 [B, D]
    输出：score [B, 1]（经 sigmoid → [0,1]，用于图内排序）
    """

    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [B, 1]


# ============================================================
# 5. 构建回归损失
# ============================================================
def build_regression_criterion(loss_type):
    """返回 reduction='none' 的回归损失函数。"""
    if loss_type == "smoothl1":
        return nn.SmoothL1Loss(reduction="none")
    return nn.MSELoss(reduction="none")


# ============================================================
# 6. 训练一个 epoch（soft target + image weight）
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    每条候选线的 loss = criterion(sigmoid(logit), target_score) * image_weight。
    只对 image_weight > 0 的样本有效，最终取加权平均。
    """
    model.train()
    total_weighted_loss = 0.0
    total_weight = 0.0

    for X_batch, target_batch, weight_batch, _ in dataloader:
        X_batch = X_batch.to(device)
        target_batch = target_batch.to(device).unsqueeze(1)   # [B, 1]
        weight_batch = weight_batch.to(device).unsqueeze(1)   # [B, 1]

        logits = model(X_batch)          # [B, 1]
        preds = torch.sigmoid(logits)    # [B, 1]  → [0, 1]

        per_sample_loss = criterion(preds, target_batch)  # [B, 1]
        weighted_loss = per_sample_loss * weight_batch     # [B, 1]

        batch_weight_sum = weight_batch.sum()
        if batch_weight_sum.item() > 0:
            loss = weighted_loss.sum() / batch_weight_sum
        else:
            loss = weighted_loss.sum() * 0.0  # 无有效样本，zero grad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_weighted_loss += weighted_loss.sum().item()
        total_weight += batch_weight_sum.item()

    avg_loss = total_weighted_loss / max(total_weight, 1e-8)
    return avg_loss


# ============================================================
# 6. 验证：逐候选线指标 + 按图重排序指标
# ============================================================
def evaluate(model, dataloader, dataset, criterion, device, split_summary=None):
    """
    验证函数，同时输出：
      1. candidate-level 指标（regression loss / 与硬 label 的兼容统计）
      2. image-level 指标（reranker 选择 vs det_score baseline vs heuristic baseline）

    模型输出 logit → sigmoid 作为排序分数，在同一张图内选分数最高者。
    """
    model.eval()
    all_logits = []
    all_indices = []
    total_weighted_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for X_batch, target_batch, weight_batch, idx_batch in dataloader:
            X_batch = X_batch.to(device)
            target_batch = target_batch.to(device).unsqueeze(1)
            weight_batch = weight_batch.to(device).unsqueeze(1)

            logits = model(X_batch)
            preds = torch.sigmoid(logits)
            per_sample_loss = criterion(preds, target_batch)
            weighted_loss = per_sample_loss * weight_batch

            total_weighted_loss += weighted_loss.sum().item()
            total_weight += weight_batch.sum().item()

            all_logits.append(logits.cpu().squeeze(1))
            all_indices.append(idx_batch)

    all_logits = torch.cat(all_logits).numpy()       # [N]
    all_indices = torch.cat(all_indices).numpy()      # [N]
    all_scores = 1.0 / (1.0 + np.exp(-all_logits))   # sigmoid → [0, 1]

    avg_loss = total_weighted_loss / max(total_weight, 1e-8)

    # ------ candidate-level 指标 ------
    meta = dataset.meta
    all_labels = np.array([meta[int(idx)]["label"] for idx in all_indices], dtype=np.float32)
    all_target_scores = dataset.target_scores.numpy()[all_indices.astype(int)]

    # 兼容性统计：用硬 label 计算 AUC 和分类指标（仅供参考）
    preds_binary = (all_scores >= 0.5).astype(np.float32)
    tp = float(((preds_binary == 1) & (all_labels == 1)).sum())
    fp = float(((preds_binary == 1) & (all_labels == 0)).sum())
    fn = float(((preds_binary == 0) & (all_labels == 1)).sum())
    tn = float(((preds_binary == 0) & (all_labels == 0)).sum())

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    auc = _compute_auc(all_labels, all_scores)

    candidate_metrics = {
        "val_loss": avg_loss,
        "val_soft_target_mean": float(np.mean(all_target_scores)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "num_candidates": len(all_labels),
        "num_positive": int(all_labels.sum()),
        "num_negative": int((all_labels == 0).sum()),
    }

    # ------ image-level 重排序指标 ------
    image_groups = defaultdict(list)
    for i, global_idx in enumerate(all_indices):
        m = meta[int(global_idx)]
        image_groups[m["image_stem"]].append({
            "idx": int(global_idx),
            "reranker_score": float(all_scores[i]),
            "reranker_logit": float(all_logits[i]),
            "det_score": float(m["det_score"]),
            "heuristic_score": float(m["heuristic_score"]) if math.isfinite(float(m.get("heuristic_score", float("nan")))) else None,
            "endpoint_err": float(m["endpoint_err"]),
            "is_best_match": int(m["is_best_match"]),
            "candidate_rank": m["candidate_rank"],
            "label": int(m["label"]),
        })

    reranker_errs = []
    det_baseline_errs = []
    heuristic_errs = []
    any_image_has_valid_heuristic = False
    per_image_results = []

    for stem, candidates in image_groups.items():
        best_reranker = max(candidates, key=lambda c: c["reranker_score"])
        reranker_errs.append(best_reranker["endpoint_err"])

        best_det = max(candidates, key=lambda c: c["det_score"])
        det_baseline_errs.append(best_det["endpoint_err"])

        valid_heur = [c for c in candidates if c["heuristic_score"] is not None
                      and math.isfinite(c["heuristic_score"])]
        if valid_heur:
            best_heur = max(valid_heur, key=lambda c: c["heuristic_score"])
            heuristic_errs.append(best_heur["endpoint_err"])
            any_image_has_valid_heuristic = True
        else:
            heuristic_errs.append(best_det["endpoint_err"])

        per_image_results.append({
            "image_stem": stem,
            "num_candidates": len(candidates),
            "reranker_selected_err": best_reranker["endpoint_err"],
            "reranker_selected_rank": best_reranker["candidate_rank"],
            "det_baseline_selected_err": best_det["endpoint_err"],
            "det_baseline_selected_rank": best_det["candidate_rank"],
        })

    image_reranker = _compute_err_stats(reranker_errs, "reranker")
    image_det_baseline = _compute_err_stats(det_baseline_errs, "det_score_baseline")
    image_heuristic = (_compute_err_stats(heuristic_errs, "heuristic_baseline")
                       if any_image_has_valid_heuristic else None)

    image_reranker = enrich_image_level_stats_with_summary(image_reranker, split_summary)
    image_det_baseline = enrich_image_level_stats_with_summary(image_det_baseline, split_summary)
    if image_heuristic is not None:
        image_heuristic = enrich_image_level_stats_with_summary(image_heuristic, split_summary)

    # ------ per-row predictions ------
    all_predictions = []
    for i, global_idx in enumerate(all_indices):
        m = meta[int(global_idx)]
        all_predictions.append({
            "split": "val",
            "image_id": m["image_id"],
            "image_stem": m["image_stem"],
            "candidate_rank": m["candidate_rank"],
            "label": int(m["label"]),
            "det_score": float(m["det_score"]),
            "reranker_score": float(all_scores[i]),
            "endpoint_err": float(m["endpoint_err"]),
            "is_best_match": int(m["is_best_match"]),
        })

    results = {
        "candidate_level_metrics": candidate_metrics,
        "image_level_metrics_reranker": image_reranker,
        "image_level_metrics_det_score_baseline": image_det_baseline,
        "per_image_results": per_image_results,
        "predictions": all_predictions,
    }
    if image_heuristic is not None:
        results["image_level_metrics_heuristic_baseline"] = image_heuristic

    return results


def _compute_err_stats(errs, name):
    """计算 endpoint error 统计（仅基于 CSV 中出现候选线的图像）。"""
    errs = np.array(errs, dtype=np.float64)
    n = len(errs)
    valid = errs[np.isfinite(errs)]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "method": name,
            "num_images": n,
            "num_images_with_candidate": 0,
            "num_images_without_candidate": n,
            "mean_endpoint_err": float("nan"),
            "median_endpoint_err": float("nan"),
        }

    return {
        "method": name,
        "num_images": n,
        "num_images_with_candidate": n_valid,
        "num_images_without_candidate": n - n_valid,
        "mean_endpoint_err": float(np.mean(valid)),
        "median_endpoint_err": float(np.median(valid)),
        "std_endpoint_err": float(np.std(valid)),
        "pct_le_5": float(np.mean(valid <= 5) * 100),
        "pct_le_10": float(np.mean(valid <= 10) * 100),
        "pct_le_20": float(np.mean(valid <= 20) * 100),
        "pct_le_50": float(np.mean(valid <= 50) * 100),
    }


def load_split_summary(export_dir, split_name):
    """
    从导出目录读取 {split}_summary.json，获取该 split 的真实图像统计。

    Returns:
        dict 或 None — 包含 num_images / num_images_no_lines 等字段
    """
    summary_path = os.path.join(_resolve(export_dir), f"{split_name}_summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def enrich_image_level_stats_with_summary(stats_dict, split_summary):
    """
    用 split_summary.json 中的真实统计修正 image-level 指标中的
    num_images / num_images_with_candidate / num_images_without_candidate。

    逻辑：
      - num_images = split_summary["num_images"]（数据集真实图像总数）
      - num_images_with_candidate = stats_dict 中从 CSV 推算的有候选线图像数
      - num_images_without_candidate = num_images - num_images_with_candidate

    如果 split_summary 为 None，则不做修正（保留 CSV 近似统计）。
    """
    if split_summary is None:
        return stats_dict

    total = split_summary.get("num_images", None)
    if total is None:
        return stats_dict

    n_with = stats_dict.get("num_images_with_candidate", 0)
    stats_dict["num_images"] = total
    stats_dict["num_images_with_candidate"] = n_with
    stats_dict["num_images_without_candidate"] = total - n_with
    return stats_dict


def _compute_auc(labels, scores):
    """
    Wilcoxon-Mann-Whitney AUC 计算（不依赖 sklearn）。

    等价于标准 ROC AUC：遍历所有 (正, 负) 配对，
    统计正样本 score > 负样本 score 的比例（tied 算 0.5）。

    通过排序实现 O(N log N)。
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # 按 score 升序排列，计算正样本秩和
    order = np.argsort(scores)  # 升序
    sorted_labels = labels[order]

    # 分配秩（1-based），处理 tied scores
    sorted_scores = scores[order]
    n = len(sorted_scores)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # tied group [i, j): 平均秩
        avg_rank = (i + 1 + j) / 2.0  # 1-based
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # AUC = (正样本秩和 - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    pos_rank_sum = float(ranks[sorted_labels == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


# ============================================================
# 7. 保存与加载
# ============================================================
def save_checkpoint(model, filepath):
    """保存模型权重。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"  已保存模型: {filepath}")


def save_feature_config(scaler, filepath):
    """
    保存 feature_config.json，供推理时复用：
      - feature_columns / mean / std（标准化参数）
      - hidden_dims / dropout（模型结构参数）
      - tau / t_good / t_drop / reg_loss_type（训练配置）
    """
    config = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "tau": TAU,
        "t_good": T_GOOD,
        "t_drop": T_DROP,
        "reg_loss_type": REG_LOSS_TYPE,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  已保存 feature_config: {filepath}")


def save_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions_csv(predictions, filepath):
    """保存 val_predictions.csv。"""
    if not predictions:
        return
    columns = [
        "split", "image_id", "image_stem", "candidate_rank",
        "label", "det_score", "reranker_score", "endpoint_err", "is_best_match",
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            writer.writerow({k: row.get(k, "") for k in columns})
    print(f"  已保存 predictions: {filepath}  ({len(predictions)} 行)")


# ============================================================
# 8. 随机种子
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# main
# ============================================================
def main():
    print("=" * 60)
    print("  Stage-2 Reranker Training (soft target + image weight)")
    print(f"  EXPORT_DIR  = {EXPORT_DIR}")
    print(f"  OUTPUT_DIR  = {OUTPUT_DIR}")
    print(f"  FEATURES    = {len(FEATURE_COLUMNS)} columns")
    print(f"  HIDDEN_DIMS = {HIDDEN_DIMS}")
    print(f"  DROPOUT     = {DROPOUT}")
    print(f"  BATCH_SIZE  = {BATCH_SIZE}")
    print(f"  NUM_EPOCHS  = {NUM_EPOCHS}")
    print(f"  LR          = {LR}")
    print(f"  TAU         = {TAU}")
    print(f"  T_GOOD      = {T_GOOD}")
    print(f"  T_DROP      = {T_DROP}")
    print(f"  LOSS        = {REG_LOSS_TYPE}")
    print(f"  BEST_METRIC = {BEST_METRIC}")
    print(f"  SEED        = {RANDOM_SEED}")
    print("=" * 60)

    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    output_dir = _resolve(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # [1/7] 加载 CSV + split summary
    # ================================================================
    print("\n[1/7] 加载训练与验证 CSV...")
    train_csv_path = _get_csv_path(TRAIN_CSV, EXPORT_DIR, "train")
    val_csv_path = _get_csv_path(VAL_CSV, EXPORT_DIR, "val")

    train_rows = load_csv(train_csv_path)
    val_rows = load_csv(val_csv_path)

    # 尝试读取导出时的 split summary（用于修正 image-level 统计）
    val_split_summary = load_split_summary(EXPORT_DIR, "val")
    if val_split_summary is not None:
        print(f"  已加载 val split summary: num_images={val_split_summary.get('num_images')}")
    else:
        print("  [INFO] val_summary.json 不存在，image-level 统计将使用 CSV 近似值")

    # ================================================================
    # [2/7] 提取特征 + 清洗 + 标准化 + soft target + image weight
    # ================================================================
    print("\n[2/7] 特征提取与标准化...")
    X_train_raw, meta_train = extract_features_and_meta(train_rows, FEATURE_COLUMNS)
    X_val_raw, meta_val = extract_features_and_meta(val_rows, FEATURE_COLUMNS)

    X_train_clean = clean_features(X_train_raw)
    X_val_clean = clean_features(X_val_raw)

    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X_train_clean)
    X_val = scaler.transform(X_val_clean)

    # soft target
    target_train = compute_soft_targets(meta_train, TAU)
    target_val = compute_soft_targets(meta_val, TAU)

    # image weight
    weight_train, train_img_stats = compute_image_weights(meta_train, T_GOOD, T_DROP)
    weight_val, val_img_stats = compute_image_weights(meta_val, T_GOOD, T_DROP)

    n_label_pos_train = sum(1 for m in meta_train if m["label"] == 1)
    n_label_neg_train = len(meta_train) - n_label_pos_train
    print(f"  Train: {X_train.shape[0]} 样本, hard_label: {n_label_pos_train} pos / {n_label_neg_train} neg")
    print(f"    soft target mean={target_train.mean():.4f}, "
          f"image weight: {train_img_stats['num_images_used_for_training']}/{train_img_stats['num_images_total']} used, "
          f"{train_img_stats['num_images_dropped_by_weight']} dropped")

    n_label_pos_val = sum(1 for m in meta_val if m["label"] == 1)
    n_label_neg_val = len(meta_val) - n_label_pos_val
    print(f"  Val  : {X_val.shape[0]} 样本, hard_label: {n_label_pos_val} pos / {n_label_neg_val} neg")
    print(f"    soft target mean={target_val.mean():.4f}, "
          f"image weight: {val_img_stats['num_images_used_for_training']}/{val_img_stats['num_images_total']} used")

    # 保存 feature_config
    feature_config_path = os.path.join(output_dir, "feature_config.json")
    save_feature_config(scaler, feature_config_path)

    # ================================================================
    # [3/7] 构建 Dataset & DataLoader
    # ================================================================
    print("\n[3/7] 构建 DataLoader...")
    train_dataset = CandidateDataset(X_train, target_train, weight_train, meta_train)
    val_dataset = CandidateDataset(X_val, target_val, weight_val, meta_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ================================================================
    # [4/7] 构建模型 + 损失 + 优化器
    # ================================================================
    print("\n[4/7] 构建模型...")
    input_dim = len(FEATURE_COLUMNS)
    model = RerankerMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model.to(device)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_regression_criterion(REG_LOSS_TYPE)
    print(f"  损失函数: {REG_LOSS_TYPE} (reduction=none, weighted by image_weight)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # ================================================================
    # [5/7] 训练循环
    # ================================================================
    print("\n[5/7] 开始训练...")
    metrics_history = []
    best_val_score = float("inf")    # 越小越好（mean_endpoint_err 或 val_loss）
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- train ----
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ---- validate ----
        val_results = evaluate(model, val_loader, val_dataset, criterion, device,
                              split_summary=val_split_summary)
        cand_m = val_results["candidate_level_metrics"]
        img_reranker = val_results["image_level_metrics_reranker"]
        img_det = val_results["image_level_metrics_det_score_baseline"]

        val_loss = cand_m["val_loss"]
        val_auc = cand_m["auc"]
        reranker_mean_err = img_reranker.get("mean_endpoint_err", float("nan"))
        det_mean_err = img_det.get("mean_endpoint_err", float("nan"))

        # ---- best model 判定 ----
        if BEST_METRIC == "image_mean_err":
            current_score = reranker_mean_err if math.isfinite(reranker_mean_err) else val_loss
        else:
            current_score = val_loss

        is_best = current_score < best_val_score
        if is_best:
            best_val_score = current_score
            best_epoch = epoch
            save_checkpoint(model, os.path.join(output_dir, "best_model.pth"))
            best_val_results = {
                "candidate_level_metrics": cand_m,
                "image_level_metrics_reranker": img_reranker,
                "image_level_metrics_det_score_baseline": img_det,
                "image_weight_stats": val_img_stats,
            }
            if "image_level_metrics_heuristic_baseline" in val_results:
                best_val_results["image_level_metrics_heuristic_baseline"] = (
                    val_results["image_level_metrics_heuristic_baseline"]
                )
            _best_predictions = val_results.get("predictions", [])

        # ---- 记录 ----
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": cand_m["f1"],
            "reranker_mean_err": reranker_mean_err,
            "det_baseline_mean_err": det_mean_err,
            "is_best": is_best,
        }
        metrics_history.append(epoch_record)

        # ---- 打印 ----
        best_mark = " *BEST*" if is_best else ""
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train={train_loss:.4f}  "
              f"val={val_loss:.4f}  "
              f"auc={val_auc:.4f}  "
              f"reranker_err={reranker_mean_err:.2f}  "
              f"det_err={det_mean_err:.2f}"
              f"{best_mark}")

    # ---- 保存 last model ----
    save_checkpoint(model, os.path.join(output_dir, "last_model.pth"))

    # ================================================================
    # [6/7] 保存训练日志与最终评估
    # ================================================================
    print("\n[6/7] 保存训练日志与评估结果...")

    # 训练日志
    log_path = os.path.join(output_dir, "metrics_history.json")
    save_json(metrics_history, log_path)
    print(f"  已保存训练日志: {log_path}  ({len(metrics_history)} epochs)")

    # best epoch 的验证报告
    best_val_results["best_epoch"] = best_epoch
    best_val_results["best_metric"] = BEST_METRIC
    best_val_results["best_metric_value"] = best_val_score
    best_val_results["training_config"] = {
        "tau": TAU, "t_good": T_GOOD, "t_drop": T_DROP,
        "reg_loss_type": REG_LOSS_TYPE, "lr": LR,
        "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
    }
    best_val_results["train_image_weight_stats"] = train_img_stats
    val_results_path = os.path.join(output_dir, "val_results_best.json")
    save_json(best_val_results, val_results_path)
    print(f"  已保存最佳验证结果: {val_results_path}")

    # val_predictions.csv
    pred_csv_path = os.path.join(output_dir, "val_predictions.csv")
    save_predictions_csv(_best_predictions, pred_csv_path)

    # ---- 最终汇总 ----
    print("\n" + "=" * 60)
    print(f"  [7/7] 训练完成")
    print(f"  Best epoch: {best_epoch}  ({BEST_METRIC} = {best_val_score:.4f})")
    print(f"  输出目录: {output_dir}")
    print()

    # 打印 best epoch 的 image-level 对比
    img_r = best_val_results["image_level_metrics_reranker"]
    img_d = best_val_results["image_level_metrics_det_score_baseline"]
    print(f"  === Image-Level Endpoint Error (Best Epoch {best_epoch}) ===")
    print(f"  {'Method':<25s} {'Mean':>8s} {'Median':>8s} {'<=5':>7s} {'<=10':>7s} {'<=20':>7s} {'<=50':>7s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for tag, d in [("Reranker", img_r), ("Det-Score Baseline", img_d)]:
        print(f"  {tag:<25s} "
              f"{d.get('mean_endpoint_err', float('nan')):8.2f} "
              f"{d.get('median_endpoint_err', float('nan')):8.2f} "
              f"{d.get('pct_le_5', float('nan')):6.1f}% "
              f"{d.get('pct_le_10', float('nan')):6.1f}% "
              f"{d.get('pct_le_20', float('nan')):6.1f}% "
              f"{d.get('pct_le_50', float('nan')):6.1f}%")
    if "image_level_metrics_heuristic_baseline" in best_val_results:
        img_h = best_val_results["image_level_metrics_heuristic_baseline"]
        print(f"  {'Heuristic Baseline':<25s} "
              f"{img_h.get('mean_endpoint_err', float('nan')):8.2f} "
              f"{img_h.get('median_endpoint_err', float('nan')):8.2f} "
              f"{img_h.get('pct_le_5', float('nan')):6.1f}% "
              f"{img_h.get('pct_le_10', float('nan')):6.1f}% "
              f"{img_h.get('pct_le_20', float('nan')):6.1f}% "
              f"{img_h.get('pct_le_50', float('nan')):6.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
