"""
export_reranker_data.py — 第二阶段 reranker 训练数据导出脚本

从已训练的 LINEA / LINEA_ENTROPY 模型的 top-K 候选线中提取特征，
为后续独立 reranker 训练准备候选线表（CSV + summary JSON）。

复用已有接口：
  - 配置加载：LINEA/util/slconfig.SLConfig
  - 模型构建：LINEA/models/registry.MODULE_BUILD_FUNCS
  - 后处理  ：LINEA/models/linea/linea.PostProcess
  - 数据集  ：stage1_linea_entropy/datasets.build_musid_dataset
  - Collate  ：LINEA/datasets/collate.BatchImageCollateFunction

baseline vs entropy 模式的熵特征获取策略：
  - entropy 模式：直接使用 batch 中的 entropy_map（dataset include_entropy=True）。
  - baseline 模式：dataset include_entropy=False（2-tuple），
    通过 stem → entropy 目录离线回查 .npy 文件来获取 entropy map。

用法：
  在 PyCharm 中修改顶部全局变量后直接运行本文件。
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# 项目路径设置
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent           # stage2_reranker/
_PROJECT_ROOT = _SCRIPT_DIR.parent                      # sealine_detection/
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "LINEA"))

# ============================================================
# 顶部全局变量配置 — 在 PyCharm 中直接修改
# ============================================================
MODE = "entropy"                 # "baseline" 或 "entropy"
CONFIG_FILE = "stage1_linea_entropy/configs/linea_entropy_musid.py"
WEIGHTS_PATH = "output/linea_entropy_musid/checkpoint.pth"
DEVICE = "cuda"

IMG_DIR = "Hashmani's Dataset/MU-SID"
ENTROPY_DIR = "Hashmani's Dataset/MU-SID_entropy_blue"
SPLIT_DIR = "splits_musid"

IMG_SIZE = 640
BATCH_SIZE = 1                   # 建议保持 1，逐张处理便于特征对齐
NUM_WORKERS = 0

# 候选线导出参数
TOPK_CANDIDATES = 50             # 每张图导出的 top-K 候选线数
POSITIVE_ERR_THRESH = 20.0       # label=1 的 endpoint_err 阈值（letterbox px）
MAX_HORIZON_DEV_DEG = 15.0       # 启发式角度门控
MIN_LENGTH_RATIO = 0.2           # 启发式长度门控

# 双侧熵采样参数
ENT_BAND_WIDTH = 10              # 沿法向采样带宽度（像素）
ENT_BAND_OFFSET = 2              # 法向偏移起点（像素，避免采在线上）
ENT_NUM_SAMPLES = 20             # 沿线方向均匀采样点数

# 梯度特征参数
GRAD_NUM_SAMPLES = 20            # 沿线方向均匀采样点数
GRAD_BAND_WIDTH = 5              # 法向带状采样半径（像素）

# 输出
SAVE_ROOT = "stage2_reranker/exports"
SAVE_MERGED = True               # 是否额外保存 train+val 合并结果
SPLITS_TO_EXPORT = ["train", "val"]   # 导出哪些 split


# ============================================================
# LINEA 归一化参数
# ============================================================
MEAN = np.array([0.538, 0.494, 0.453], dtype=np.float32)
STD = np.array([0.257, 0.263, 0.273], dtype=np.float32)

# CSV 列定义
CSV_COLUMNS = [
    # 一、基础标识
    "split", "image_id", "image_stem", "image_path", "candidate_rank",
    # 二、几何与检测特征
    "det_score", "length", "length_norm", "horizon_dev_deg",
    "x_span", "x_span_norm", "y_mid", "y_mid_norm",
    # 三、双侧局部熵特征
    "ent_upper", "ent_lower", "delta_ent", "ent_consistency",
    # 四、梯度特征
    "grad_normal", "grad_tangent", "grad_ratio", "grad_consistency",
    # 五、与 GT 的关系
    "endpoint_err", "is_best_match", "label",
    # 六、启发式辅助字段
    "pass_angle_gate", "pass_length_gate", "heuristic_score",
]


# ============================================================
# 路径工具
# ============================================================
def _resolve_path(p):
    """将相对路径基于项目根目录解析为绝对路径。"""
    if not p:
        return p
    pp = Path(p)
    if not pp.is_absolute():
        pp = _PROJECT_ROOT / pp
    return str(pp)


def _get_entropy_dir_abs():
    """返回 entropy 目录的绝对路径。"""
    return str(_PROJECT_ROOT / ENTROPY_DIR) if not Path(ENTROPY_DIR).is_absolute() else ENTROPY_DIR


# ============================================================
# 1. 配置加载（复用 test_linea_stage1 逻辑）
# ============================================================
def load_config(config_file):
    from util.slconfig import SLConfig

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    cfg = SLConfig.fromfile(str(config_path))
    args = cfg

    sz = getattr(args, 'eval_spatial_size', IMG_SIZE)
    if isinstance(sz, int):
        sz = [sz, sz]
    args.eval_spatial_size = sz

    # 数据目录（相对路径基于项目根目录解析）
    args.musid_img_dir = str(_PROJECT_ROOT / IMG_DIR) if not Path(IMG_DIR).is_absolute() else IMG_DIR
    args.musid_entropy_dir = _get_entropy_dir_abs()
    args.musid_split_dir = str(_PROJECT_ROOT / SPLIT_DIR) if not Path(SPLIT_DIR).is_absolute() else SPLIT_DIR

    args.pretrained = False
    return args


# ============================================================
# 2. 模型构建（复用 test_linea_stage1 逻辑）
# ============================================================
def build_model(args, mode):
    from models.registry import MODULE_BUILD_FUNCS
    model_name = "LINEA" if mode == "baseline" else "LINEA_ENTROPY"
    args.modelname = model_name
    assert model_name in MODULE_BUILD_FUNCS._module_dict, (
        f"模型 '{model_name}' 未注册。请检查 LINEA/models/__init__.py"
    )
    build_fn = MODULE_BUILD_FUNCS.get(model_name)
    model, postprocessor = build_fn(args)
    return model, postprocessor


def load_weights(model, weights_path, device):
    weights_path = _resolve_path(weights_path)
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[WARN] 权重文件不存在或未指定: {weights_path}")
        print("[WARN] 使用随机初始化权重（仅供调试）")
        return
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    info = model.load_state_dict(state_dict, strict=False)
    if info.missing_keys:
        print(f"[INFO] Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:10]:
            print(f"  - {k}")
    if info.unexpected_keys:
        print(f"[INFO] Unexpected keys ({len(info.unexpected_keys)}):")
        for k in info.unexpected_keys[:10]:
            print(f"  - {k}")
    print(f"[OK] 权重已加载: {weights_path}")


# ============================================================
# 3. 数据集构建
# ============================================================
def build_split_dataset(image_set, args, mode):
    """
    构建指定 split 的数据集。

    entropy 模式：include_entropy=True，返回 4-tuple (image, target, entropy_map, meta)。
    baseline 模式：include_entropy=False，返回 2-tuple (image, target)。
      baseline 下的 entropy map 通过 stem 离线回查获取。

    导出时关闭训练增强（augmentation），通过构建后覆盖 is_train=False 实现。
    """
    from stage1_linea_entropy.datasets import build_musid_dataset

    args.entropy_mode = mode

    dataset = build_musid_dataset(image_set, args)

    # 关闭训练增强（导出时不做 hflip / color jitter）
    dataset.is_train = False

    return dataset


def build_collate_fn():
    from datasets.collate import BatchImageCollateFunction
    return BatchImageCollateFunction(base_size=IMG_SIZE)


# ============================================================
# 4. baseline 模式下的 entropy map 回查
# ============================================================
_IMAGE_EXTS = ('', '.JPG', '.jpg', '.png', '.jpeg', '.JPEG', '.PNG')


def resolve_sample_identity(target, meta, global_idx, dataset):
    """
    稳健地从 batch 中恢复样本身份信息（stem, img_path, image_id）。

    优先级：
      1. meta dict（entropy 模式 4-tuple 才有）
      2. dataset.data 按 global_idx 查行
      3. fallback 到 global_idx

    Returns:
        (stem: str, img_path: str, image_id: int)
    """
    stem = ''
    img_path = ''

    # ---- 从 meta 中拿 ----
    if meta is not None:
        stem = meta.get('stem', '')
        img_path = meta.get('img_path', '')

    # ---- 从 dataset.data 按 index 拿 ----
    if not stem and dataset is not None and hasattr(dataset, 'data'):
        if 0 <= global_idx < len(dataset.data):
            stem = str(dataset.data.iloc[global_idx].iloc[0])

    # ---- fallback ----
    if not stem:
        stem = f"sample_{global_idx}"

    # ---- image_id: 稳健提取 ----
    image_id = _extract_image_id(target, global_idx)

    return stem, img_path, image_id


def _extract_image_id(target, fallback_idx):
    """
    从 target dict 中稳健提取 image_id（兼容 int / 0-dim tensor / shape=[1] tensor）。
    """
    raw = target.get('image_id', fallback_idx)
    if isinstance(raw, (int, float)):
        return int(raw)
    if torch.is_tensor(raw):
        return int(raw.item()) if raw.dim() == 0 else int(raw[0].item())
    if isinstance(raw, np.ndarray):
        return int(raw.item()) if raw.ndim == 0 else int(raw[0])
    return int(fallback_idx)


def find_entropy_map_path_by_stem(stem, entropy_dir=None):
    """
    根据 stem 在 entropy 目录中查找对应 .npy 文件。

    查找策略：
      1. 直接 stem + '.npy'
      2. 去掉 stem 的扩展名后 + '.npy'
      3. 遍历常见图片扩展名，去掉后 + '.npy'

    Args:
        stem       : str — 来自 CSV / meta 的样本标识
        entropy_dir: str or None — entropy 目录绝对路径

    Returns:
        str or None — 匹配到的 .npy 路径，或 None
    """
    if entropy_dir is None:
        entropy_dir = _get_entropy_dir_abs()
    if not os.path.isdir(entropy_dir):
        return None

    # 直接匹配
    p = os.path.join(entropy_dir, f"{stem}.npy")
    if os.path.isfile(p):
        return p

    # 去掉 stem 自带的扩展名
    base = Path(stem).stem
    if base != stem:
        p = os.path.join(entropy_dir, f"{base}.npy")
        if os.path.isfile(p):
            return p

    return None


def load_entropy_map_for_baseline(stem, img_size, dataset, global_idx):
    """
    baseline 模式下，根据 stem 离线回查 entropy map 并做 letterbox 变换。

    回查顺序：
      1. stem → entropy 目录匹配 .npy
      2. dataset.data[global_idx] 的 stem → entropy 目录匹配
      3. 全部失败 → 返回 None

    如果找到 .npy 文件，还需要做与训练一致的 letterbox 变换：
      - 等比缩放到 img_size
      - 右下角零填充
      - clip / 8.0 归一化到 [0, 1]

    Args:
        stem       : str
        img_size   : int
        dataset    : MUSIDLineaEntropyDataset（用于拿 scale/size 信息和 fallback stem）
        global_idx : int

    Returns:
        ndarray [img_size, img_size] float32 in [0, 1]，或 None
    """
    entropy_dir = _get_entropy_dir_abs()

    # ---- 策略 1: 直接用 stem 匹配 ----
    ent_path = find_entropy_map_path_by_stem(stem, entropy_dir)

    # ---- 策略 2: fallback 到 dataset 行的 stem ----
    if ent_path is None and dataset is not None and hasattr(dataset, 'data'):
        if 0 <= global_idx < len(dataset.data):
            ds_stem = str(dataset.data.iloc[global_idx].iloc[0])
            if ds_stem != stem:
                ent_path = find_entropy_map_path_by_stem(ds_stem, entropy_dir)

    if ent_path is None:
        return None

    # ---- 加载 + letterbox 变换 ----
    try:
        ent_np = np.load(ent_path).astype(np.float32)
    except Exception:
        return None

    # 需要知道 letterbox 参数：与 dataset 中的逻辑一致
    # 从原始 entropy 图尺寸推算 scale
    orig_h, orig_w = ent_np.shape[:2]
    scale = min(img_size / orig_w, img_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    ent_resized = cv2.resize(ent_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ent_resized = np.clip(ent_resized / 8.0, 0.0, 1.0).astype(np.float32)
    ent_out = np.zeros((img_size, img_size), dtype=np.float32)
    ent_out[:new_h, :new_w] = ent_resized

    return ent_out


# ============================================================
# 5. 图像反归一化工具
# ============================================================
def denorm_image_to_gray(img_tensor):
    """
    将归一化后的图像 tensor 转为 uint8 灰度图，用于梯度计算。

    Args:
        img_tensor : Tensor [3, H, W]

    Returns:
        ndarray [H, W] uint8 灰度图
    """
    img = img_tensor.cpu().numpy().copy()  # [3, H, W]
    for c in range(3):
        img[c] = img[c] * STD[c] + MEAN[c]
    img = np.clip(img, 0, 1)
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return (gray * 255).astype(np.uint8)


# ============================================================
# 6. 几何特征计算
# ============================================================
def compute_geometry_features(line, img_size):
    """
    计算单条候选线的几何特征。

    Args:
        line : ndarray [4] — (x1, y1, x2, y2) 像素坐标
        img_size : int — 正方形尺寸

    Returns:
        dict of features
    """
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx ** 2 + dy ** 2)
    angle = math.degrees(math.atan2(dy, dx))
    dev = min(abs(angle), 180.0 - abs(angle))
    x_span = abs(dx)
    y_mid = (y1 + y2) / 2.0

    return {
        'length': length,
        'length_norm': length / img_size,
        'horizon_dev_deg': dev,
        'x_span': x_span,
        'x_span_norm': x_span / img_size,
        'y_mid': y_mid,
        'y_mid_norm': y_mid / img_size,
    }


# ============================================================
# 7. 双侧局部熵特征
# ============================================================
def compute_entropy_features(line, ent_map, img_size):
    """
    沿候选线法向，在上下两侧采样窄带区域，计算局部熵均值。

    ent_map: ndarray [H, W] float32, 已做 letterbox + 归一化 [0, 1]。
             若为 None 则返回全 NaN。

    Args:
        line    : ndarray [4] — (x1, y1, x2, y2) 像素坐标
        ent_map : ndarray [H, W] 或 None
        img_size: int

    Returns:
        dict 包含 ent_upper, ent_lower, delta_ent, ent_consistency
    """
    nan_result = {
        'ent_upper': float('nan'),
        'ent_lower': float('nan'),
        'delta_ent': float('nan'),
        'ent_consistency': float('nan'),
    }

    if ent_map is None:
        return nan_result

    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length < 1e-3:
        return nan_result

    # 单位切向量和法向量（法向量指向"上方"：取使 ny < 0 的方向）
    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx
    if ny > 0:
        nx, ny = -nx, -ny

    H, W = ent_map.shape
    n_pts = ENT_NUM_SAMPLES

    upper_vals = []
    lower_vals = []

    for i in range(n_pts):
        t = (i + 0.5) / n_pts
        cx = x1 + t * dx
        cy = y1 + t * dy

        for d in range(ENT_BAND_OFFSET, ENT_BAND_OFFSET + ENT_BAND_WIDTH):
            ux = int(round(cx + d * nx))
            uy = int(round(cy + d * ny))
            if 0 <= ux < W and 0 <= uy < H:
                upper_vals.append(ent_map[uy, ux])

            lx = int(round(cx - d * nx))
            ly = int(round(cy - d * ny))
            if 0 <= lx < W and 0 <= ly < H:
                lower_vals.append(ent_map[ly, lx])

    if len(upper_vals) == 0 or len(lower_vals) == 0:
        return nan_result

    ent_upper = float(np.mean(upper_vals))
    ent_lower = float(np.mean(lower_vals))
    delta_ent = ent_lower - ent_upper

    # ent_consistency: 沿线方向逐段计算局部 delta_ent 的标准差倒数
    seg_deltas = []
    seg_size = max(1, n_pts // 5)
    for seg_start in range(0, n_pts, seg_size):
        seg_end = min(seg_start + seg_size, n_pts)
        seg_upper = []
        seg_lower = []
        for i in range(seg_start, seg_end):
            t = (i + 0.5) / n_pts
            cx = x1 + t * dx
            cy = y1 + t * dy
            for d in range(ENT_BAND_OFFSET, ENT_BAND_OFFSET + ENT_BAND_WIDTH):
                ux = int(round(cx + d * nx))
                uy = int(round(cy + d * ny))
                if 0 <= ux < W and 0 <= uy < H:
                    seg_upper.append(ent_map[uy, ux])
                lx = int(round(cx - d * nx))
                ly = int(round(cy - d * ny))
                if 0 <= lx < W and 0 <= ly < H:
                    seg_lower.append(ent_map[ly, lx])
        if seg_upper and seg_lower:
            seg_deltas.append(np.mean(seg_lower) - np.mean(seg_upper))

    if len(seg_deltas) >= 2:
        ent_consistency = 1.0 / (float(np.std(seg_deltas)) + 1e-6)
    else:
        ent_consistency = float('nan')

    return {
        'ent_upper': ent_upper,
        'ent_lower': ent_lower,
        'delta_ent': delta_ent,
        'ent_consistency': ent_consistency,
    }


# ============================================================
# 8. 梯度特征（带状采样）
# ============================================================
def compute_gradient_features(line, gray_img, img_size):
    """
    基于灰度图，沿候选线做带状采样计算法向/切向梯度响应。

    对沿线方向的每个采样点，在法向 [-GRAD_BAND_WIDTH, +GRAD_BAND_WIDTH] 范围内
    取梯度并投影到法向/切向，然后对整个带区域取均值。

    Args:
        line     : ndarray [4] — (x1, y1, x2, y2) 像素坐标
        gray_img : ndarray [H, W] uint8 灰度图
        img_size : int

    Returns:
        dict 包含 grad_normal, grad_tangent, grad_ratio, grad_consistency
    """
    nan_result = {
        'grad_normal': float('nan'),
        'grad_tangent': float('nan'),
        'grad_ratio': float('nan'),
        'grad_consistency': float('nan'),
    }

    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length < 1e-3:
        return nan_result

    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx

    # Sobel 梯度（整张图只算一次）
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    H, W = gray_img.shape
    n_pts = GRAD_NUM_SAMPLES
    bw = GRAD_BAND_WIDTH

    # 每个沿线采样点的带状区域均值，用于 consistency 计算
    per_point_normal = []     # 每个点的带状区域法向均值
    per_point_tangent = []

    all_normal = []
    all_tangent = []

    for i in range(n_pts):
        t = (i + 0.5) / n_pts
        cx = x1 + t * dx
        cy = y1 + t * dy

        pt_normal = []
        pt_tangent = []

        # 在法向 [-bw, +bw] 范围内采样
        for d in range(-bw, bw + 1):
            sx = int(round(cx + d * nx))
            sy = int(round(cy + d * ny))

            if 0 <= sx < W and 0 <= sy < H:
                gx = grad_x[sy, sx]
                gy = grad_y[sy, sx]
                pt_normal.append(abs(gx * nx + gy * ny))
                pt_tangent.append(abs(gx * tx + gy * ty))

        if pt_normal:
            mn = float(np.mean(pt_normal))
            mt = float(np.mean(pt_tangent))
            per_point_normal.append(mn)
            per_point_tangent.append(mt)
            all_normal.extend(pt_normal)
            all_tangent.extend(pt_tangent)

    if len(all_normal) == 0:
        return nan_result

    grad_normal = float(np.mean(all_normal))
    grad_tangent = float(np.mean(all_tangent))
    grad_ratio = grad_normal / (grad_tangent + 1e-6)

    # grad_consistency: 逐采样点法向均值的一致性（mean / std）
    if len(per_point_normal) >= 2 and grad_normal > 1e-6:
        grad_consistency = float(np.mean(per_point_normal)) / (float(np.std(per_point_normal)) + 1e-6)
    else:
        grad_consistency = 0.0

    return {
        'grad_normal': grad_normal,
        'grad_tangent': grad_tangent,
        'grad_ratio': grad_ratio,
        'grad_consistency': grad_consistency,
    }


# ============================================================
# 9. Endpoint error（复用 test_linea_stage1 逻辑）
# ============================================================
def compute_endpoint_error(pred_line, gt_line):
    """Endpoint error（letterbox 640×640 坐标系），端点顺序无关。"""
    p1, p2 = pred_line[:2], pred_line[2:]
    g1, g2 = gt_line[:2], gt_line[2:]
    err_a = (np.linalg.norm(p1 - g1) + np.linalg.norm(p2 - g2)) / 2.0
    err_b = (np.linalg.norm(p1 - g2) + np.linalg.norm(p2 - g1)) / 2.0
    return min(err_a, err_b)


def gt_line_from_target(target, img_size):
    """从 target 中提取 GT 线段像素坐标（letterbox 坐标系）。"""
    lines = target['lines']
    if torch.is_tensor(lines):
        lines = lines.cpu().numpy()
    return (lines[0] * img_size).astype(np.float64)


# ============================================================
# 10. 启发式辅助字段
# ============================================================
def compute_heuristic_fields(line, score, img_size):
    """与第一阶段启发式规则对应的辅助字段，便于后续对比。"""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx ** 2 + dy ** 2)
    angle = math.degrees(math.atan2(dy, dx))
    dev = min(abs(angle), 180.0 - abs(angle))

    pass_angle = 1 if dev <= MAX_HORIZON_DEV_DEG else 0
    pass_length = 1 if length >= MIN_LENGTH_RATIO * img_size else 0
    heuristic_score = float(score) if (pass_angle and pass_length) else 0.0

    return {
        'pass_angle_gate': pass_angle,
        'pass_length_gate': pass_length,
        'heuristic_score': heuristic_score,
    }


# ============================================================
# 11. 单张图处理：提取所有候选线的特征行
# ============================================================
def process_one_image(split_name, image_id, stem, img_path,
                      lines_np, scores_np, gt_line,
                      img_tensor, ent_map, img_size):
    """
    处理单张图像的 top-K 候选线，返回 list of dict（每条线一行）。

    Args:
        split_name : str — "train" / "val"
        image_id   : int
        stem       : str
        img_path   : str
        lines_np   : ndarray [N, 4] — top-K 候选线（像素坐标）
        scores_np  : ndarray [N] — 对应分数
        gt_line    : ndarray [4] — GT 线段（像素坐标）
        img_tensor : Tensor [3, H, W] — 归一化图像（用于梯度）
        ent_map    : ndarray [H, W] 或 None — letterbox 后的 entropy map
        img_size   : int

    Returns:
        list of dict（每个 dict 对应一条候选线的完整特征行）
    """
    rows = []
    if len(lines_np) == 0:
        return rows

    gray_img = denorm_image_to_gray(img_tensor)

    # 先算所有候选线的 endpoint_err，找 best match
    endpoint_errs = np.array([
        compute_endpoint_error(lines_np[k], gt_line) for k in range(len(lines_np))
    ])
    best_match_idx = int(np.argmin(endpoint_errs))
    best_match_err = endpoint_errs[best_match_idx]

    for k in range(len(lines_np)):
        line = lines_np[k]
        score = float(scores_np[k])

        geo = compute_geometry_features(line, img_size)
        ent = compute_entropy_features(line, ent_map, img_size)
        grad = compute_gradient_features(line, gray_img, img_size)

        err = endpoint_errs[k]
        is_best = 1 if k == best_match_idx else 0
        label = 1 if (is_best and best_match_err <= POSITIVE_ERR_THRESH) else 0

        heur = compute_heuristic_fields(line, score, img_size)

        row = {
            'split': split_name,
            'image_id': image_id,
            'image_stem': stem,
            'image_path': img_path or '',
            'candidate_rank': k,
            'det_score': score,
            **geo,
            **ent,
            **grad,
            'endpoint_err': float(err),
            'is_best_match': is_best,
            'label': label,
            **heur,
        }
        rows.append(row)

    return rows


# ============================================================
# 12. 推理 + 特征提取（单个 split）
# ============================================================
@torch.no_grad()
def export_one_split(split_name, model, postprocessor, dataloader, dataset,
                     device, mode, img_size):
    """
    对一个 split 做推理 + 特征提取，返回所有候选线行。

    entropy 模式：
      batch 是 4-tuple (images, targets, entropy_maps, metas)。
      entropy_map 同时用于模型推理和特征提取。

    baseline 模式：
      batch 是 2-tuple (images, targets)。
      模型推理时不传 entropy_map。
      特征提取时通过 stem 离线回查 entropy map。
    """
    model.eval()
    all_rows = []
    num_images_processed = 0
    num_images_no_lines = 0
    images_seen = set()         # 跟踪所有已处理的 stem
    images_with_positive = set()
    ent_lookup_miss = 0         # baseline 模式下熵图回查失败计数

    is_entropy_mode = (mode == "entropy")

    for batch_idx, batch in enumerate(dataloader):
        # ---- 解包 batch ----
        if is_entropy_mode:
            # 4-tuple: (images, targets, entropy_maps, metas)
            if len(batch) == 4:
                images, targets, entropy_maps, metas = batch
            elif len(batch) == 3:
                images, targets, entropy_maps = batch
                metas = None
            else:
                images, targets = batch[:2]
                entropy_maps = None
                metas = None
        else:
            # baseline: 2-tuple (images, targets)
            images, targets = batch[:2]
            entropy_maps = None
            metas = None

        images = images.to(device)

        # ---- 前向推理 ----
        if is_entropy_mode and entropy_maps is not None:
            entropy_maps_dev = entropy_maps.to(device)
            outputs = model(images, entropy_map=entropy_maps_dev)
        else:
            outputs = model(images)

        # ---- PostProcess ----
        B = images.shape[0]
        target_sizes = torch.tensor([[img_size, img_size]], device=device).repeat(B, 1)
        results_batch = postprocessor(outputs, target_sizes)

        # ---- 逐张处理 ----
        for i in range(B):
            global_idx = batch_idx * BATCH_SIZE + i
            num_images_processed += 1

            # ---- 样本身份信息 ----
            meta_i = metas[i] if metas is not None else None
            stem, img_path, image_id = resolve_sample_identity(
                targets[i], meta_i, global_idx, dataset)
            images_seen.add(stem)

            # ---- GT 线段 ----
            gt_line = gt_line_from_target(targets[i], img_size)

            # ---- entropy map ----
            if is_entropy_mode and entropy_maps is not None:
                # entropy 模式：直接从 batch 拿
                ent_map = entropy_maps[i, 0].cpu().numpy()  # [H, W]
            else:
                # baseline 模式：离线回查
                ent_map = load_entropy_map_for_baseline(
                    stem, img_size, dataset, global_idx)
                if ent_map is None:
                    ent_lookup_miss += 1

            # ---- 模型预测 ----
            pred_lines = results_batch[i]['lines'].cpu().numpy()
            pred_scores = results_batch[i]['scores'].cpu().numpy()

            # top-K 筛选
            if len(pred_scores) > TOPK_CANDIDATES:
                topk_idx = np.argsort(pred_scores)[::-1][:TOPK_CANDIDATES]
                pred_lines = pred_lines[topk_idx]
                pred_scores = pred_scores[topk_idx]
            else:
                sorted_idx = np.argsort(pred_scores)[::-1]
                pred_lines = pred_lines[sorted_idx]
                pred_scores = pred_scores[sorted_idx]

            if len(pred_lines) == 0:
                num_images_no_lines += 1
                continue

            # ---- 提取特征行 ----
            rows = process_one_image(
                split_name=split_name,
                image_id=image_id,
                stem=stem,
                img_path=img_path,
                lines_np=pred_lines,
                scores_np=pred_scores,
                gt_line=gt_line,
                img_tensor=images[i],
                ent_map=ent_map,
                img_size=img_size,
            )

            # 跟踪正样本
            for r in rows:
                if r['label'] == 1:
                    images_with_positive.add(stem)
                    break

            all_rows.extend(rows)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  [{split_name}] 进度: {batch_idx + 1}/{len(dataloader)}  "
                  f"累计候选线: {len(all_rows)}")

    if not is_entropy_mode and ent_lookup_miss > 0:
        print(f"  [{split_name}] baseline 熵图回查失败: {ent_lookup_miss}/{num_images_processed}")

    print(f"  [{split_name}] 完成: {len(all_rows)} 条候选线, "
          f"{num_images_no_lines} 张图无线段")

    # 返回额外统计供 summary 使用
    split_stats = {
        'num_images_processed': num_images_processed,
        'num_images_no_lines': num_images_no_lines,
        'images_seen': images_seen,
        'images_with_positive': images_with_positive,
        'ent_lookup_miss': ent_lookup_miss,
    }

    return all_rows, split_stats


# ============================================================
# 13. 保存 CSV + summary
# ============================================================
def save_csv(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  已保存 CSV: {filepath}  ({len(rows)} 行)")


def compute_summary(rows, split_name, num_images, split_stats=None):
    """
    计算 split 级别的 summary 统计。

    保证：num_images_with_positive + num_images_without_positive == num_images
    其中"没有正样本的图像"包括完全没有候选线的图像。
    """
    num_candidates = len(rows)
    num_positive = sum(1 for r in rows if r['label'] == 1)

    # 精确统计有正样本的图像数
    if split_stats is not None:
        images_with_positive = split_stats.get('images_with_positive', set())
        num_images_no_lines = split_stats.get('num_images_no_lines', 0)
        ent_lookup_miss = split_stats.get('ent_lookup_miss', 0)
    else:
        # fallback：从 rows 中推算（用于 merged summary）
        images_with_positive = set()
        for r in rows:
            if r['label'] == 1:
                images_with_positive.add(r['image_stem'])
        num_images_no_lines = 0
        ent_lookup_miss = 0

    n_with_pos = len(images_with_positive)
    n_without_pos = num_images - n_with_pos

    # best match 的 endpoint_err 统计
    best_match_errs = [r['endpoint_err'] for r in rows if r['is_best_match'] == 1]
    best_match_errs = np.array(best_match_errs) if best_match_errs else np.array([])

    # NaN 统计
    ent_nan_count = sum(1 for r in rows if math.isnan(r.get('ent_upper', float('nan'))))
    grad_nan_count = sum(1 for r in rows if math.isnan(r.get('grad_normal', float('nan'))))

    summary = {
        'split': split_name,
        'mode': MODE,
        'weights_path': WEIGHTS_PATH,
        'positive_err_thresh': POSITIVE_ERR_THRESH,
        'topk_candidates': TOPK_CANDIDATES,
        'img_size': IMG_SIZE,
        'num_images': num_images,
        'num_images_no_lines': num_images_no_lines,
        'num_candidates': num_candidates,
        'num_positive': num_positive,
        'num_images_with_positive': n_with_pos,
        'num_images_without_positive': n_without_pos,
        'positive_rate': num_positive / max(num_candidates, 1),
        'num_ent_nan': ent_nan_count,
        'num_ent_lookup_miss': ent_lookup_miss,
        'num_grad_nan': grad_nan_count,
    }

    if len(best_match_errs) > 0:
        summary['best_match_mean_endpoint_err'] = float(np.mean(best_match_errs))
        summary['best_match_median_endpoint_err'] = float(np.median(best_match_errs))
        summary['best_match_std_endpoint_err'] = float(np.std(best_match_errs))
        summary['best_match_max_endpoint_err'] = float(np.max(best_match_errs))
        summary['best_match_pct_le_5'] = float(np.mean(best_match_errs <= 5) * 100)
        summary['best_match_pct_le_10'] = float(np.mean(best_match_errs <= 10) * 100)
        summary['best_match_pct_le_20'] = float(np.mean(best_match_errs <= 20) * 100)

    if num_candidates > 0:
        det_scores = [r['det_score'] for r in rows]
        summary['det_score_mean'] = float(np.mean(det_scores))
        summary['det_score_median'] = float(np.median(det_scores))

    return summary


def save_summary(summary, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  已保存 summary: {filepath}")


# ============================================================
# 14. 输出目录
# ============================================================
def make_output_dir():
    weights_stem = Path(WEIGHTS_PATH).stem if WEIGHTS_PATH else "no_weights"
    dir_name = f"{MODE}__{weights_stem}"
    out_dir = os.path.join(_resolve_path(SAVE_ROOT), dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ============================================================
# main
# ============================================================
def main():
    print("=" * 60)
    print(f"  Stage-2 reranker 数据导出")
    print(f"  MODE={MODE}")
    print(f"  CONFIG_FILE={CONFIG_FILE}")
    print(f"  WEIGHTS_PATH={WEIGHTS_PATH}")
    print(f"  TOPK_CANDIDATES={TOPK_CANDIDATES}")
    print(f"  POSITIVE_ERR_THRESH={POSITIVE_ERR_THRESH}")
    print(f"  SPLITS={SPLITS_TO_EXPORT}")
    print("=" * 60)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    output_dir = make_output_dir()
    print(f"  输出目录: {output_dir}")

    # ---- 加载配置 ----
    print("\n[1/4] 加载配置...")
    args = load_config(CONFIG_FILE)
    args.entropy_mode = MODE

    # ---- 构建模型 ----
    print("[2/4] 构建模型...")
    model, postprocessor = build_model(args, MODE)
    model.to(device)

    # ---- 加载权重 ----
    print("[3/4] 加载权重...")
    load_weights(model, WEIGHTS_PATH, device)

    # ---- 逐 split 导出 ----
    print("[4/4] 导出候选线特征...")
    collate_fn = build_collate_fn()
    all_merged_rows = []
    all_split_stats = []        # 收集各 split 统计，用于合并 summary
    total_num_images = 0

    for split_name in SPLITS_TO_EXPORT:
        print(f"\n--- 处理 split: {split_name} ---")

        dataset = build_split_dataset(split_name, args, MODE)
        num_images = len(dataset)
        total_num_images += num_images
        print(f"  数据集样本数: {num_images}")

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            drop_last=False,
        )

        rows, split_stats = export_one_split(
            split_name=split_name,
            model=model,
            postprocessor=postprocessor,
            dataloader=loader,
            dataset=dataset,
            device=device,
            mode=MODE,
            img_size=IMG_SIZE,
        )

        # 保存该 split 的 CSV + summary
        csv_path = os.path.join(output_dir, f"{split_name}_reranker_candidates.csv")
        save_csv(rows, csv_path)

        summary = compute_summary(rows, split_name, num_images, split_stats)
        summary_path = os.path.join(output_dir, f"{split_name}_summary.json")
        save_summary(summary, summary_path)

        all_merged_rows.extend(rows)
        all_split_stats.append(split_stats)

    # ---- 合并保存 ----
    if SAVE_MERGED and len(SPLITS_TO_EXPORT) > 1:
        print(f"\n--- 保存合并结果 ---")
        merged_csv = os.path.join(output_dir, "reranker_candidates_all.csv")
        save_csv(all_merged_rows, merged_csv)

        # 合并 split_stats
        merged_stats = {
            'num_images_processed': sum(s['num_images_processed'] for s in all_split_stats),
            'num_images_no_lines': sum(s['num_images_no_lines'] for s in all_split_stats),
            'images_seen': set().union(*(s['images_seen'] for s in all_split_stats)),
            'images_with_positive': set().union(*(s['images_with_positive'] for s in all_split_stats)),
            'ent_lookup_miss': sum(s['ent_lookup_miss'] for s in all_split_stats),
        }

        merged_summary = compute_summary(
            all_merged_rows, "all",
            num_images=total_num_images,
            split_stats=merged_stats,
        )
        merged_summary_path = os.path.join(output_dir, "summary_all.json")
        save_summary(merged_summary, merged_summary_path)

    print("\n" + "=" * 60)
    print(f"  [DONE] 导出完成")
    print(f"  输出目录: {output_dir}")
    print(f"  总候选线数: {len(all_merged_rows)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
