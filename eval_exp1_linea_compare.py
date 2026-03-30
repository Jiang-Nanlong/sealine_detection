#!/usr/bin/env python3
"""
实验 1：LINEA-L vs LINEA-N 精度对比（在服务器上运行）

对比两个模型在 MU-SID 测试集上的 sAP5/sAP10/sAP15 指标，
同时记录参数量、FLOPs、单帧推理时间。

用法：
  python eval_exp1_linea_compare.py

输出：
  eval_exp1_results.txt — 格式化结果对比表
"""

import sys
import os
import time
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---- 路径设置 ----
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "method1_linea_entropy" / "LINEA"))
sys.path.insert(1, str(_PROJECT_ROOT / "method1_linea_entropy"))
sys.path.insert(2, str(_PROJECT_ROOT))

from util.slconfig import SLConfig
from datasets import build_dataset, LineEvaluator, BatchImageCollateFunction
from engine import test
import method1_linea_entropy.models
from models.registry import MODULE_BUILD_FUNCS

# ============================================================
# 配置：两个模型
# ============================================================
MODELS = [
    {
        "name": "LINEA-L (Entropy-B-Enhanced)",
        "config": str(_PROJECT_ROOT / "method1_linea_entropy" / "configs" / "linea_entropy_b_enhanced_musid.py"),
        "weights": str(_PROJECT_ROOT / "output" / "linea_entropy_b_enhanced_musid_v2_e130" / "best_checkpoint.pth"),
    },
    {
        "name": "LINEA-N (Entropy, 轻量版)",
        "config": str(_PROJECT_ROOT / "method1_linea_entropy" / "configs" / "linea_entropy_n_musid.py"),
        "weights": str(_PROJECT_ROOT / "output" / "linea_entropy_n_musid_e200" / "best_checkpoint.pth"),
    },
]

# 如果 best_checkpoint.pth 不存在，尝试 weights/ 目录
FALLBACK_WEIGHTS = {
    "LINEA-L (Entropy-B-Enhanced)": str(_PROJECT_ROOT / "weights" / "linea_entropy_b_enhanced_best.pth"),
    "LINEA-N (Entropy, 轻量版)": str(_PROJECT_ROOT / "weights" / "linea_entropy_n_best.pth"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = str(_PROJECT_ROOT / "eval_exp1_results.txt")
WARMUP_ITERS = 10
TIMING_ITERS = 50


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_speed(model, img_size, entropy_channels, device):
    """测量单帧推理时间 (ms)。"""
    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_ent = torch.randn(1, entropy_channels, img_size, img_size, device=device)

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP_ITERS):
            try:
                model(dummy_img, entropy_map=dummy_ent)
            except Exception:
                model(dummy_img)
        torch.cuda.synchronize()

        # Timing
        t0 = time.time()
        for _ in range(TIMING_ITERS):
            try:
                model(dummy_img, entropy_map=dummy_ent)
            except Exception:
                model(dummy_img)
        torch.cuda.synchronize()
        t1 = time.time()

    return (t1 - t0) / TIMING_ITERS * 1000  # ms


def evaluate_model(model_info):
    """评估单个模型，返回结果字典。"""
    print(f"\n{'='*60}")
    print(f"  评估: {model_info['name']}")
    print(f"{'='*60}")

    # ---- 加载配置 ----
    cfg = SLConfig.fromfile(model_info["config"])
    cfg.pretrained = False
    cfg.eval_spatial_size = getattr(cfg, 'eval_spatial_size', [640, 640])

    # ---- 构建模型 ----
    model_name = getattr(cfg, 'modelname', 'LINEA_ENTROPY')
    cfg.modelname = model_name
    build_fn = MODULE_BUILD_FUNCS.get(model_name)
    model, postprocessors = build_fn(cfg)

    total_params, trainable_params = count_parameters(model)
    print(f"  参数量: {total_params:,} (可训练: {trainable_params:,})")

    # ---- 加载权重 ----
    weights_path = model_info["weights"]
    if not os.path.isfile(weights_path):
        weights_path = FALLBACK_WEIGHTS.get(model_info["name"], weights_path)
    if not os.path.isfile(weights_path):
        print(f"  [错误] 权重文件不存在: {weights_path}")
        return None

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    print(f"  权重已加载: {weights_path}")
    if "epoch" in ckpt:
        print(f"  训练轮次: {ckpt['epoch']}")

    model = model.to(DEVICE)
    model.eval()

    # ---- 测速 ----
    entropy_mode = getattr(cfg, 'entropy_mode', 'entropy')
    entropy_ch = 1 if entropy_mode == 'entropy' else 3
    img_size = cfg.eval_spatial_size[0] if isinstance(cfg.eval_spatial_size, (list, tuple)) else cfg.eval_spatial_size
    infer_ms = measure_speed(model, img_size, entropy_ch, DEVICE)
    fps = 1000.0 / infer_ms
    print(f"  推理速度: {infer_ms:.1f} ms/帧 ({fps:.1f} FPS) @ {img_size}x{img_size}")

    # ---- 构建测试集 ----
    dataset_val = build_dataset(image_set='test', args=cfg)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val, 64, sampler=sampler_val, drop_last=False,
        collate_fn=BatchImageCollateFunction(), num_workers=4
    )
    print(f"  测试集样本数: {len(dataset_val)}")

    # ---- 构建损失函数 (test 函数需要) ----
    from models.registry import MODULE_BUILD_FUNCS as MBF
    criterion_name = getattr(cfg, 'criterionname', 'LINEACRITERION')
    criterion_build = MBF.get(criterion_name)
    if criterion_build is not None:
        criterion, _ = criterion_build(cfg)
        criterion = criterion.to(DEVICE)
    else:
        criterion = None

    # ---- 评估 sAP ----
    evaluator = LineEvaluator()
    test(model, criterion, postprocessors, evaluator,
         data_loader_val, DEVICE, cfg.output_dir, args=cfg)

    sap_results = evaluator.sap_results

    result = {
        "name": model_info["name"],
        "backbone": getattr(cfg, 'backbone', '?'),
        "hidden_dim": getattr(cfg, 'hidden_dim', '?'),
        "dec_layers": getattr(cfg, 'dec_layers', '?'),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "infer_ms": infer_ms,
        "fps": fps,
        "img_size": img_size,
        "entropy_mode": entropy_mode,
        "sAP5": sap_results.get('sap5', 0),
        "sAP10": sap_results.get('sap10', 0),
        "sAP15": sap_results.get('sap15', 0),
    }
    return result


def format_results(results):
    """格式化为对比表。"""
    lines = []
    lines.append("=" * 72)
    lines.append("  实验 1: LINEA-L vs LINEA-N 精度对比 (MU-SID 测试集)")
    lines.append("=" * 72)
    lines.append("")

    # 表头
    header = f"{'指标':<24} {'LINEA-L':>20} {'LINEA-N':>20}"
    lines.append(header)
    lines.append("-" * 64)

    r_l = results[0] if results[0] else {}
    r_n = results[1] if results[1] else {}

    rows = [
        ("Backbone", r_l.get("backbone", "-"), r_n.get("backbone", "-")),
        ("Hidden Dim", str(r_l.get("hidden_dim", "-")), str(r_n.get("hidden_dim", "-"))),
        ("Decoder Layers", str(r_l.get("dec_layers", "-")), str(r_n.get("dec_layers", "-"))),
        ("Entropy Mode", r_l.get("entropy_mode", "-"), r_n.get("entropy_mode", "-")),
        ("参数量", f"{r_l.get('total_params', 0):,}", f"{r_n.get('total_params', 0):,}"),
        ("推理速度 (ms)", f"{r_l.get('infer_ms', 0):.1f}", f"{r_n.get('infer_ms', 0):.1f}"),
        ("FPS", f"{r_l.get('fps', 0):.1f}", f"{r_n.get('fps', 0):.1f}"),
        ("", "", ""),
        ("sAP@5", f"{r_l.get('sAP5', 0):.1f}", f"{r_n.get('sAP5', 0):.1f}"),
        ("sAP@10", f"{r_l.get('sAP10', 0):.1f}", f"{r_n.get('sAP10', 0):.1f}"),
        ("sAP@15", f"{r_l.get('sAP15', 0):.1f}", f"{r_n.get('sAP15', 0):.1f}"),
    ]

    for label, v1, v2 in rows:
        lines.append(f"{label:<24} {v1:>20} {v2:>20}")

    lines.append("")

    # 压缩率
    if r_l.get("total_params") and r_n.get("total_params"):
        ratio = r_l["total_params"] / r_n["total_params"]
        lines.append(f"参数压缩比: {ratio:.1f}x ({r_l['total_params']:,} → {r_n['total_params']:,})")
    if r_l.get("infer_ms") and r_n.get("infer_ms"):
        speedup = r_l["infer_ms"] / r_n["infer_ms"]
        lines.append(f"推理加速比: {speedup:.1f}x ({r_l['infer_ms']:.1f}ms → {r_n['infer_ms']:.1f}ms)")
    if r_l.get("sAP10") and r_n.get("sAP10"):
        delta = r_n["sAP10"] - r_l["sAP10"]
        lines.append(f"sAP@10 变化: {delta:+.1f} ({r_l['sAP10']:.1f} → {r_n['sAP10']:.1f})")

    lines.append("")
    lines.append("注: 推理速度在当前设备上测量，Jetson 上会有所不同。")
    lines.append("    sAP@N 表示在 N 像素端点距离阈值下的结构平均精度 (%)。")

    return "\n".join(lines)


def main():
    results = []
    for m in MODELS:
        r = evaluate_model(m)
        results.append(r)

    report = format_results(results)
    print("\n" + report)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")
        f.write("\n\n--- Raw JSON ---\n")
        for r in results:
            if r:
                f.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    print(f"\n结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
