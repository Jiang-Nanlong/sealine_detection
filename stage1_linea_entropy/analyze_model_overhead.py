"""
analyze_model_overhead.py — 模型开销分析

统计 4 组关键配置的参数量、FLOPs、推理速度（FPS）：
  ① 基线 LINEA-L
  ③ +MSLEP
  ④ +MSLEP+SAI (共享权重，结构差异仅在注入方式)
  ⑤ +MSLEP+SAI+EGAB
  ⑦ 全量 (+HASH+排序损失) — 排序损失为训练期开销，推理零增量

用法：
  python -m stage1_linea_entropy.analyze_model_overhead
"""

import sys, os, time, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
IMG_SIZE = 640
DEVICE = "cuda"
WARMUP_ITERS = 50
BENCH_ITERS = 200

EXPERIMENTS = [
    {
        "tag": "Baseline",
        "mode": "baseline",
        "config": "stage1_linea_entropy/configs/linea_baseline_musid.py",
        "weights": "output/linea_baseline_musid_e100/best_checkpoint.pth",
    },
    {
        "tag": "+MSLEP",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/ablation_mslep_only_musid.py",
        "weights": "output/ablation_mslep_only_musid_e130/best_checkpoint.pth",
    },
    {
        "tag": "+MSLEP+SAI",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/ablation_mslep_sai_musid.py",
        "weights": "output/ablation_mslep_sai_musid_e130/best_checkpoint.pth",
    },
    {
        "tag": "+MSLEP+SAI+EGAB",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/ablation_mslep_sai_egab_musid.py",
        "weights": "output/linea_entropy_b_enhanced_musid_v2_e130/best_checkpoint.pth",
    },
    {
        "tag": "Full(+HASH+Rank)",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py",
        "weights": "output/linea_entropy_b_enhanced_musid_v2_e130/best_checkpoint.pth",
    },
]

OUTPUT_DIR = "stage1_linea_entropy/overhead_analysis"


def count_parameters(model):
    """统计总参数和可训练参数。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_module_parameters(model):
    """按顶层子模块统计参数量。"""
    module_params = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            module_params[name] = n
    return module_params


def measure_fps(model, device, include_entropy, img_size=640,
                warmup=WARMUP_ITERS, bench=BENCH_ITERS):
    """用随机输入测量纯推理 FPS（不含数据加载和后处理）。"""
    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_entropy = torch.randn(1, 3, img_size, img_size, device=device) if include_entropy else None

    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            if dummy_entropy is not None:
                model(dummy_img, entropy_map=dummy_entropy)
            else:
                model(dummy_img)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench):
            if dummy_entropy is not None:
                model(dummy_img, entropy_map=dummy_entropy)
            else:
                model(dummy_img)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = bench / elapsed
    ms_per_img = elapsed / bench * 1000
    return fps, ms_per_img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    import stage1_linea_entropy.test_linea_stage1 as T

    results = []

    for exp in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"  开销分析: {exp['tag']}")
        print("=" * 70)

        # 构建模型
        T.MODE = exp['mode']  # build_model/load_config 依赖全局 MODE
        args = T.load_config(exp['config'])
        model, postprocessor = T.build_model(args)
        model.to(device)
        T.load_weights(model, exp['weights'], device)
        model.eval()

        # 参数量
        total_params, trainable_params = count_parameters(model)
        module_params = count_module_parameters(model)

        print(f"  总参数:   {total_params:>12,} ({total_params/1e6:.2f}M)")
        print(f"  可训练:   {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
        print(f"  子模块参数:")
        for name, n in module_params.items():
            print(f"    {name:20s} {n:>12,} ({n/1e6:.2f}M)")

        # 推理速度
        include_entropy = exp['mode'] != 'baseline'
        fps, ms_per_img = measure_fps(model, device, include_entropy)
        print(f"  推理速度: {fps:.1f} FPS ({ms_per_img:.1f} ms/img)")

        # 新增参数相对基线
        info = {
            'tag': exp['tag'],
            'total_params': total_params,
            'total_params_M': round(total_params / 1e6, 2),
            'trainable_params': trainable_params,
            'module_params': module_params,
            'fps': round(fps, 1),
            'ms_per_img': round(ms_per_img, 1),
        }
        results.append(info)

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 计算相对基线的增量
    baseline_params = results[0]['total_params']
    baseline_fps = results[0]['fps']
    for r in results:
        r['delta_params'] = r['total_params'] - baseline_params
        r['delta_params_M'] = round(r['delta_params'] / 1e6, 2)
        r['delta_params_pct'] = round(r['delta_params'] / baseline_params * 100, 2) if baseline_params > 0 else 0
        r['fps_ratio'] = round(r['fps'] / baseline_fps, 3) if baseline_fps > 0 else 0

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "overhead_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] 结果已保存: {out_path}")

    # 打印对比表
    print("\n" + "=" * 100)
    print(f"{'实验':<25} {'参数(M)':>10} {'增量(M)':>10} {'增量%':>8} {'FPS':>8} {'ms/img':>8} {'速度比':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['tag']:<25} {r['total_params_M']:>10.2f} {r['delta_params_M']:>+10.2f} {r['delta_params_pct']:>+7.2f}% {r['fps']:>8.1f} {r['ms_per_img']:>8.1f} {r['fps_ratio']:>7.3f}x")
    print("=" * 100)

    # 额外说明
    print("\n说明:")
    print("  - 排序损失（Ranking Loss）仅在训练阶段计算，推理阶段零增量，因此⑤和⑦的推理开销相同")
    print("  - 熵先验图（MSLEP）在离线预计算时有额外开销，但不计入在线推理时间")
    print(f"  - 测试配置: 输入 {IMG_SIZE}×{IMG_SIZE}, Warmup {WARMUP_ITERS}, Benchmark {BENCH_ITERS} iters")

    # ============================================================
    # 绘 图
    # ============================================================
    print("\n绘制模型开销可视化图...")
    plot_overhead_figures(results, OUTPUT_DIR)
    print("[DONE] 所有图片已保存")


# ============================================================
# 绘图函数
# ============================================================
def plot_overhead_figures(results, output_dir):
    """生成模型开销相关的全部图片。"""
    tags = [r['tag'] for r in results]
    short_tags = ['Baseline', '+MSLEP', '+SAI', '+EGAB', 'Full']
    colors = ['#1f77b4', '#9467bd', '#2ca02c', '#ff7f0e', '#d62728']
    x = np.arange(len(short_tags))

    # --- 图 1: 参数量柱状图 + 增量标注 ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    params_M = [r['total_params_M'] for r in results]
    bars = ax.bar(x, params_M, 0.5, color=colors, edgecolor='white')
    for i, bar in enumerate(bars):
        h = bar.get_height()
        label = f'{h:.2f}M'
        if i > 0:
            delta = results[i]['delta_params_M']
            pct = results[i]['delta_params_pct']
            label += f'\n(+{delta:.2f}M, +{pct:.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, label,
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Parameters (M)', fontsize=12)
    ax.set_title('Model Parameters Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    # 设置y轴从合适的起点开始
    min_p = min(params_M)
    ax.set_ylim(min_p * 0.95, max(params_M) * 1.12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overhead_params.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: overhead_params.png")

    # --- 图 2: FPS 柱状图 ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fps_vals = [r['fps'] for r in results]
    bars = ax.bar(x, fps_vals, 0.5, color=colors, edgecolor='white')
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ratio = results[i]['fps_ratio']
        label = f'{h:.1f}\n({ratio:.3f}x)' if i > 0 else f'{h:.1f}\n(1.000x)'
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, label,
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('FPS (frames per second)', fontsize=12)
    ax.set_title('Inference Speed Comparison (RTX 4090D)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overhead_fps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: overhead_fps.png")

    # --- 图 3: 推理时延 ms/img ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ms_vals = [r['ms_per_img'] for r in results]
    bars = ax.bar(x, ms_vals, 0.5, color=colors, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f} ms',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Latency (ms / image)', fontsize=12)
    ax.set_title('Inference Latency per Image', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overhead_latency.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: overhead_latency.png")

    # --- 图 4: 参数增量 + FPS 双轴图 ---
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    w = 0.35
    bars1 = ax1.bar(x - w/2, params_M, w, label='Params (M)', color='#4C72B0', edgecolor='white', alpha=0.8)
    line = ax2.plot(x, fps_vals, 'o-', color='#C44E52', linewidth=2, markersize=8, label='FPS', zorder=5)
    for i, (bar, fps) in enumerate(zip(bars1, fps_vals)):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.2f}M',
                 ha='center', va='bottom', fontsize=8)
        ax2.text(x[i] + 0.02, fps + 0.5, f'{fps:.1f}', ha='left', va='bottom',
                 fontsize=9, color='#C44E52', fontweight='bold')
    ax1.set_ylabel('Parameters (M)', fontsize=12, color='#4C72B0')
    ax2.set_ylabel('FPS', fontsize=12, color='#C44E52')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_tags, fontsize=10)
    ax1.set_title('Parameters vs Inference Speed Trade-off', fontsize=13)
    # 扩大参数量纵轴范围，防止柱状图顶部被图例遮挡
    ax1.set_ylim(0, max(params_M) * 1.35)
    ax2.set_ylim(min(fps_vals) - 5, max(fps_vals) + 8)
    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overhead_params_vs_fps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: overhead_params_vs_fps.png")

    # --- 图 5: 子模块参数分布堆叠柱状图 ---
    # 收集所有子模块名
    all_modules = set()
    for r in results:
        all_modules.update(r['module_params'].keys())
    all_modules = sorted(all_modules)
    module_colors = plt.cm.Set3(np.linspace(0, 1, max(len(all_modules), 3)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(results))
    for j, mod in enumerate(all_modules):
        vals = [r['module_params'].get(mod, 0) / 1e6 for r in results]
        ax.bar(x, vals, 0.5, bottom=bottom, label=mod, color=module_colors[j], edgecolor='white')
        bottom += np.array(vals)
    ax.set_ylabel('Parameters (M)', fontsize=12)
    ax.set_title('Parameter Distribution by Module', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=10)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overhead_module_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: overhead_module_breakdown.png")


if __name__ == "__main__":
    main()
