"""
analyze_ranking_quality.py — 候选线排序质量分析

对 4 组关键模型进行推理，统计：
  1. GT 最优候选在得分排序中的平均名次（Rank@GT）
  2. Top-1 / Top-5 / Top-10 命中率
  3. GT 最优候选与最强伪候选的 score gap
  4. 上述指标的中位数和 P95

用法：在项目根目录运行
  python -m stage1_linea_entropy.analyze_ranking_quality
"""

import sys, os, json, math, time
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
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = "cuda"

# 分析的实验组
EXPERIMENTS = [
    {
        "tag": "Baseline",
        "mode": "baseline",
        "config": "stage1_linea_entropy/configs/linea_baseline_musid.py",
        "weights": "output/linea_baseline_musid_e100/best_checkpoint.pth",
    },
    {
        "tag": "w/o HASH(std)",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/ablation_mslep_sai_egab_musid.py",
        "weights": "output/linea_entropy_b_enhanced_musid_v1_e150/best_checkpoint.pth",
    },
    {
        "tag": "+HASH(std)",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py",
        "weights": "output/linea_entropy_b_enhanced_musid_v1_e150/best_checkpoint.pth",
    },
    {
        "tag": "Full(+Rank)",
        "mode": "entropy_b_enhanced",
        "config": "stage1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py",
        "weights": "output/linea_entropy_b_enhanced_musid_v2_e130/best_checkpoint.pth",
    },
]

OUTPUT_DIR = "stage1_linea_entropy/ranking_analysis"


# ============================================================
# 工具函数
# ============================================================
def compute_epe_to_gt(candidate_line, gt_line):
    """计算候选线与 GT 的 mean endpoint error（像素）。"""
    p1, p2 = candidate_line[:2], candidate_line[2:]
    g1, g2 = gt_line[:2], gt_line[2:]
    err_a = (np.linalg.norm(p1 - g1) + np.linalg.norm(p2 - g2)) / 2.0
    err_b = (np.linalg.norm(p1 - g2) + np.linalg.norm(p2 - g1)) / 2.0
    return min(err_a, err_b)


def analyze_ranking_for_image(all_lines, all_scores, gt_line):
    """
    分析单张图的候选排序质量。

    Args:
        all_lines  : ndarray [N, 4] — 全部候选线 (x1,y1,x2,y2)
        all_scores : ndarray [N] — 对应得分 (sigmoid 后)
        gt_line    : ndarray [4] — GT 海天线

    Returns:
        dict 包含排序指标
    """
    N = len(all_lines)
    if N == 0:
        return None

    # 计算每条候选线与 GT 的 EPE
    epes = np.array([compute_epe_to_gt(all_lines[i], gt_line) for i in range(N)])

    # GT 最优候选：EPE 最小的那条
    gt_best_idx = np.argmin(epes)
    gt_best_epe = epes[gt_best_idx]
    gt_best_score = all_scores[gt_best_idx]

    # 按得分降序排列的名次（1-indexed）
    score_rank_order = np.argsort(all_scores)[::-1]
    gt_rank = int(np.where(score_rank_order == gt_best_idx)[0][0]) + 1

    # Top-K 命中：GT 最优候选是否在得分前 K 名中
    top1_hit = gt_rank <= 1
    top5_hit = gt_rank <= 5
    top10_hit = gt_rank <= 10

    # Score gap：GT 最优候选的得分 vs 得分最高的伪候选
    # 选出得分排名第一的线，如果它不是 GT 最优候选，则计算 gap
    rank1_idx = score_rank_order[0]
    if rank1_idx == gt_best_idx:
        # GT 最优候选已经排第一，gap 为与第二名的差
        if N > 1:
            rank2_idx = score_rank_order[1]
            score_gap = gt_best_score - all_scores[rank2_idx]
        else:
            score_gap = gt_best_score
    else:
        # GT 最优候选不是第一名
        score_gap = gt_best_score - all_scores[rank1_idx]  # 负值表示不如最强伪候选

    # 实际选中的线段（得分第一名）与 GT 的 EPE
    actual_selected_epe = epes[rank1_idx]

    return {
        'gt_best_epe': float(gt_best_epe),
        'gt_best_score': float(gt_best_score),
        'gt_rank': gt_rank,
        'top1_hit': top1_hit,
        'top5_hit': top5_hit,
        'top10_hit': top10_hit,
        'score_gap': float(score_gap),
        'actual_selected_epe': float(actual_selected_epe),
        'num_candidates': N,
    }


# ============================================================
# 主流程
# ============================================================
def run_experiment(exp_cfg, device):
    """对单个实验跑排序分析。"""
    import stage1_linea_entropy.test_linea_stage1 as T

    # 设置 MODE（build_model 依赖全局 MODE 变量）
    T.MODE = exp_cfg['mode']

    # 加载配置
    args = T.load_config(exp_cfg['config'])

    # 构建模型
    model, postprocessor = T.build_model(args)
    model.to(device)

    # 加载权重
    T.load_weights(model, exp_cfg['weights'], device)
    model.eval()

    # 构建数据集
    test_dataset = T.build_test_dataset(args)
    collate_fn = T.build_test_collate_fn()
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )

    include_entropy = exp_cfg['mode'] in ("entropy", "entropy_a", "entropy_b", "entropy_b_enhanced")

    per_image_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 解包
            if include_entropy:
                if len(batch) == 4:
                    images, targets, entropy_maps, metas = batch
                else:
                    images, targets, entropy_maps = batch
                    metas = None
            else:
                if len(batch) == 3:
                    images, targets, metas = batch
                else:
                    images, targets = batch[:2]
                    metas = None
                entropy_maps = None

            images = images.to(device)
            if entropy_maps is not None:
                entropy_maps = entropy_maps.to(device)

            # 前向
            if entropy_maps is not None:
                outputs = model(images, entropy_map=entropy_maps)
            else:
                outputs = model(images)

            # PostProcess
            B = images.shape[0]
            target_sizes = torch.tensor([[IMG_SIZE, IMG_SIZE]], device=device).repeat(B, 1)
            results_batch = postprocessor(outputs, target_sizes)

            for i in range(B):
                if metas is not None:
                    stem = metas[i].get('stem', f"sample_{batch_idx * BATCH_SIZE + i}")
                else:
                    stem = f"sample_{batch_idx * BATCH_SIZE + i}"

                gt_line = T.gt_line_from_target(targets[i], IMG_SIZE)
                pred_lines = results_batch[i]['lines'].cpu().numpy()   # [N, 4]
                pred_scores = results_batch[i]['scores'].cpu().numpy()  # [N]

                # 不做 topk 截断，直接分析全部候选线的排序
                ranking_info = analyze_ranking_for_image(pred_lines, pred_scores, gt_line)
                if ranking_info is not None:
                    ranking_info['stem'] = stem
                    per_image_results.append(ranking_info)

    return per_image_results


def summarize_results(per_image_results, tag):
    """汇总单组实验的排序指标。"""
    N = len(per_image_results)
    if N == 0:
        return {}

    ranks = np.array([r['gt_rank'] for r in per_image_results])
    gaps = np.array([r['score_gap'] for r in per_image_results])
    gt_epes = np.array([r['gt_best_epe'] for r in per_image_results])
    actual_epes = np.array([r['actual_selected_epe'] for r in per_image_results])
    top1_hits = np.array([r['top1_hit'] for r in per_image_results])
    top5_hits = np.array([r['top5_hit'] for r in per_image_results])
    top10_hits = np.array([r['top10_hit'] for r in per_image_results])

    summary = {
        'tag': tag,
        'num_samples': N,
        'rank_mean': float(np.mean(ranks)),
        'rank_median': float(np.median(ranks)),
        'rank_p95': float(np.percentile(ranks, 95)),
        'rank_max': int(np.max(ranks)),
        'top1_rate': float(np.mean(top1_hits) * 100),
        'top5_rate': float(np.mean(top5_hits) * 100),
        'top10_rate': float(np.mean(top10_hits) * 100),
        'score_gap_mean': float(np.mean(gaps)),
        'score_gap_median': float(np.median(gaps)),
        'gt_best_epe_mean': float(np.mean(gt_epes)),
        'actual_selected_epe_mean': float(np.mean(actual_epes)),
        'epe_loss_ratio': float(np.mean(actual_epes) / max(np.mean(gt_epes), 1e-8)),
    }
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    all_summaries = []
    all_per_image = []   # 保存每组的逐样本结果，用于绘图

    for exp in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"  排序分析: {exp['tag']}")
        print(f"  Config:  {exp['config']}")
        print(f"  Weights: {exp['weights']}")
        print("=" * 70)

        per_image = run_experiment(exp, device)

        summary = summarize_results(per_image, exp['tag'])
        all_summaries.append(summary)
        all_per_image.append(per_image)

        # 保存逐样本结果
        tag_safe = exp['tag'].replace(' ', '_').replace('+', '_').replace('（', '_').replace('）', '_')
        out_path = os.path.join(OUTPUT_DIR, f"ranking_{tag_safe}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(per_image, f, indent=2, ensure_ascii=False)
        print(f"  逐样本结果已保存: {out_path}")

        # 打印摘要
        print(f"\n  --- 排序质量摘要 ---")
        print(f"  GT最优候选平均名次:   {summary['rank_mean']:.1f} (中位: {summary['rank_median']:.0f}, P95: {summary['rank_p95']:.0f}, Max: {summary['rank_max']})")
        print(f"  Top-1 命中率:          {summary['top1_rate']:.1f}%")
        print(f"  Top-5 命中率:          {summary['top5_rate']:.1f}%")
        print(f"  Top-10 命中率:         {summary['top10_rate']:.1f}%")
        print(f"  Score gap 均值:        {summary['score_gap_mean']:.4f}")
        print(f"  GT最优候选 EPE均值:    {summary['gt_best_epe_mean']:.2f} px")
        print(f"  实际选中 EPE均值:      {summary['actual_selected_epe_mean']:.2f} px")
        print(f"  EPE损失比:             {summary['epe_loss_ratio']:.2f}x")

    # 保存汇总
    summary_path = os.path.join(OUTPUT_DIR, "ranking_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] 汇总已保存: {summary_path}")

    # 打印对比表
    print("\n" + "=" * 90)
    print(f"{'实验':<25} {'Rank@GT':>8} {'Top-1%':>8} {'Top-5%':>8} {'Top-10%':>8} {'Gap':>8} {'EPE选中':>8}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s['tag']:<25} {s['rank_mean']:>8.1f} {s['top1_rate']:>7.1f}% {s['top5_rate']:>7.1f}% {s['top10_rate']:>7.1f}% {s['score_gap_mean']:>8.4f} {s['actual_selected_epe_mean']:>7.2f}")
    print("=" * 90)

    # ============================================================
    # 绘 图
    # ============================================================
    print("\n绘制排序质量可视化图...")
    plot_ranking_figures(all_summaries, all_per_image, OUTPUT_DIR)
    print("[DONE] 所有图片已保存")


# ============================================================
# 绘图函数
# ============================================================
def plot_ranking_figures(summaries, all_per_image, output_dir):
    """生成排序质量相关的全部图片。"""
    tags = [s['tag'] for s in summaries]
    short_tags = ['Baseline', '+EGAB', '+HASH', 'Full']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    # --- 图 1: Top-K 命中率分组柱状图 ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(short_tags))
    w = 0.22
    top1 = [s['top1_rate'] for s in summaries]
    top5 = [s['top5_rate'] for s in summaries]
    top10 = [s['top10_rate'] for s in summaries]
    bars1 = ax.bar(x - w, top1, w, label='Top-1', color='#d62728', edgecolor='white')
    bars2 = ax.bar(x, top5, w, label='Top-5', color='#ff7f0e', edgecolor='white')
    bars3 = ax.bar(x + w, top10, w, label='Top-10', color='#2ca02c', edgecolor='white')
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title('Candidate Ranking: Top-K Hit Rate', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ranking_topk_hitrate.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: ranking_topk_hitrate.png")

    # --- 图 2: Rank@GT 均值 + P95 柱状图 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    rank_mean = [s['rank_mean'] for s in summaries]
    rank_p95 = [s['rank_p95'] for s in summaries]
    w = 0.3
    bars1 = ax.bar(x - w/2, rank_mean, w, label='Mean Rank', color=colors, edgecolor='white')
    bars2 = ax.bar(x + w/2, rank_p95, w, label='P95 Rank', color=colors, edgecolor='white', alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.0f}',
                ha='center', va='bottom', fontsize=9, alpha=0.7)
    ax.set_ylabel('Rank (lower is better)', fontsize=12)
    ax.set_title('GT-Best Candidate Rank in Score Ordering', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ranking_rank_at_gt.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: ranking_rank_at_gt.png")

    # --- 图 3: EPE 对比（GT最优 vs 实际选中）---
    fig, ax = plt.subplots(figsize=(8, 5))
    gt_epe = [s['gt_best_epe_mean'] for s in summaries]
    actual_epe = [s['actual_selected_epe_mean'] for s in summaries]
    w = 0.3
    bars1 = ax.bar(x - w/2, gt_epe, w, label='GT-Best EPE (oracle)', color='#2ca02c', edgecolor='white')
    bars2 = ax.bar(x + w/2, actual_epe, w, label='Selected EPE (actual)', color='#d62728', edgecolor='white')
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean EPE (px)', fontsize=12)
    ax.set_title('Endpoint Error: Oracle vs Actual Selection', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ranking_epe_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: ranking_epe_comparison.png")

    # --- 图 4: GT Rank 分布箱线图 ---
    if all_per_image:
        fig, ax = plt.subplots(figsize=(8, 5))
        rank_data = []
        for per_image in all_per_image:
            ranks = [r['gt_rank'] for r in per_image]
            rank_data.append(ranks)
        bp = ax.boxplot(rank_data, labels=short_tags, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('GT-Best Candidate Rank', fontsize=12)
        ax.set_title('Distribution of GT-Best Rank Across Test Images', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'ranking_rank_distribution_boxplot.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  已保存: ranking_rank_distribution_boxplot.png")

    # --- 图 5: Score Gap 柱状图 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    gaps = [s['score_gap_mean'] for s in summaries]
    bars = ax.bar(x, gaps, 0.5, color=colors, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.002 if h >= 0 else -0.005),
                f'{h:.4f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Score Gap (positive = GT ranked higher)', fontsize=11)
    ax.set_title('Score Gap: GT-Best vs Top-Ranked Candidate', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ranking_score_gap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: ranking_score_gap.png")

    # --- 图 6: EPE 损失比柱状图 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    loss_ratios = [s['epe_loss_ratio'] for s in summaries]
    bars = ax.bar(x, loss_ratios, 0.5, color=colors, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f'{h:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Oracle (1.0x)')
    ax.set_ylabel('EPE Loss Ratio (actual / oracle)', fontsize=11)
    ax.set_title('Selection Quality: EPE Loss Ratio', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_tags, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ranking_epe_loss_ratio.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: ranking_epe_loss_ratio.png")


if __name__ == "__main__":
    main()
