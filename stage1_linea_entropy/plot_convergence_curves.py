"""
plot_convergence_curves.py — 训练收敛曲线

从 output/*/log.txt 解析训练日志，绘制：
  1. 训练 loss（test_loss）收敛曲线
  2. 对比基线 vs 全量模型的收敛速度

用法：
  python -m stage1_linea_entropy.plot_convergence_curves
"""

import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
LOG_CONFIGS = [
    {
        "tag": "Baseline LINEA-L (100 ep)",
        "log_path": "output/linea_baseline_musid_e100/log.txt",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "tag": "+MSLEP+SAI (130 ep)",
        "log_path": "output/ablation_mslep_sai_musid_e130/log.txt",
        "color": "#2ca02c",
        "linestyle": "--",
    },
    {
        "tag": "+MSLEP+SAI+EGAB (130 ep)",
        "log_path": "output/ablation_mslep_sai_egab_musid_standalone_e130/log.txt",
        "color": "#9467bd",
        "linestyle": "--",
    },
    {
        "tag": "+HASH, std loss (150 ep)",
        "log_path": "output/linea_entropy_b_enhanced_musid_v1_e150/log.txt",
        "color": "#ff7f0e",
        "linestyle": "-.",
    },
    {
        "tag": "Full, +ranking loss (130 ep)",
        "log_path": "output/linea_entropy_b_enhanced_musid_v2_e130/log.txt",
        "color": "#d62728",
        "linestyle": "-",
    },
]

OUTPUT_DIR = "stage1_linea_entropy/convergence_analysis"


def parse_log(log_path):
    """解析 log.txt，每行一个 JSON，返回按 epoch 去重后的列表。"""
    records = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                epoch = rec['epoch']
                records[epoch] = rec  # 用最后一次出现的记录（处理重启重复）
            except (json.JSONDecodeError, KeyError):
                continue
    # 按 epoch 排序
    sorted_epochs = sorted(records.keys())
    return [records[e] for e in sorted_epochs]


def smooth(values, window=5):
    """简单移动平均平滑。"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


def plot_test_loss_curves(all_data, output_dir):
    """绘制 test_loss 收敛曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cfg, records in all_data:
        epochs = [r['epoch'] for r in records]
        test_losses = [r['test_loss'] for r in records]

        ax.plot(epochs, test_losses, label=cfg['tag'],
                color=cfg['color'], linestyle=cfg['linestyle'],
                linewidth=1.5, alpha=0.4)
        # 叠加平滑线
        smoothed = smooth(np.array(test_losses), window=7)
        ax.plot(epochs, smoothed, color=cfg['color'],
                linestyle=cfg['linestyle'], linewidth=2.0)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Test Loss', fontsize=13)
    ax.set_title('Training Convergence: Test Loss', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 135)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "convergence_test_loss.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def plot_train_line_loss_curves(all_data, output_dir):
    """绘制 train_loss_line 收敛曲线（核心回归损失）。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cfg, records in all_data:
        epochs = [r['epoch'] for r in records]
        line_losses = [r.get('train_loss_line', r.get('train_loss', 0)) for r in records]

        ax.plot(epochs, line_losses, label=cfg['tag'],
                color=cfg['color'], linestyle=cfg['linestyle'],
                linewidth=1.5, alpha=0.4)
        smoothed = smooth(np.array(line_losses), window=7)
        ax.plot(epochs, smoothed, color=cfg['color'],
                linestyle=cfg['linestyle'], linewidth=2.0)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Train Line Loss', fontsize=13)
    ax.set_title('Training Convergence: Line Regression Loss', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 135)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "convergence_train_line_loss.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path}")
    return out_path


def analyze_convergence_point(records, threshold_ratio=0.05):
    """
    估计收敛拐点：test_loss 下降到最终值附近（在最低值的 threshold_ratio 范围内）
    的最早 epoch。
    """
    test_losses = np.array([r['test_loss'] for r in records])
    epochs = np.array([r['epoch'] for r in records])

    # 用后 20% 数据的均值作为"最终水平"
    tail = max(1, len(test_losses) // 5)
    final_level = np.mean(test_losses[-tail:])
    min_loss = np.min(test_losses)

    # 收敛阈值：在 final_level 的 threshold_ratio 以内
    threshold = final_level * (1 + threshold_ratio)

    # 找第一个连续 5 epoch 都低于阈值的起始点
    window = 5
    for i in range(len(test_losses) - window):
        if np.all(test_losses[i:i+window] < threshold):
            return int(epochs[i]), float(final_level), float(min_loss)

    return int(epochs[-1]), float(final_level), float(min_loss)


def plot_test_loss_zoomed(all_data, output_dir):
    """绘制 test_loss 后半段放大图（epoch 40+），展示稳态区域的差异。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cfg, records in all_data:
        epochs = [r['epoch'] for r in records if r['epoch'] >= 40]
        test_losses = [r['test_loss'] for r in records if r['epoch'] >= 40]
        if not epochs:
            continue

        ax.plot(epochs, test_losses, label=cfg['tag'],
                color=cfg['color'], linestyle=cfg['linestyle'],
                linewidth=1.2, alpha=0.4)
        smoothed = smooth(np.array(test_losses), window=7)
        ax.plot(epochs, smoothed, color=cfg['color'],
                linestyle=cfg['linestyle'], linewidth=2.0)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Test Loss', fontsize=13)
    ax.set_title('Training Convergence: Test Loss (Epoch 40+, Zoomed)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "convergence_test_loss_zoomed.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path}")


def plot_train_total_loss_curves(all_data, output_dir):
    """绘制 train_loss（总损失）收敛曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cfg, records in all_data:
        epochs = [r['epoch'] for r in records]
        train_losses = [r.get('train_loss', 0) for r in records]

        ax.plot(epochs, train_losses, label=cfg['tag'],
                color=cfg['color'], linestyle=cfg['linestyle'],
                linewidth=1.5, alpha=0.4)
        smoothed = smooth(np.array(train_losses), window=7)
        ax.plot(epochs, smoothed, color=cfg['color'],
                linestyle=cfg['linestyle'], linewidth=2.0)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Train Total Loss', fontsize=13)
    ax.set_title('Training Convergence: Total Training Loss', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 135)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "convergence_train_total_loss.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path}")


def plot_convergence_bar(convergence_info, output_dir):
    """绘制收敛拐点 + 最终 loss 柱状图。"""
    tags_short = [info['tag'].split('(')[0].strip() for info in convergence_info]
    colors = ['#1f77b4', '#2ca02c', '#d62728'][:len(convergence_info)]
    x = np.arange(len(convergence_info))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：收敛拐点 epoch
    conv_epochs = [info['convergence_epoch'] for info in convergence_info]
    bars1 = ax1.bar(x, conv_epochs, 0.5, color=colors, edgecolor='white')
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'Ep {h:.0f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Epoch', fontsize=12)
    ax1.set_title('Convergence Point (Epoch)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tags_short, fontsize=9, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 右：最终 test_loss
    final_losses = [info['final_test_loss'] for info in convergence_info]
    min_losses = [info['min_test_loss'] for info in convergence_info]
    w = 0.3
    bars2 = ax2.bar(x - w/2, final_losses, w, label='Final (avg)', color=colors, edgecolor='white')
    bars3 = ax2.bar(x + w/2, min_losses, w, label='Min', color=colors, edgecolor='white', alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=8, alpha=0.7)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Final / Min Test Loss', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tags_short, fontsize=9, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "convergence_summary_bar.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 去重：⑤和⑦用同一个 log（因为它们用同一次训练的权重）
    # 为了绘图清晰，将⑤和⑦合并为一条线
    unique_configs = []
    seen_paths = set()
    for cfg in LOG_CONFIGS:
        if cfg['log_path'] not in seen_paths:
            seen_paths.add(cfg['log_path'])
            unique_configs.append(cfg)
        else:
            # 跳过重复 log path（⑤和⑦共享）
            pass

    all_data = []
    convergence_info = []

    for cfg in unique_configs:
        log_path = cfg['log_path']
        if not os.path.exists(log_path):
            print(f"  [SKIP] 日志不存在: {log_path}")
            continue

        records = parse_log(log_path)
        print(f"  {cfg['tag']}: {len(records)} epochs loaded (epoch {records[0]['epoch']}-{records[-1]['epoch']})")

        all_data.append((cfg, records))

        # 收敛点分析
        conv_epoch, final_level, min_loss = analyze_convergence_point(records)
        info = {
            'tag': cfg['tag'],
            'total_epochs': len(records),
            'convergence_epoch': conv_epoch,
            'final_test_loss': round(final_level, 4),
            'min_test_loss': round(min_loss, 4),
        }
        convergence_info.append(info)
        print(f"    收敛拐点 ~epoch {conv_epoch}, 最终 test_loss={final_level:.4f}, min={min_loss:.4f}")

    if not all_data:
        print("[ERROR] 无有效日志数据")
        return

    # 绘图
    print("\n绘制收敛曲线...")
    plot_test_loss_curves(all_data, OUTPUT_DIR)
    plot_train_line_loss_curves(all_data, OUTPUT_DIR)
    plot_test_loss_zoomed(all_data, OUTPUT_DIR)
    plot_train_total_loss_curves(all_data, OUTPUT_DIR)
    plot_convergence_bar(convergence_info, OUTPUT_DIR)

    # 保存收敛分析
    info_path = os.path.join(OUTPUT_DIR, "convergence_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(convergence_info, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] 收敛分析已保存: {info_path}")

    # 打印摘要表
    print("\n" + "=" * 80)
    print(f"{'实验':<30} {'总epochs':>8} {'收敛epoch':>10} {'最终loss':>10} {'最低loss':>10}")
    print("-" * 80)
    for info in convergence_info:
        print(f"{info['tag']:<30} {info['total_epochs']:>8} {info['convergence_epoch']:>10} {info['final_test_loss']:>10.4f} {info['min_test_loss']:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
