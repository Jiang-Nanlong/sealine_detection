"""
plot_multistage_loss.py — 图4.5 多阶段训练策略的损失变化趋势

从 archive/experiment_outputs/train_log_stage_*.csv 读取各阶段训练日志，
用分段着色的连续曲线展示 train_loss 随训练阶段的变化趋势，
背景色块区分各阶段。

用法（在项目根目录运行）：
  py method2_unet_radon/plot_multistage_loss.py
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
LOG_DIR = "archive/experiment_outputs"

# 阶段顺序 (key, short_label, desc, bg_color, line_color, loss_type)
# C1 虽名为"联合训练"，但 loss = crit_rest + bridge_aux，等效于 L_rest
STAGES = [
    ("a",  "阶段A",  "复原预训练",         "#cce5ff", "#2196F3", "rest"),
    ("b",  "阶段B",  "分割对齐",           "#fff2cc", "#FF9800", "seg"),
    ("c1", "阶段C1", "联合训练①",          "#d5f5d5", "#2196F3", "rest"),
    ("b2", "阶段B2", "分割修复",           "#ffe6cc", "#FF9800", "seg"),
    ("c2", "阶段C2", "联合训练②",          "#fce4ec", "#B71C1C", "total"),
]

OUTPUT_DIR = "method2_unet_radon"

# ============================================================
# 读取日志
# ============================================================
def read_stage_log(stage_key):
    path = os.path.join(LOG_DIR, f"train_log_stage_{stage_key}.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'epoch': int(r['epoch']),
                'train_loss': float(r['train_loss']),
                'val_joint': float(r['val_joint']),
            })
    return rows


def main():
    # 读取所有阶段
    all_stages = []
    for key, short, desc, bg, lc, lt in STAGES:
        rows = read_stage_log(key)
        if rows:
            print(f"  Stage {key}: {len(rows)} epochs")
            all_stages.append((key, short, desc, bg, lc, lt, rows))

    if not all_stages:
        print("[ERROR] 无有效日志")
        return

    # 构建每个阶段的 x 范围（连续 epoch）
    stage_info = []  # (x_start, x_end, key, short, desc, bg, lc, lt, xs, train_ys, val_ys)
    x_offset = 0
    all_val_x, all_val_y = [], []

    for key, short, desc, bg, lc, lt, rows in all_stages:
        xs = []
        train_ys = []
        val_ys = []
        x_start = x_offset
        for r in rows:
            xs.append(x_offset)
            train_ys.append(r['train_loss'])
            val_ys.append(r['val_joint'])
            all_val_x.append(x_offset)
            all_val_y.append(r['val_joint'])
            x_offset += 1
        x_end = x_offset
        stage_info.append((x_start, x_end, key, short, desc, bg, lc, lt,
                           np.array(xs), np.array(train_ys), np.array(val_ys)))

    total_epochs = x_offset

    # ============================================================
    # 绘图
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.0))

    # 1) 背景色块
    for x_start, x_end, key, short, desc, bg, lc, lt, xs, ty, vy in stage_info:
        ax.axvspan(x_start - 0.5, x_end - 0.5, alpha=0.22, color=bg, zorder=0)

    # 2) 分段着色的 train_loss 曲线
    #    按阶段各自画段，相邻阶段之间加一个连接点保证视觉连续
    prev_last = None
    for i, (x_start, x_end, key, short, desc, bg, lc, lt, xs, ty, vy) in enumerate(stage_info):
        # 连接与上一段的过渡（灰色虚线）
        if prev_last is not None:
            ax.plot([prev_last[0], xs[0]], [prev_last[1], ty[0]],
                    color='#999', linewidth=1.0, linestyle=':', alpha=0.5, zorder=2)
        # 分段着色线条
        label_map = {"rest": r'$\mathcal{L}_{rest}$',
                     "seg":  r'$\mathcal{L}_{seg}$',
                     "total": r'$\mathcal{L}_{total}$'}
        # 同类型只标一次 label
        used = set()
        lbl = None
        if lt not in used:
            lbl = label_map[lt]
            used.add(lt)
        ax.plot(xs, ty, color=lc, linewidth=2.2, alpha=0.9, zorder=3,
                label=lbl if i == 0 or lt not in [s[7] for s in stage_info[:i]] else None)
        prev_last = (xs[-1], ty[-1])

    # 3) val_joint 连续灰线
    ax.plot(all_val_x, all_val_y, color='#9E9E9E', linewidth=1.2,
            label='val_joint', alpha=0.55, zorder=2)

    # 4) 阶段分隔竖线
    for x_start, x_end, *_ in stage_info:
        if x_start > 0:
            ax.axvline(x=x_start - 0.5, color='#aaa', linestyle='--',
                       linewidth=0.7, alpha=0.6, zorder=1)

    # 5) y 范围
    all_train = np.concatenate([s[9] for s in stage_info])
    all_vals = np.array(all_val_y)
    y_max_data = max(all_train.max(), all_vals.max())
    y_top = min(y_max_data * 1.20, 2.5)
    ax.set_ylim(0, y_top)
    ax.set_xlim(-1, total_epochs + 0.5)

    # 6) 顶部阶段标签（简短，不重叠）
    for x_start, x_end, key, short, desc, bg, lc, lt, *_ in stage_info:
        mid = (x_start + x_end) / 2 - 0.5
        width = x_end - x_start
        fs = 9 if width > 12 else 7.5
        ax.text(mid, y_top * 0.98, f"{short}\n{desc}",
                ha='center', va='top', fontsize=fs, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=bg, alpha=0.75,
                          edgecolor='none'))

    # 7) 底部阶段冻结描述
    descs_bottom = {
        'a':  "冻结分割分支\n仅优化 $\\mathcal{L}_{rest}$",
        'b':  "冻结编码器+复原\n仅优化 $\\mathcal{L}_{seg}$",
        'c1': "解冻全部\n$\\mathcal{L}_{rest}$",
        'b2': "冻结复原\n$\\mathcal{L}_{seg}$",
        'c2': "全部解冻\n$\\mathcal{L}_{total}$",
    }
    for x_start, x_end, key, short, desc, bg, lc, lt, *_ in stage_info:
        mid = (x_start + x_end) / 2 - 0.5
        width = x_end - x_start
        fs = 7 if width > 12 else 6
        ax.text(mid, -0.08, descs_bottom[key], ha='center', va='top', fontsize=fs,
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor='#ccc'))

    # 8) 在各阶段标注 epoch 数
    for x_start, x_end, key, *_ in stage_info:
        n_ep = x_end - x_start
        mid = (x_start + x_end) / 2 - 0.5
        ax.text(mid, -0.02, f"{n_ep} ep", ha='center', va='top', fontsize=7,
                color='#666', transform=ax.get_xaxis_transform())

    ax.set_xlabel('训练过程（epoch）', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)

    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=10, loc='upper right', framealpha=0.9)

    ax.grid(True, alpha=0.12)
    ax.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    out_png = os.path.join(OUTPUT_DIR, "fig4_5_multistage_loss.png")
    out_pdf = os.path.join(OUTPUT_DIR, "fig4_5_multistage_loss.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[DONE] 已保存: {out_png}")
    print(f"       已保存: {out_pdf}")


if __name__ == "__main__":
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    main()
