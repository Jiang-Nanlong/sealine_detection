"""生成图5-13: 精度-速度散点图
横轴 log(FPS), 纵轴精度指标, 标注各方法数据点
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_PNG = 'fig5_13_pareto.png'
OUT_PDF = 'fig5_13_pareto.pdf'

# ── 数据 (Jetson Xavier NX, MODE 20W 6CORE) ──
methods = {
    'Full Pipeline':           {'fps': 0.136, 'acc': 36.69, 'color': '#E74C3C', 'marker': 's'},
    'ESRLE-L':              {'fps': 2.3,   'acc': 40.0,  'color': '#8E44AD', 'marker': 'D'},
    'ESRLE-N':              {'fps': 5.0,   'acc': 39.5,  'color': '#3498DB', 'marker': 'D'},
    'seg-only\n(PyTorch FP16)': {'fps': 24.8,  'acc': 40.46, 'color': '#F39C12', 'marker': 'o'},
    'seg-only\n(TRT FP16)':     {'fps': 62.5,  'acc': 40.46, 'color': '#2ECC71', 'marker': 'o'},
}
# 注: LINEA的精度指标用sAP@10, UNet用VE(px), 体系不同
# 为了统一展示, 使用"综合精度分"概念: 越高越好
# sAP@10本身越高越好; VE越低越好
# 转换: score = 100 - VE (使其越高越好), sAP直接用
methods_unified = {
    'Full Pipeline':       {'fps': 0.136, 'score': 63.3, 'color': '#E74C3C', 'marker': 's', 'metric': 'VE=36.7px'},
    'ESRLE-L':              {'fps': 2.3,   'score': 94.4, 'color': '#8E44AD', 'marker': 'D', 'metric': 'sAP@10=94.4'},
    'ESRLE-N':              {'fps': 5.0,   'score': 91.2, 'color': '#3498DB', 'marker': 'D', 'metric': 'sAP@10=91.2'},
    'seg-only\n(PyTorch FP16)': {'fps': 24.8,  'score': 59.5, 'color': '#F39C12', 'marker': 'o', 'metric': 'VE=40.5px'},
    'seg-only\n(TRT FP16)':     {'fps': 62.5,  'score': 59.5, 'color': '#2ECC71', 'marker': 'o', 'metric': 'VE=40.5px'},
}

fig, ax = plt.subplots(figsize=(9, 5.5))

for name, d in methods_unified.items():
    ax.scatter(d['fps'], d['score'], c=d['color'], marker=d['marker'],
               s=180, edgecolors='black', linewidths=0.8, zorder=5)
    # 标注
    offset_x, offset_y = 1.15, 1.5
    ha = 'left'
    if 'TRT' in name:
        offset_x, offset_y = 1.15, -3
    elif 'Pipeline' in name:
        offset_x, offset_y = 1.3, 0
    elif 'ESRLE-L' in name:
        offset_x, offset_y = 0.6, 2.0
        ha = 'right'
    elif 'ESRLE-N' in name:
        offset_x, offset_y = 1.3, 2.5
    elif 'PyTorch' in name:
        offset_x, offset_y = 0.7, 2.5
        ha = 'right'

    ax.annotate(f'{name}\n{d["metric"]}',
                xy=(d['fps'], d['score']),
                xytext=(d['fps'] * offset_x, d['score'] + offset_y),
                fontsize=8.5, ha=ha, color=d['color'],
                fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=d['color'], lw=0.8, alpha=0.6))

ax.set_xscale('log')
ax.set_xlabel('FPS (log scale)', fontsize=13)
ax.set_ylabel('Accuracy Score', fontsize=13)
ax.set_title('Accuracy vs Speed on Jetson Xavier NX', fontsize=14)
ax.tick_params(axis='both', labelsize=11)

# 实时线 (30fps)
ax.axvline(30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(32, 55, 'Real-time\n(30 FPS)', fontsize=9, color='gray', alpha=0.7)

ax.set_xlim(0.05, 200)
ax.set_ylim(50, 100)
ax.grid(True, alpha=0.3)

# 注释说明不同精度体系
ax.text(0.02, 0.02, 'Note: ESRLE uses sAP@10 (higher=better); seg-only uses 100-VE (higher=better)',
        transform=ax.transAxes, fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT_PNG} / {OUT_PDF}')
