#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-8  周期性损失 vs 普通角度差损失 对比曲线
  红色：L_naive(θ) = |θ - θ*|  （未做周期处理，边界跳变）
  蓝色：L_per(θ)   = ‖v(θ) - v(θ*)‖₂  （倍角映射，周期性损失）
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 参数
# ============================================================================
theta_star_deg = 1.0          # 真值角 θ* = 1°
N = 1000                      # 扫描点数

# ============================================================================
# 扫描
# ============================================================================
theta_deg = np.linspace(0, 180, N)       # 预测角 θ ∈ [0°, 180°]
theta_rad = np.deg2rad(theta_deg)
theta_star_rad = np.deg2rad(theta_star_deg)

# 曲线1：普通角度差损失（L1 形式，不做 wrap）
L_naive = np.abs(theta_deg - theta_star_deg)       # 单位：度

# 曲线2：倍角映射周期性损失  v(θ) = (cos2θ, sin2θ)
#   L_per = ‖v(θ) - v(θ*)‖₂
v_pred = np.stack([np.cos(2 * theta_rad), np.sin(2 * theta_rad)], axis=1)   # (N, 2)
v_gt   = np.array([np.cos(2 * theta_star_rad), np.sin(2 * theta_star_rad)]) # (2,)
L_per  = np.linalg.norm(v_pred - v_gt[None, :], axis=1)                     # (N,)

# 为了在同一幅图中对比形状，将 L_naive 归一化到 [0, 1]，L_per 最大值为 2
L_naive_norm = L_naive / L_naive.max()    # 归一化到 [0, 1]
L_per_norm   = L_per / 2.0               # ‖v-v*‖ 最大 = 2，归一化到 [0, 1]

# ============================================================================
# 画图
# ============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

ax.plot(theta_deg, L_naive_norm, color='red',  linewidth=2.0,
        label=r'$L_{\mathrm{naive}}(\theta) = |\theta - \theta^*|}$  (no wrap)')
ax.plot(theta_deg, L_per_norm,   color='blue', linewidth=2.0,
        label=r'$L_{\mathrm{per}}(\theta) = \|v(\theta)-v(\theta^*)\|_2$  (double-angle)')

# 标注真值角位置
ax.axvline(x=theta_star_deg, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(theta_star_deg + 2, 0.92, r'$\theta^*=' + f'{theta_star_deg:.0f}°$',
        fontsize=10, color='gray')

# 标注边界跳变区域
ax.annotate('boundary\njump',
            xy=(175, L_naive_norm[np.argmin(np.abs(theta_deg - 175))]),
            xytext=(140, 0.75),
            fontsize=9, color='red', alpha=0.8,
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

ax.set_xlabel(r'Predicted angle $\theta$ (degrees)', fontsize=11)
ax.set_ylabel('Normalized loss', fontsize=11)
ax.set_title(r'Fig. 5-8  Naive loss vs periodic loss ($\theta^*=1°$)', fontsize=12, pad=8)
ax.set_xlim(0, 180)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='center right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = output_dir / "fig5_8_periodic_loss_comparison.png"
fig.savefig(str(out_path), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f"✓ 已保存: {out_path}")
