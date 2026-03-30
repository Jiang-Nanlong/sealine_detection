import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 参数 =========
theta_star_deg = 1.0
N = 3000
period_deg = 180.0
num_periods = 3  # 画几个周期
theta_min, theta_max = 0.0, period_deg * num_periods  # 0~540°

# ========= 扫描 =========
theta_deg = np.linspace(theta_min, theta_max, N)
theta_rad = np.deg2rad(theta_deg)
theta_star_rad = np.deg2rad(theta_star_deg)

# 1) naive（不考虑周期）
delta = np.abs(theta_deg - theta_star_deg)
L_naive = delta
L_naive_norm = L_naive / (L_naive.max() + 1e-12)

# 2) wrapped 最短周期距离：先对周期取模再取最短
#    关键：Δ 先 mod 180°，再 min(Δ, 180-Δ)
delta_mod = np.mod(delta, period_deg)
L_wrap = np.minimum(delta_mod, period_deg - delta_mod)
L_wrap_norm = L_wrap / (L_wrap.max() + 1e-12)

# 3) periodic（倍角映射）
v_pred = np.stack([np.cos(2 * theta_rad), np.sin(2 * theta_rad)], axis=1)
v_gt = np.array([np.cos(2 * theta_star_rad), np.sin(2 * theta_star_rad)])
L_per = np.linalg.norm(v_pred - v_gt[None, :], axis=1)   # [0,2]
L_per_norm = L_per / 2.0

# ========= 画图（中文不乱码） =========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(8.2, 4.6))

ax.plot(theta_deg, L_naive_norm, color='red', linewidth=2.2,
        label=r'$L_{\mathrm{naive}}(\theta)=|\theta-\theta^*|$（不做周期处理）')
ax.plot(theta_deg, L_wrap_norm, color='black', linestyle='--', linewidth=2.0,
        label=r'$d_{\pi}=\min(\Delta\,\mathrm{mod}\,180^\circ,\ 180^\circ-(\Delta\,\mathrm{mod}\,180^\circ))$（最短周期距离）')
ax.plot(theta_deg, L_per_norm, color='blue', linewidth=2.2,
        label=r'$L_{\mathrm{per}}=\|v(\theta)-v(\theta^*)\|_2,\ v(\theta)=(\cos2\theta,\sin2\theta)$')

# 标注真值角（在第一个周期）
ax.axvline(x=theta_star_deg, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
ax.text(theta_star_deg + 5, 0.92, r'$\theta^*=1^\circ$', color='gray')

# 标周期分界线（180°, 360°）
for k in range(1, num_periods):
    ax.axvline(x=period_deg*k, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

ax.set_xlim(theta_min, theta_max)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r'预测角度 $\theta$（度）')
ax.set_ylabel('归一化损失')
ax.set_title(r'图5-8 小窗：普通角度差 vs 周期距离 vs 周期性损失（$\theta^*=1^\circ$）')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

fig.tight_layout()
out_path = Path("fig5_8_loss_inset_multi_period.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("saved:", out_path)