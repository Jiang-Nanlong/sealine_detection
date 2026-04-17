"""生成图5-9: UNet三种推理模式时间柱状图"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

modes = ['PyTorch FP32', 'PyTorch FP16', 'TensorRT FP16']
times = [156, 51, 16]  # ms
fps = [6.4, 19.6, 62.5]
colors = ['#E74C3C', '#F39C12', '#2ECC71']

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(modes, times, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

# 在柱子上方标注数值
for bar, t, f in zip(bars, times, fps):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f'{t} ms\n({f:.1f} FPS)', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Inference Time per Frame (ms)', fontsize=13)
ax.set_ylim(0, 200)
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig('thesis_drafts/fig5_9_unet_speed.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/fig5_9_unet_speed.pdf', bbox_inches='tight')
print('已保存 fig5_9_unet_speed.png / .pdf')
