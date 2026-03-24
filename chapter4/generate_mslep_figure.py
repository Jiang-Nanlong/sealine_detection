"""
generate_mslep_figure.py — 生成 MSLEP 模块示意图所需的子图

生成文件:
  1. original.png       — 原始彩色图像
  2. blue_channel.png   — Blue 通道灰度图
  3. entropy_k1.png     — 局部熵图 (k=1, identity)
  4. entropy_k5.png     — 局部熵图 (AvgPool k=5)
  5. entropy_k11.png    — 局部熵图 (AvgPool k=11)
"""
import os

# ============================================================
# 全局配置 — 在 PyCharm 中直接修改这两个变量后运行
# ============================================================
INPUT_IMAGE = r"F:\code_manager\Menglong Cao\sealine_detection\Hashmani's Dataset\clear\DSC_0717_4.JPG"
OUTPUT_DIR  = r"F:\code_manager\Menglong Cao\sealine_detection\chapter4\fig_mslep"
# ============================================================

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk


def compute_local_entropy(src_u8: np.ndarray, radius: int) -> np.ndarray:
    return entropy(src_u8, disk(radius)).astype(np.float32)


def avg_pool_np(img: np.ndarray, k: int) -> np.ndarray:
    """NumPy 实现的 AvgPool (stride=1, same padding)"""
    if k <= 1:
        return img.copy()
    pad = k // 2
    padded = np.pad(img, pad, mode='reflect')
    kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    from scipy.signal import fftconvolve
    result = fftconvolve(padded, kernel, mode='valid')
    return result.astype(np.float32)


def save_heatmap(data: np.ndarray, path: str, cmap='jet'):
    """保存热力图, 无边框无坐标轴"""
    h, w = data.shape
    dpi = 150
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bgr = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'无法读取图片: {INPUT_IMAGE}')

    # 1. 原始彩色图
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'original.png'), bgr)
    print(f'[1/5] original.png  saved  ({bgr.shape[1]}x{bgr.shape[0]})')

    # 2. Blue 通道灰度图
    blue = bgr[:, :, 0]  # OpenCV BGR, channel 0 = Blue
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'blue_channel.png'), blue)
    print(f'[2/5] blue_channel.png  saved')

    # 3. 局部熵图 (disk r=4, 对应约 window_size=9)
    ent_map = compute_local_entropy(blue, radius=4)
    print(f'      entropy range: [{ent_map.min():.2f}, {ent_map.max():.2f}]')

    # 4. 三个尺度的熵图热力图 (统一 colorbar 范围)
    vmin, vmax = ent_map.min(), ent_map.max()

    ent_k1 = avg_pool_np(ent_map, k=1)   # identity
    ent_k5 = avg_pool_np(ent_map, k=5)
    ent_k11 = avg_pool_np(ent_map, k=11)

    for name, data in [('entropy_k1.png', ent_k1),
                       ('entropy_k5.png', ent_k5),
                       ('entropy_k11.png', ent_k11)]:
        save_heatmap(data, os.path.join(OUTPUT_DIR, name))

    idx = list(enumerate([('entropy_k1.png', ent_k1),
                          ('entropy_k5.png', ent_k5),
                          ('entropy_k11.png', ent_k11)]))
    for i, (name, _) in idx:
        print(f'[{i+3}/5] {name}  saved')

    print(f'\n全部子图已保存到: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
