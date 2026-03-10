"""
entropy_precompute.py — MU-SID 图像局部熵离线预计算

功能：
  对 MU-SID 每张图像提取 blue 通道（或灰度），用滑动窗口计算局部 Shannon 熵，
  输出单通道 float32 的 .npy 文件（保存原始熵值，不做归一化）。

关键设计：
  - 支持 blue / gray 两种模式，默认 blue
  - 保存原始熵值（约 [0, 8]），归一化留到 Dataset 读取阶段
  - 使用 skimage.filters.rank.entropy + disk 结构元素

用法：
  直接运行本文件，或修改顶部全局变量后运行。
"""
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm

# ============================================================
# Global config for PyCharm / server-side direct execution
# Edit these variables directly before running this file.
# ============================================================
# 自动定位项目根目录（stage1_linea_entropy/data/ 往上两级）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

INPUT_DIR = os.path.join(_PROJECT_ROOT, "Hashmani's Dataset", "MU-SID")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "Hashmani's Dataset", "MU-SID_entropy_blue")
WINDOW_SIZE = 9
ENTROPY_MODE = "blue"   # "blue" or "gray"
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF')
OVERWRITE = False

# If True, compute only the first N images for debugging.
DEBUG = False
DEBUG_LIMIT = 20


def list_images(input_dir: str, exts=EXTS):
    files = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and name.endswith(exts):
            files.append(p)
    return files


def compute_entropy_input(bgr: np.ndarray, mode: str = 'blue') -> np.ndarray:
    if mode == 'blue':
        src = bgr[:, :, 0]  # OpenCV BGR, channel 0 is blue
    elif mode == 'gray':
        src = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f'Unsupported ENTROPY_MODE: {mode}')
    return src.astype(np.uint8)


def compute_local_entropy(src_u8: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError('WINDOW_SIZE must be positive')
    radius = max(1, int(round((window_size - 1) / 2)))
    return entropy(src_u8, disk(radius)).astype(np.float32)


def process_single_image(image_path: str, out_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')

    src = compute_entropy_input(bgr, mode=ENTROPY_MODE)
    ent_map = compute_local_entropy(src, WINDOW_SIZE)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, ent_map.astype(np.float32))


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f'INPUT_DIR not found: {INPUT_DIR}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        raise RuntimeError(f'No images found in {INPUT_DIR}')

    if DEBUG:
        image_paths = image_paths[:DEBUG_LIMIT]

    skipped = 0
    processed = 0
    for image_path in tqdm(image_paths, ncols=100, desc=f'entropy-{ENTROPY_MODE}'):
        stem = Path(image_path).stem
        out_path = os.path.join(OUTPUT_DIR, f'{stem}.npy')

        if os.path.exists(out_path) and not OVERWRITE:
            skipped += 1
            continue

        process_single_image(image_path, out_path)
        processed += 1

    print(f'[Done] mode={ENTROPY_MODE}, window={WINDOW_SIZE}, processed={processed}, skipped={skipped}')
    print(f'[Output] {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
