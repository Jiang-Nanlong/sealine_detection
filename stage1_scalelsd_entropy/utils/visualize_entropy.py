import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Global config for PyCharm / server-side direct execution
# Edit these variables directly before running this file.
# ============================================================
IMAGE_PATH = "Hashmani's Dataset/MU-SID/DSC_0051_9.JPG"
ENTROPY_PATH = "Hashmani's Dataset/MU-SID_entropy_blue/DSC_0051_9.npy"
ENTROPY_PATH_2 = ""   # optional second entropy map for comparison
LABEL_1 = "blue"
LABEL_2 = "gray"
SAVE_DIR = "stage1_scalelsd_entropy/vis_check"
OVERLAY_ALPHA = 0.45


def load_rgb(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_entropy(entropy_path: str):
    ent = np.load(entropy_path).astype(np.float32)
    return np.clip(ent / 8.0, 0.0, 1.0)


def main():
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f'IMAGE_PATH not found: {IMAGE_PATH}')
    if not os.path.isfile(ENTROPY_PATH):
        raise FileNotFoundError(f'ENTROPY_PATH not found: {ENTROPY_PATH}')
    if ENTROPY_PATH_2 and not os.path.isfile(ENTROPY_PATH_2):
        raise FileNotFoundError(f'ENTROPY_PATH_2 not found: {ENTROPY_PATH_2}')

    rgb = load_rgb(IMAGE_PATH)
    ent1 = load_entropy(ENTROPY_PATH)
    ent2 = load_entropy(ENTROPY_PATH_2) if ENTROPY_PATH_2 else None

    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = Path(IMAGE_PATH).stem

    if ent1.shape[:2] != rgb.shape[:2]:
        ent1 = cv2.resize(ent1, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if ent2 is not None and ent2.shape[:2] != rgb.shape[:2]:
        ent2 = cv2.resize(ent2, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    if ent2 is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title('image')
        axes[1].imshow(ent1, cmap='jet', vmin=0.0, vmax=1.0)
        axes[1].set_title(LABEL_1)
        axes[2].imshow(rgb)
        axes[2].imshow(ent1, cmap='jet', alpha=OVERLAY_ALPHA, vmin=0.0, vmax=1.0)
        axes[2].set_title('overlay')
    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb)
        axes[0].set_title('image')
        axes[1].imshow(ent1, cmap='jet', vmin=0.0, vmax=1.0)
        axes[1].set_title(LABEL_1)
        axes[2].imshow(ent2, cmap='jet', vmin=0.0, vmax=1.0)
        axes[2].set_title(LABEL_2)
        diff = np.clip(np.abs(ent1 - ent2), 0.0, 1.0)
        axes[3].imshow(diff, cmap='magma', vmin=0.0, vmax=1.0)
        axes[3].set_title('|diff|')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, f'{stem}_entropy_vis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {out_path}')


if __name__ == '__main__':
    main()
