import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_rgb(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_entropy(entropy_path: str):
    ent = np.load(entropy_path).astype(np.float32)
    return np.clip(ent / 8.0, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='Visualize entropy map for manual checking')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--entropy', type=str, required=True)
    parser.add_argument('--entropy2', type=str, default='')
    parser.add_argument('--label1', type=str, default='entropy1')
    parser.add_argument('--label2', type=str, default='entropy2')
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    rgb = load_rgb(args.image)
    ent1 = load_entropy(args.entropy)
    ent2 = load_entropy(args.entropy2) if args.entropy2 else None

    os.makedirs(args.save_dir, exist_ok=True)
    stem = Path(args.image).stem

    if ent2 is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title('image')
        axes[1].imshow(ent1, cmap='jet', vmin=0.0, vmax=1.0)
        axes[1].set_title(args.label1)
        axes[2].imshow(rgb)
        axes[2].imshow(ent1, cmap='jet', alpha=0.45, vmin=0.0, vmax=1.0)
        axes[2].set_title('overlay')
    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb)
        axes[0].set_title('image')
        axes[1].imshow(ent1, cmap='jet', vmin=0.0, vmax=1.0)
        axes[1].set_title(args.label1)
        axes[2].imshow(ent2, cmap='jet', vmin=0.0, vmax=1.0)
        axes[2].set_title(args.label2)
        diff = np.clip(np.abs(ent1 - ent2), 0.0, 1.0)
        axes[3].imshow(diff, cmap='magma', vmin=0.0, vmax=1.0)
        axes[3].set_title('|diff|')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(args.save_dir, f'{stem}_entropy_vis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {out_path}')


if __name__ == '__main__':
    main()
