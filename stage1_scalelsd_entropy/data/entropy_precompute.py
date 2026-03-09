import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF')


def resolve_image_path(img_dir: str, name_in_csv_or_dir: str):
    """
    Resolve an image path from a MU-SID stem or filename.
    Compatible with the user's existing split/CSV convention.
    """
    base = os.path.join(img_dir, str(name_in_csv_or_dir))
    candidates = [
        base,
        base + '.JPG',
        base + '.jpg',
        base + '.png',
        base + '.jpeg',
        base + '.JPEG',
        base + '.PNG',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def list_images(input_dir: str, exts: str = None):
    if exts:
        allow = tuple(e.strip() if e.strip().startswith('.') else '.' + e.strip() for e in exts.split(','))
    else:
        allow = IMAGE_EXTS
    files = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and name.endswith(allow):
            files.append(p)
    return files


def compute_entropy_input(bgr: np.ndarray, mode: str = 'blue') -> np.ndarray:
    if mode == 'blue':
        src = bgr[:, :, 0]  # OpenCV is BGR
    elif mode == 'gray':
        src = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f'Unsupported entropy_mode: {mode}')
    return src.astype(np.uint8)


def compute_local_entropy(src_u8: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError('window_size must be positive')
    # skimage.morphology.disk uses radius, so window_size=9 -> radius=4 approx.
    radius = max(1, int(round((window_size - 1) / 2)))
    ent = entropy(src_u8, disk(radius)).astype(np.float32)
    return ent


def process_single_image(image_path: str, out_path: str, window_size: int = 9, entropy_mode: str = 'blue'):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Failed to read image: {image_path}')

    src = compute_entropy_input(bgr, mode=entropy_mode)
    ent_map = compute_local_entropy(src, window_size)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, ent_map.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description='Offline local-entropy precompute for MU-SID')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--window_size', type=int, default=9)
    parser.add_argument('--entropy_mode', type=str, default='blue', choices=['blue', 'gray'])
    parser.add_argument('--exts', type=str, default='', help='Optional comma-separated extensions, e.g. .JPG,.png')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = list_images(args.input_dir, args.exts)
    if not image_paths:
        raise RuntimeError(f'No images found in {args.input_dir}')

    for image_path in tqdm(image_paths, ncols=80, desc=f'entropy-{args.entropy_mode}'):
        stem = Path(image_path).stem
        out_path = os.path.join(args.output_dir, f'{stem}.npy')
        process_single_image(
            image_path=image_path,
            out_path=out_path,
            window_size=args.window_size,
            entropy_mode=args.entropy_mode,
        )

    print(f'[Done] Saved {len(image_paths)} entropy maps to {args.output_dir}')


if __name__ == '__main__':
    main()
