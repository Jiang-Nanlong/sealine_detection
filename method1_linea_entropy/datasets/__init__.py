"""
method1_linea_entropy/datasets — MU-SID dataset 构建入口

模式控制（通过 args.entropy_mode）：
  'baseline'   → include_entropy=False, dataset 返回 2-tuple (image, target)
  'entropy'    → include_entropy=True,  dataset 返回 4-tuple, entropy_map [1, H, W]
  'entropy_ms' → include_entropy=True,  dataset 返回 4-tuple, entropy_map [3, H, W] (多尺度)

使用方式：
  from method1_linea_entropy.datasets import build_musid_dataset
  dataset_train = build_musid_dataset('train', args)
  dataset_val   = build_musid_dataset('val', args)

必需 args 字段：
  musid_img_dir      : str  — MU-SID 图像目录
  musid_entropy_dir  : str  — 预计算 entropy .npy 目录
  musid_split_dir    : str  — splits_musid/ 目录
  eval_spatial_size  : list or int — 正方形尺寸 (e.g. [640, 640] 或 640)

可选 args 字段：
  entropy_mode       : str  — 'baseline' / 'entropy' / 'entropy_ms'（默认 'baseline'）
"""

import os
from method1_linea_entropy.datasets.musid_linea_entropy import MUSIDLineaEntropyDataset


_SPLIT_MAP = {
    'train': 'GroundTruth_train.csv',
    'val':   'GroundTruth_val.csv',
    'test':  'GroundTruth_test.csv',
}


def build_musid_dataset(image_set, args):
    """
    构建 MU-SID dataset，根据 args.entropy_mode 决定是否包含 entropy map。

    Args:
        image_set : 'train' / 'val' / 'test'
        args      : 配置对象

    Returns:
        MUSIDLineaEntropyDataset
    """
    if image_set not in _SPLIT_MAP:
        raise ValueError(f'Unknown image_set: {image_set}')

    csv_file = os.path.join(args.musid_split_dir, _SPLIT_MAP[image_set])

    sz = args.eval_spatial_size
    if isinstance(sz, (list, tuple)):
        sz = sz[0]

    mode = getattr(args, 'entropy_mode', 'baseline')
    include_entropy = (mode in ('entropy', 'entropy_ms'))
    multiscale_entropy = (mode == 'entropy_ms')

    return MUSIDLineaEntropyDataset(
        csv_file=csv_file,
        img_dir=args.musid_img_dir,
        entropy_dir=args.musid_entropy_dir,
        img_size=sz,
        image_set=image_set,
        include_entropy=include_entropy,
        multiscale_entropy=multiscale_entropy,
    )
