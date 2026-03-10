from .coco import build as build_coco
from .line_eval import LineEvaluator
from .collate import BatchImageCollateFunction

def build_dataset(image_set, args):
    dataset_file = getattr(args, 'dataset_file', 'coco')
    if dataset_file == 'musid':
        from stage1_linea_entropy.datasets import build_musid_dataset
        return build_musid_dataset(image_set, args)
    return build_coco(image_set, args)
