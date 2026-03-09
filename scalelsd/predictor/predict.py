import torch
import random 
import numpy as np
import os
import os.path as osp
import glob
from tqdm import tqdm

from scalelsd.base import setup_logger, MetricLogger, show, WireframeGraph

from scalelsd.ssl.datasets import dataset_util
from scalelsd.ssl.models.detector import ScaleLSD
from scalelsd.ssl.misc.train_utils import load_scalelsd_model

from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader

from pathlib import Path
import argparse, yaml, logging, time, datetime, cv2, copy, sys, json
from easydict import EasyDict
import accelerate
from accelerate import load_checkpoint_and_dispatch
import matplotlib
import matplotlib.pyplot as plt 

def parse_args():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-c', '--ckpt', default='models/scalelsd-vitbase-v1-train-sa1b.pt', type=str, help='the path for loading checkpoints')
    aparser.add_argument('-t','--threshold', default=10,type=float)
    aparser.add_argument('-i', '--img', required=True, type=str)
    aparser.add_argument('--width', default=512, type=int)
    aparser.add_argument('--height', default=512,type=int)
    aparser.add_argument('--whitebg', default=0.0, type=float)
    aparser.add_argument('--saveto', default=None, type=str,)
    aparser.add_argument('-e','--ext', default='pdf', type=str, choices=['pdf','png','json','txt'])
    aparser.add_argument('--device', default='cuda', type=str, choices=['cuda','cpu','mps'])
    aparser.add_argument('--disable-show', default=False, action='store_true')
    aparser.add_argument('--draw-junctions-only', default=False, action='store_true')
    aparser.add_argument('--use_lsd', default=False, action='store_true')
    aparser.add_argument('--use_nms', default=False, action='store_true')

    ScaleLSD.cli(aparser)

    args = aparser.parse_args()
    
    ScaleLSD.configure(args)

    return args


def main():
    args = parse_args()

    model = load_scalelsd_model(args.ckpt, device=args.device)

    # Set up output directory and painter
    if args.saveto is None:
        print('No output directory specified, saving outputs to folder: temp_output/ScaleLSD')
        args.saveto = 'temp_output/ScaleLSD'
    os.makedirs(args.saveto,exist_ok=True)

    show.painters.HAWPainter.confidence_threshold = args.threshold
    # show.painters.HAWPainter.line_width = 2
    # show.painters.HAWPainter.marker_size = 4
    show.Canvas.show = not args.disable_show
    if args.whitebg > 0.0:
        show.Canvas.white_overlay = args.whitebg
    painter = show.painters.HAWPainter()
    edge_color = 'orange' # 'midnightblue'
    vertex_color = 'Cyan' # 'deeppink'

    # Prepare images
    all_images = []
    if os.path.isfile(args.img) and args.img.endswith(('.jpg', '.png')):
        all_images.append(args.img)
    elif os.path.isdir(args.img):
        for file in os.listdir(args.img):
            if file.endswith(('.jpg', '.png')):
                fname = os.path.join(args.img, file)
                all_images.append(fname)
        all_images = sorted(all_images)
    else:
        raise ValueError('Input must be a file or a directory containing images.')

    # Inference
    for fname in tqdm(all_images):
        pname = Path(fname)
        image = cv2.imread(fname,0)
        
        # for resize input, default shape is [512, 512]
        ori_shape = image.shape[:2]
        image_cp = copy.deepcopy(image)
        image_ = cv2.resize(image_cp, (args.width, args.height))
        image_ = torch.from_numpy(image_).float()/255.0
        image_ = image_[None,None].to(args.device)
        
        meta = {
            'width': ori_shape[1],
            'height':ori_shape[0],
            'filename': '',
            'use_lsd': args.use_lsd,
            'use_nms': args.use_nms,
        }

        with torch.no_grad():
            outputs, _ = model(image_, meta)
            outputs = outputs[0]


        if args.saveto is not None:

            if args.ext in ['png', 'pdf']:
                fig_file = osp.join(args.saveto, pname.with_suffix('.'+args.ext).name)
                with show.image_canvas(fname, fig_file=fig_file) as ax:
                    if args.draw_junctions_only:
                        painter.draw_junctions(ax,outputs)
                    else:
                        # painter.draw_wireframe(ax,outputs)
                        painter.draw_wireframe(ax,outputs, edge_color=edge_color, vertex_color=vertex_color)
            elif args.ext == 'json':
                indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'],outputs['lines_pred'])
                wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'], indices, outputs['lines_score'], outputs['width'], outputs['height'])
                outpath = osp.join(args.saveto, pname.with_suffix('.json').name)
                with open(outpath,'w') as f:
                    json.dump(wireframe.jsonize(),f)
            else:
                raise ValueError('Unsupported extension: {} is not in [png, pdf, json]'.format(args.ext))
        

if __name__ == "__main__":
    main()
