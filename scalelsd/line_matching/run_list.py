import argparse
import os
from os.path import join
import sys

import cv2
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
# from gluestick.models.two_view_pipeline import TwoViewPipeline
from line_matching.two_view_pipeline import TwoViewPipeline

from scalelsd.base import show, WireframeGraph

def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog='GlueStick Demo',
        description='Demo app to show the point and line matches obtained by GlueStick')
    parser.add_argument('-inum', default=None, type=int)
    parser.add_argument('-imax', default=None, type=int)
    parser.add_argument('-img1', default=join('resources' + os.path.sep + 'img1.jpg'))
    parser.add_argument('-img2', default=join('resources' + os.path.sep + 'img2.jpg'))
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--model', default='scalelsd', type=str)
    parser.add_argument('--test_root', type=str, default='data-ssl/0images-pre/')
    args = parser.parse_args()

    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
                # 'merge_line_endpoints': False,
            },
            'max_n_lines': args.max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    pipeline_model.extractor.update_conf(None)
   
    md = args.model

    root = args.test_root
    if args.inum is not None:
        ids = [args.inum]
    elif args.imax is not None:
        ids = range(args.inum, args.imax+1)
    else:
        l_imgs = int(len(os.listdir(root))/2)
        ids = range(l_imgs)

    for id in tqdm(ids):    
        saveto = f'temp_output/matching_results/{md}/{id}'
        os.makedirs(saveto, exist_ok=True)

        args.img1 = root + f'ref_{str(id)}.png'
        args.img2 = root + f'tgt_{str(id)}.png'

        gray0 = cv2.imread(args.img1, 0)
        gray1 = cv2.imread(args.img2, 0)

        torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
        torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]

        x = {'image0': torch_gray0, 'image1': torch_gray1}
        pred = pipeline_model(x)

        pred = batch_to_np(pred)
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
        line_matches = pred["line_matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]

        valid_matches = line_matches != -1
        match_indices = line_matches[valid_matches]
        matched_lines0 = line_seg0[valid_matches]
        matched_lines1 = line_seg1[match_indices]

        # Plot the matches
        gray0 = cv2.imread(args.img1, 0)
        gray1 = cv2.imread(args.img2, 0)
        img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        
        plot_images([img0, img1], dpi=200, pad=2.0)
        plot_lines([line_seg0, line_seg1], ps=4, lw=2)
        plt.gcf().canvas.manager.set_window_title('Detected Lines')
        # plt.tight_layout()
        plt.savefig(f'{saveto}/{md}_det_{id}.png')

        plot_images([img0, img1], dpi=200, pad=2.0)
        plot_color_line_matches([matched_lines0, matched_lines1], lw=3)
        plt.gcf().canvas.manager.set_window_title('Line Matches')
        # plt.tight_layout()
        plt.savefig(f'{saveto}/{md}_mat_{id}.png')

        whitebg = 1
        show.Canvas.white_overlay = whitebg
        painter = show.painters.HAWPainter()

        fig_file = f'{saveto}/{md}_det1.png'
        outputs = {'lines_pred': line_seg0.reshape(-1,4)}
        with show.image_canvas(args.img1, fig_file=fig_file) as ax:
            # painter.draw_wireframe(ax,outputs, edge_color='orange', vertex_color='Cyan')
            painter.draw_wireframe(ax,outputs, edge_color='midnightblue', vertex_color='deeppink')
        fig_file = f'{saveto}/{md}_det2.png'
        outputs = {'lines_pred': line_seg1.reshape(-1,4)}
        with show.image_canvas(args.img2, fig_file=fig_file) as ax:
            # painter.draw_wireframe(ax,outputs, edge_color='orange', vertex_color='Cyan')
            painter.draw_wireframe(ax,outputs, edge_color='midnightblue', vertex_color='deeppink')



if __name__ == '__main__':
    main()
