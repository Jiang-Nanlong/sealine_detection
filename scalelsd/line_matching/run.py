import argparse
import os
from os.path import join
import sys
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from line_matching.two_view_pipeline import TwoViewPipeline

from scalelsd.base import show, WireframeGraph
from scalelsd.ssl.datasets.transforms.homographic_transforms import sample_homography
from kornia.geometry import warp_perspective,transform_points

class HADConfig:
    num_iter = 1
    valid_border_margin = 3
    translation = True
    rotation = True
    scale = True
    perspective = True 
    scaling_amplitude = 0.2
    perspective_amplitude_x = 0.2
    perspective_amplitude_y = 0.2
    allow_artifacts = False
    patch_ratio = 0.85
had_cfg = HADConfig()

def sample_homographics(height, width):

    def scale_homography(H, stride):
        H_scaled = H.clone()
        H_scaled[:, :, 2, :2] *= stride
        H_scaled[:, :, :2, 2] /= stride
        return H_scaled

    homographic = sample_homography(
        shape = (height, width),
        perspective = had_cfg.perspective,
        scaling = had_cfg.scale,
        rotation = had_cfg.rotation,
        translation = had_cfg.translation,
        scaling_amplitude = had_cfg.scaling_amplitude,
        perspective_amplitude_x = had_cfg.perspective_amplitude_x,
        perspective_amplitude_y = had_cfg.perspective_amplitude_y,
        patch_ratio = had_cfg.patch_ratio,
        allow_artifacts = False
        )[0]

    homographic = torch.from_numpy(homographic[None]).float().cuda()
    homographic_inv = torch.inverse(homographic)

    H = {
        'h.1': homographic,
        'ih.1': homographic_inv,
    }
    
    return H 

def trans_image_with_homograpy(image):
    h, w = image.shape[:2]
    H = sample_homographics(height=h, width=w)

    image_warped = warp_perspective(torch.Tensor(image).permute(2,0,1)[None].cuda(), H['h.1'], (h,w))
    image_warped_ = image_warped[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    plt.imshow(image_warped_)
    plt.show()
    return image_warped_


def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog='GlueStick Demo',
        description='Demo app to show the point and line matches obtained by GlueStick')
    parser.add_argument('-img1', default='assets/figs/sa_1119229.jpg')
    parser.add_argument('-img2', default=None)
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--model', type=str, default='models/paper-sa1b-997pkgs-model.pt')
    args = parser.parse_args()

    # important
    if args.img1 is None and args.img2 is None:
        raise ValueError("Input at least one path of image1 or image2")

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
   
    saveto = f'temp_output/matching_results'
    os.makedirs(saveto, exist_ok=True)

    image1 = cv2.cvtColor(cv2.imread(args.img1), cv2.COLOR_BGR2RGB)
    if args.img2 is None:
        image2 = trans_image_with_homograpy(image1)
        cv2.imwrite(f'{saveto}/warped_image.png', image2)
        args.img2 = f'{saveto}/warped_image.png'

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
    plt.savefig(f'{saveto}/det.png')

    plot_images([img0, img1], dpi=200, pad=2.0)
    plot_color_line_matches([matched_lines0, matched_lines1], lw=3)
    plt.gcf().canvas.manager.set_window_title('Line Matches')
    # plt.tight_layout()
    plt.savefig(f'{saveto}/mat.png')

    whitebg = 1
    show.Canvas.white_overlay = whitebg
    painter = show.painters.HAWPainter()

    fig_file = f'{saveto}/det1.png'
    outputs = {'lines_pred': line_seg0.reshape(-1,4)}
    with show.image_canvas(args.img1, fig_file=fig_file) as ax:
        # painter.draw_wireframe(ax,outputs, edge_color='orange', vertex_color='Cyan')
        painter.draw_wireframe(ax,outputs, edge_color='midnightblue', vertex_color='deeppink')
    fig_file = f'{saveto}/det2.png'
    outputs = {'lines_pred': line_seg1.reshape(-1,4)}
    with show.image_canvas(args.img2, fig_file=fig_file) as ax:
        painter.draw_wireframe(ax,outputs, edge_color='midnightblue', vertex_color='deeppink')



if __name__ == '__main__':
    main()
