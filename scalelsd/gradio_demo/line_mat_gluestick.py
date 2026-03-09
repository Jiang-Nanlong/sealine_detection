import argparse
import os
from os.path import join
import sys
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import gradio as gr
import random

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches

from scalelsd.ssl.models.detector import ScaleLSD
from scalelsd.base import show, WireframeGraph
from scalelsd.ssl.datasets.transforms.homographic_transforms import sample_homography
from scalelsd.ssl.misc.train_utils import fix_seeds
from line_matching.two_view_pipeline import TwoViewPipeline

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

# Evaluation config
default_conf = {
    'name': 'two_view_pipeline',
    'use_lines': True,
    'extractor': {
        'name': 'wireframe',
        'sp_params': {
            'force_num_keypoints': False,
            'max_num_keypoints': 2048,
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
            # 'merge_line_endpoints': False,
        },
        'max_n_lines': 512,
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

# Title for the Gradio interface
_TITLE = 'ScaleLSD-GlueStick Line Matching'
MAX_SEED = 1000

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

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """random seed"""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def stop_run():
    """stop run"""
    return (
        gr.update(value="Run", variant="primary", visible=True),
        gr.update(visible=False),
    )

def clear_image2():
    return None  # returning None will clear the image component

def process_image(
    input_image1='assets/figs/sa_1119229.jpg',
    input_image2=None,
    model_name='scalelsd-vitbase-v1-train-sa1b.pt',
    save_name='temp',
    threshold=5,
    junction_threshold_hm=0.008,
    num_junctions_inference=4096,
    width=512,
    height=512,
    line_width=2,
    juncs_size=4,
    whitebg=1.0,
    draw_junctions_only=False,
    use_lsd=False,
    use_nms=False,
    edge_color='midnightblue',
    vertex_color='deeppink',
    output_format='png',
    seed=0,
    randomize_seed=False
):
    """core processing function for image inference"""
    # set random seed
    seed = int(randomize_seed_fn(seed, randomize_seed))
    fix_seeds(seed)
    
    conf = {
        'model_name': model_name,
        'threshold': threshold,
        'junction_threshold_hm': junction_threshold_hm,
        'num_junctions_inference': num_junctions_inference,
        'use_lsd': use_lsd,
        'use_nms': use_nms,
        'width': width,
        'height': height,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_model = TwoViewPipeline(default_conf).to(device).eval()
    pipeline_model.extractor.update_conf(conf)

    saveto = f'temp_output/matching_results'
    image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{saveto}/image.png', image1)
    input_image1 = f'{saveto}/image.png'
    if input_image2 is None:
        image2 = trans_image_with_homograpy(image1)
    else:
        image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{saveto}/image2.png', image2)
    input_image2 = f'{saveto}/image2.png'
    
    gray0 = cv2.imread(input_image1, 0)
    gray1 = cv2.imread(input_image2, 0)

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

    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    
    mat_file = f'{saveto}/{save_name}_mat.png'
    plot_images([img0, img1], dpi=200, pad=2.0)
    plot_lines([line_seg0, line_seg1], ps=4, lw=2)
    plt.gcf().canvas.manager.set_window_title('Detected Lines')
    # plt.tight_layout()
    plt.savefig(mat_file)
    det_image = cv2.imread(mat_file)[:,:,::-1]

    det_file = f'{saveto}/{save_name}_mat.png'
    plot_images([img0, img1], dpi=200, pad=2.0)
    plot_color_line_matches([matched_lines0, matched_lines1], lw=3)
    plt.gcf().canvas.manager.set_window_title('Line Matches')
    # plt.tight_layout()
    plt.savefig(det_file)
    mat_image = cv2.imread(det_file)[:,:,::-1]

    show.Canvas.white_overlay = whitebg
    painter = show.painters.HAWPainter()

    fig_file = f'{saveto}/{save_name}_det1.png'
    outputs = {'lines_pred': line_seg0.reshape(-1,4)}
    with show.image_canvas(input_image1, fig_file=fig_file) as ax:
        painter.draw_wireframe(ax,outputs, edge_color=edge_color, vertex_color=vertex_color)
    det1_image = cv2.imread(fig_file)[:,:,::-1]

    fig_file = f'{saveto}/{save_name}_det2.png'
    outputs = {'lines_pred': line_seg1.reshape(-1,4)}
    with show.image_canvas(input_image2, fig_file=fig_file) as ax:
        painter.draw_wireframe(ax,outputs, edge_color=edge_color, vertex_color=vertex_color)
    det2_image = cv2.imread(fig_file)[:,:,::-1]

    return image2[:,:,::-1], mat_image, det_image, det1_image, det2_image, mat_file, det_file
    

def demo():
    """create the Gradio demo interface"""
    css = """
    #col-container {
        margin: 0 auto;
        max-width: 800px;
    }
    """
    
    with gr.Blocks(css=css, title=_TITLE) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(f'# {_TITLE}')
            gr.Markdown("Detect wireframe structures in images using ScaleLSD model")
            
            pid = gr.State()
            figs_root = "assets/mat_figs"
            example_single = [os.path.join(figs_root, 'single', iname) for iname in os.listdir(figs_root+'/single')]
            example_pairs = [[img, None] for img in example_single]
            example_pairs += [
                [os.path.join(figs_root, 'pairs', f'ref_{i}.png'), 
                 os.path.join(figs_root, 'pairs', f'tgt_{i}.png')]
                for i in [10, 72, 76, 95, 149, 151]
            ]
            
            with gr.Row():
                input_image1 = gr.Image(example_pairs[0][0], label="Input Image1", type="numpy")
                input_image2 = gr.Image(label="Input Image2", type="numpy")
            
            with gr.Row():
                mat_images = gr.Image(label="Matching Results")
            with gr.Row():
                det_images = gr.Image(label="Detection Results")
            with gr.Row():
                det_image1 = gr.Image(label="Detection1")
                det_image2 = gr.Image(label="Detection2")

            with gr.Row():
                run_btn = gr.Button(value="Run", variant="primary")
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)
            
            with gr.Row():
                mat_file = gr.File(label="Download Matching Result", type="filepath")
                det_file = gr.File(label="Download Detection Result", type="filepath")
            
            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    model_name = gr.Dropdown(
                        [ckpt for ckpt in os.listdir('models') if ckpt.endswith('.pt')],
                        value='scalelsd-vitbase-v1-train-sa1b.pt', 
                        label="Model Selection"
                    )

                with gr.Row():
                    save_name = gr.Textbox('temp_output', label="Save Name", placeholder="Name for saving output files")

                with gr.Row():
                    with gr.Column():
                        threshold = gr.Number(10, label="Line Threshold")
                        junction_threshold_hm = gr.Number(0.008, label="Junction Threshold")
                        num_junctions_inference = gr.Number(1024, label="Max Number of Junctions")
                        width = gr.Number(512, label="Input Width")
                        height = gr.Number(512, label="Input Height")
                    
                    with gr.Column():
                        draw_junctions_only = gr.Checkbox(False, label="Show Junctions Only")
                        use_lsd = gr.Checkbox(False, label="Use LSD-Rectifier")
                        use_nms = gr.Checkbox(True, label="Use NMS")
                        output_format = gr.Dropdown(
                            ['png', 'jpg', 'pdf'], 
                            value='png', 
                            label="Output Format"
                        )
                        whitebg = gr.Slider(0.0, 1.0, value=1.0, label="White Background Opacity")
                        line_width = gr.Number(2, label="Line Width")
                        juncs_size = gr.Number(8, label="Junctions Size")
                
                with gr.Row():
                    edge_color = gr.Dropdown(
                        ['orange', 'midnightblue', 'red', 'green'], 
                        value='midnightblue', 
                        label="Edge Color"
                    )
                    vertex_color = gr.Dropdown(
                        ['Cyan', 'deeppink', 'yellow', 'purple'], 
                        value='deeppink', 
                        label="Vertex Color"
                    )
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(False, label="Randomize Seed")
                    seed = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
            
            gr.Examples(
                examples=example_pairs,
                inputs=[input_image1, input_image2]
            )
            
            # star event handlers
            run_event = run_btn.click(
                fn=process_image,
                inputs=[
                    input_image1,
                    input_image2,
                    model_name,
                    save_name,
                    threshold,
                    junction_threshold_hm,
                    num_junctions_inference,
                    width,
                    height,
                    line_width,
                    juncs_size,
                    whitebg,
                    draw_junctions_only,
                    use_lsd,
                    use_nms,
                    edge_color,
                    vertex_color,
                    output_format,
                    seed,
                    randomize_seed
                ],
                outputs=[input_image2, mat_images, det_images, det_image1, det_image2, mat_file, det_file],
            )
            
            # stop event handlers
            stop_btn.click(
                fn=stop_run,
                outputs=[run_btn, stop_btn],
                cancels=[run_event],
                queue=False,
            )
            
            # When image1 changes, image2 is cleared
            input_image1.change(
                fn=clear_image2,
                outputs=input_image2
            )

    
    return demo

if __name__ == "__main__":
    # 启动应用
    demo = demo()
    demo.launch()
