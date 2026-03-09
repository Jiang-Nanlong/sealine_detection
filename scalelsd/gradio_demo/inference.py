import torch
import cv2
import os
import gradio as gr
import numpy as np
import random
from pathlib import Path
import json

from scalelsd.ssl.models.detector import ScaleLSD
from scalelsd.base import show, WireframeGraph
from scalelsd.ssl.misc.train_utils import fix_seeds, load_scalelsd_model

# Title for the Gradio interface
_TITLE = 'Gradio Demo of ScaleLSD for Structured Representation of Images'
MAX_SEED = 1000


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

def process_image(
    input_image,
    model_name='scalelsd-vitbase-v2-train-sa1b.pt',
    save_name='temp_output',
    threshold=10,
    junction_threshold_hm=0.008,
    num_junctions_inference=512,
    width=512,
    height=512,
    line_width=2,
    juncs_size=4,
    whitebg=0.0,
    draw_junctions_only=False,
    use_lsd=False,
    use_nms=False,
    edge_color='orange',
    vertex_color='Cyan',
    output_format='png',
    seed=0,
    randomize_seed=False
):
    """core processing function for image inference"""
    # set random seed
    seed = int(randomize_seed_fn(seed, randomize_seed))
    fix_seeds(seed)
    
    # initialize model
    ckpt = "models/" + model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_scalelsd_model(ckpt, device)

    # set model parameters
    model.junction_threshold_hm = junction_threshold_hm
    model.num_junctions_inference = num_junctions_inference

    # transform input image
    if isinstance(input_image, np.ndarray):
        image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.imread(input_image, 0)
    
    # resize
    ori_shape = image.shape[:2]
    image_resized = cv2.resize(image.copy(), (width, height))
    image_tensor = torch.from_numpy(image_resized).float() / 255.0
    image_tensor = image_tensor[None, None].to('cuda')
    
    # meta data
    meta = {
        'width': ori_shape[1],
        'height': ori_shape[0],
        'filename': '',
        'use_lsd': use_lsd,
        'use_nms': use_nms,
    }
    
    # inference
    with torch.no_grad():
        outputs, _ = model(image_tensor, meta)
        outputs = outputs[0]
    
    # visual results
    painter = show.painters.HAWPainter()
    painter.confidence_threshold = threshold
    painter.line_width = line_width
    painter.marker_size = juncs_size
    if whitebg > 0.0:
        show.Canvas.white_overlay = whitebg
    
    temp_folder = "temp_output"
    os.makedirs(temp_folder, exist_ok=True)
    fig_file = f"{temp_folder}/{save_name}.png"
    with show.image_canvas(input_image, fig_file=fig_file) as ax:
        if draw_junctions_only:
            painter.draw_junctions(ax, outputs)
        else:
            painter.draw_wireframe(ax, outputs, edge_color=edge_color, vertex_color=vertex_color)
    # read the result image
    result_image = cv2.imread(fig_file)

    if output_format != 'png':
        fig_file = f"{temp_folder}/{save_name}.{output_format}"
        with show.image_canvas(input_image, fig_file=fig_file) as ax:
            if draw_junctions_only:
                painter.draw_junctions(ax, outputs)
            else:
                painter.draw_wireframe(ax, outputs, edge_color=edge_color, vertex_color=vertex_color)

    json_file = f"{temp_folder}/{save_name}.json"
    indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'],outputs['lines_pred'])
    wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'], indices, outputs['lines_score'], outputs['width'], outputs['height'])
    with open(json_file, 'w') as f:
        json.dump(wireframe.jsonize(),f)


    return result_image[:, :, ::-1], json_file, fig_file
    

def run_demo():
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
            figs_root = "assets/figs"
            example_images = [os.path.join(figs_root, iname) for iname in os.listdir(figs_root)]
            
            with gr.Row():
                input_image = gr.Image(example_images[0], label="Input Image", type="numpy")
                output_image = gr.Image(label="Detection Result")
            
            with gr.Row():
                run_btn = gr.Button(value="Run", variant="primary")
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)
            
            with gr.Row():
                json_file = gr.File(label="Download JSON Output", type="filepath")
                image_file = gr.File(label="Download Image Output", type="filepath")
            
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
                        whitebg = gr.Slider(0.0, 1.0, value=0.7, label="White Background Opacity")
                        line_width = gr.Number(2, label="Line Width")
                        juncs_size = gr.Number(8, label="Junctions Size")
                
                with gr.Row():
                    edge_color = gr.Dropdown(
                        ['orange', 'midnightblue', 'red', 'green'], 
                        value='orange', 
                        label="Edge Color"
                    )
                    vertex_color = gr.Dropdown(
                        ['Cyan', 'deeppink', 'yellow', 'purple'], 
                        value='Cyan', 
                        label="Vertex Color"
                    )
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(False, label="Randomize Seed")
                    seed = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
            
            gr.Examples(
                examples=example_images,
                inputs=input_image,
            )
            
            # star event handlers
            run_event = run_btn.click(
                fn=process_image,
                inputs=[
                    input_image,
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
                outputs=[output_image, json_file, image_file],
            )
            
            # stop event handlers
            stop_btn.click(
                fn=stop_run,
                outputs=[run_btn, stop_btn],
                cancels=[run_event],
                queue=False,
            )

    
    return demo

if __name__ == "__main__":
    run_demo().launch()
