"""
This file contains some useful functions for train / val.
"""
import os
import numpy as np
import torch
import random 
from scalelsd.ssl.models.detector import ScaleLSD


################
## HDF5 utils ##
################
def parse_h5_data(h5_data):
    """ Parse h5 dataset. """
    output_data = {}
    for key in h5_data.keys():
        output_data[key] = np.array(h5_data[key])
        
    return output_data


def fix_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # torch.backends.cudnn.allow_tf32 = args.tf32
    # torch.backends.cuda.matmul.allow_tf32 = args.tf32
    # torch.backends.cudnn.deterministic = args.dtm

    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)


def load_scalelsd_model(ckpt_path, device='cuda'):
    """load model"""
    use_layer_scale = False if 'v1' in ckpt_path else True

    model = ScaleLSD(gray_scale=True, use_layer_scale=use_layer_scale)
    model = model.eval().to(device)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(state_dict['model_state'])
    except:
        model.load_state_dict(state_dict)

    return model

