try:
    from pcache_fileio import fileio
except:
    pass

import os
import os.path as osp
import glob
import math
import copy
from skimage.io import imread
from skimage import color
import PIL
from PIL import Image
import numpy as np
import h5py
import cv2
import pickle
import torch
import torch.utils.data.dataloader as torch_loader
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path 
import json 

from ..config.project_config import Config as cfg
from .transforms import photometric_transforms as photoaug
from .transforms import homographic_transforms as homoaug
from .transforms.utils import random_scaling
from .synthetic_util import get_line_heatmap
from ..misc.train_utils import parse_h5_data
from ..misc.geometry_utils import warp_points, mask_points
from tqdm import tqdm

def images_collate_fn(batch):
    """ Customized collate_fn for wireframe dataset. """
    batch_keys = ["image", "junction_map", "valid_mask", "heatmap",
                  "heatmap_pos", "heatmap_neg", "homography",
                  "line_points", "line_indices"]
    list_keys = ["junctions", "line_map", "line_map_pos",
                 "line_map_neg", "file_key","fname","image_origin","uuid"]

    outputs = {}
    for data_key in batch[0].keys():
        batch_match = sum([_ == data_key for _ in batch_keys])
        list_match = sum([_ == data_key for _ in list_keys])
        # print(batch_match, list_match)
        if batch_match > 0 and list_match == 0:
            outputs[data_key] = torch_loader.default_collate(
                [b[data_key] for b in batch])
        elif batch_match == 0 and list_match > 0:
            outputs[data_key] = [b[data_key] for b in batch]
        elif batch_match == 0 and list_match == 0:
            continue
        else:
            raise ValueError(
        "[Error] A key matches batch keys and list keys simultaneously.")

    return outputs

class ImageCollections(Dataset):
    def __init__(self, mode, config, homoadp=False,homoadp_resume=False):
        super(ImageCollections, self).__init__()
        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config
        h5path = config.get('gt_source_train',None)
        self.json_list = None
        self.homoadp = homoadp
        self.homoadp_resume = homoadp_resume

        if self.config['img_reg_exp'] == 'all':
            self.config['img_reg_exp'] = []
            for i in range(998):
                self.config['img_reg_exp'].append(f'sa_{i:06d}/images/*.jpg')

        if h5path is not None and h5path.endswith('.h5'):
            self.h5path = osp.join(cfg.EXPORT_ROOT,'export_datasets',h5path)
            with h5py.File(self.h5path,'r') as f:
                self.filenames = [k.decode('UTF-8') for k in f['filenames']]
            self.filenames = [osp.join(cfg.EXPORT_ROOT,f) for f in self.filenames]
        elif h5path is not None and h5path.endswith('.jsons'):
            self.use_json = True
            self.h5path = osp.join(cfg.EXPORT_ROOT,'export_datasets',h5path)
            #json_list = glob.glob(self.h5path+'/*.json')
            json_list = []
            for exp in tqdm(self.config['img_reg_exp']):
                _json_regexp = Path(exp).with_suffix('.json')
                _jsons = glob.glob(osp.join(self.h5path,str(_json_regexp)))
                json_list.extend(_jsons)
            
            if cfg.DATASET_ROOT.startswith('pcache'):
                
                if osp.isfile(Path(h5path).with_suffix('.pcache')) and (self.homoadp_resume or not self.homoadp):
                    with open(Path(h5path).with_suffix('.pcache'),'r') as _f:
                        filenames = _f.readlines()
                    filenames = [ x.rstrip('\n') for x in filenames ]
                else:
                    self.folder_regexp = []
                    filenames = []
                    print('Loading from pcache......')
                    for exp in tqdm(self.config['img_reg_exp']):
                        _path = Path(osp.join(self.config['dataset_root'][0],exp))
                        _p = osp.join(cfg.DATASET_ROOT,str(_path.parent))

                        _e = _path.suffix
                        _list = [osp.basename(_) for _ in os.listdir(_p) if _.endswith(_e)]
                        _list = [osp.join(_p,_) for _ in _list]
                        filenames.extend(_list)
                    
                    with open(Path(h5path).with_suffix('.pcache'),'w') as _f:
                        _f.writelines('\n'.join(filenames))
            else:
                self.folder_regexp = [osp.join(cfg.DATASET_ROOT,self.config['dataset_root'][0],exp) for exp in self.config['img_reg_exp']]
                filenames = sum([glob.glob(exp) for exp in self.folder_regexp],[])
                filenames = [Path(f) for f in filenames]
            
            self.dataset_root = osp.join(cfg.DATASET_ROOT,self.config['dataset_root'][0])
            filedict = {str(Path(osp.relpath(f,self.dataset_root)).with_suffix('')): f for f in filenames}
            
            jsondict = {str(Path(osp.relpath(j,h5path)).with_suffix('')): j for j in json_list}
            self.filenames = []
            self.json_list = []
            
            if self.homoadp:
                for k in filedict.keys():
                    if k in jsondict and self.homoadp_resume:
                        continue
                    else:
                        self.filenames.append(str(filedict[k]))
                self.h5path = None
                self.use_json = False
                print(f"Found {len(json_list)} json files from the folder")
                print(f"Total images are reduced from {len(filenames)} to {len(self.filenames)}")
            else:
                for k in filedict.keys():
                    if k in jsondict:
                        self.filenames.append(str(filedict[k]))
                        self.json_list.append(str(jsondict[k]))
        else:
            self.folder_regexp = [osp.join(cfg.DATASET_ROOT,self.config['dataset_root'][0],exp) for exp in self.config['img_reg_exp']]
            self.filenames = sum([glob.glob(exp) for exp in self.folder_regexp],[])
            self.h5path = None
        
        self.default_config = self.get_default_config()

        self.dataset_name = self.config['alias']

        self.size = self.config['preprocessing']['resize']
        
        print("Found %d images in %s" % (len(self),self.config['dataset_root']))
        
        self.num_pad = int(math.ceil(math.log10(len(self))))+1 if len(self)>0 else 0
        
    def __len__(self):
        return len(self.filenames)

    def get_padded_filename(self, num_pad, idx):
        file_len = len("%d" % (idx))
        filename = "0" * (num_pad - file_len) + "%d" % (idx)
        return filename

    def train_preprocessing(self, data, numpy=False):
        """ Train preprocessing for the dataset. """
        image = data['image']
        junctions = data.get('junctions',None)
        image_size = image.shape[:2]
        if not(list(image_size) == self.config['preprocessing']['resize']):
            size_old = list(image.shape)[:2]

            image = cv2.resize(image, tuple(self.config['preprocessing']['resize'][::-1]), interpolation=cv2.INTER_LINEAR)

            scales = (image.shape[0] / size_old[0], image.shape[1] / size_old[1])
            
            if junctions is not None:
                junctions *= torch.tensor(scales).reshape(1, 2)
        
        if self.config['augmentation']['photometric']['enable']:
            photo_trans_list = self.get_photo_transform()
            ### Apply photometric transforms
            np.random.shuffle(photo_trans_list)
            image_transform = transforms.Compose(photo_trans_list + [photoaug.normalize_image()])
        else:
            image_transform = photoaug.normalize_image()

        image = image_transform(image)

        if self.config['augmentation']['homographic']['enable']:
            homo_trans = self.get_homo_transform()
            outputs = homo_trans(image, junctions, data['line_map'])
            junctions = outputs["junctions"]
            image = outputs["warped_image"]
            line_map = outputs['line_map']
            data['line_map'] = torch.tensor(line_map)
            data['valid_mask'] = outputs['valid_mask']

        data['image'] = torch.from_numpy(image)[None]
        if junctions is not None:
            data['junctions'] = torch.from_numpy(junctions).float()
        
        
        return data

    def get_homo_transform(self):
        """ Get homographic transforms (according to the config). """
        # Get homographic transforms for image
        homo_config = self.config["augmentation"]["homographic"]["params"]
        if not self.config["augmentation"]["homographic"]["enable"]:
            raise ValueError(
        "[Error] Homographic augmentation is not enabled.")

        # Parse the homographic transforms
        image_shape = self.config["preprocessing"]["resize"]

        # Compute the min_label_len from config
        try:
            min_label_tmp = self.config["generation"]["min_label_len"]
        except:
            min_label_tmp = None
        
        # float label len => fraction
        if isinstance(min_label_tmp, float): # Skip if not provided
            min_label_len = min_label_tmp * min(image_shape)
        # int label len => length in pixel
        elif isinstance(min_label_tmp, int):
            scale_ratio = (self.config["preprocessing"]["resize"]
                           / self.config["generation"]["image_size"][0])
            min_label_len = (self.config["generation"]["min_label_len"]
                             * scale_ratio)
        # if none => no restriction
        else:
            min_label_len = 0
        
        # Initialize the transform
        homographic_trans = homoaug.homography_transform(
            image_shape, homo_config, 0, min_label_len)

        return homographic_trans

    def get_photo_transform(self):
        """ Get list of photometric transforms (according to the config). """
        # Get the photometric transform config
        photo_config = self.config["augmentation"]["photometric"]
        if not photo_config["enable"]:
            raise ValueError(
        "[Error] Photometric augmentation is not enabled.")
        
        # Parse photometric transforms
        trans_lst = self.parse_transforms(photo_config["primitives"],
                                          photoaug.available_augmentations)
        trans_config_lst = [photo_config["params"].get(p, {})
                            for p in trans_lst]

        # List of photometric augmentation
        photometric_trans_lst = [
            getattr(photoaug, trans)(**conf) \
            for (trans, conf) in zip(trans_lst, trans_config_lst)
        ]

        return photometric_trans_lst

    def parse_transforms(self, names, all_transforms):
        """ Parse the transform. """
        trans = all_transforms if (names == 'all') \
            else (names if isinstance(names, list) else [names])
        assert set(trans) <= set(all_transforms)
        return trans

    def check_files(self):
        h5path = self.config.get('gt_source_train',None)
        valid_filenames = []
        for filename in self.filenames:
            try:
                image_origin = np.array(PIL.Image.open(filename))
                valid_filenames.append(filename)
            except IOError:
                print(f"Unable to load image from path: {filename}")

        new_pcache_path = Path(h5path).with_name(f"{Path(h5path).stem}_filtered.pcache")
        with open(new_pcache_path, 'w') as _f:
            for filename in valid_filenames:
                _f.write(f"{filename}\n")

    def check_health(self):
        is_healthy = True
        image_fail_list = []
        json_fail_list = []
        for idx in tqdm(range(len(self))):
            #try:
            #    image_origin = np.array(PIL.Image.open(self.filenames[idx]))
            #except:
            #    is_healthy = False
            #    print(f'The image {self.filenames[idx]} is broken.')
            #    image_fail_list.append(self.filenames[idx])

            if self.h5path is not None and self.json_list is not None:
                try:
                    with open(self.json_list[idx],'r') as f:
                        data = json.load(f)
                except:
                    is_healthy = False
                    print(f'The image {self.filenames[idx]} is broken.')
                    json_fail_list.append(self.json_list[idx])
        return {
            'images': image_fail_list,
            'jsons': json_fail_list,
            'status': is_healthy
        }

    def __getitem__(self, idx):
        fname = osp.basename(self.filenames[idx])
        #image_origin = cv2.imread(self.filenames[idx])
        try:
            image_origin = np.array(PIL.Image.open(self.filenames[idx]))
        except:
            image_origin = np.array(PIL.Image.open('hawp/ssl/config/exports/sa1b/00030043_0.png')) # deal with the failed case

        if self.config['gray_scale']:
            image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        
        # image = np.array(image,dtype=np.float32)/255.0
        
        data = {
            'fname': self.filenames[idx],
            'image': image,
            # 'image': torch.from_numpy(image)[None],
            # 'valid_mask': torch.ones(self.size,dtype=torch.float32)[None],
            'image_origin': image_origin,
        }
        data['uuid'] = osp.relpath(self.filenames[idx],self.dataset_root)
        
        if self.h5path is not None and self.json_list is None:
            with h5py.File(self.h5path,'r') as f:
                gt_key = self.get_padded_filename(self.num_pad,idx)
                exported_label = parse_h5_data(f[gt_key])
            junctions = torch.tensor(exported_label['junctions']).float()
            edges = torch.tensor(exported_label['edges']).long()
            lines = junctions[edges]
            junctions_valid = torch.zeros(len(junctions),dtype=torch.bool)
            junctions_valid[edges.unique()] = 1
            junctions_idx = -torch.ones(len(junctions),dtype=torch.long)
            junctions_idx[junctions_valid] = torch.arange(junctions_valid.sum())
            edges_remapped = junctions_idx[edges]
            junctions = junctions[junctions_valid]
            lines_remapped = junctions[edges_remapped]
            line_map = torch.zeros(junctions.shape[0],junctions.shape[0],dtype=torch.float32)
            if len(edges_remapped) > 0:
                line_map[edges_remapped[:,0],edges_remapped[:,1]] = 1
                line_map[edges_remapped[:,1],edges_remapped[:,0]] = 1

            data['line_map'] = line_map
            data['junctions'] = junctions[:,[1,0]]
        elif self.h5path is not None and self.json_list is not None:
            with open(self.json_list[idx],'r') as f:
                json_data = json.load(f)
            junctions = torch.tensor(json_data['junctions']).float()
            if junctions.shape[0] == 0:
                junctions = torch.zeros((1,2)).float()
            edges = torch.tensor(json_data['edges']).long()
            lines = junctions[edges]
            junctions_valid = torch.zeros(len(junctions),dtype=torch.bool)
            junctions_valid[edges.unique()] = 1
            junctions_idx = -torch.ones(len(junctions),dtype=torch.long)
            junctions_idx[junctions_valid] = torch.arange(junctions_valid.sum())
            edges_remapped = junctions_idx[edges]
            junctions = junctions[junctions_valid]
            lines_remapped = junctions[edges_remapped]
            line_map = torch.zeros(junctions.shape[0],junctions.shape[0],dtype=torch.float32)
            if len(edges_remapped) > 0:
                line_map[edges_remapped[:,0],edges_remapped[:,1]] = 1
                line_map[edges_remapped[:,1],edges_remapped[:,0]] = 1

            data['line_map'] = line_map
            data['junctions'] = junctions[:,[1,0]]
        else:
            data['valid_mask'] = torch.ones(self.size,dtype=torch.float32)[None]
            
        return self.train_preprocessing(data)
        return data # TODO: remove this line
        
    def get_default_config(self):
        return {
            "dataset_name": "images",
            "add_augmentation_to_all_splits": False,
            "preprocessing": {
                "resize": [512,512],
                "blur_size": 11,
            },
            "augmentation": {
                "photometric": {
                    "enable": False
                },
                "homographic": {
                    "enable": False
                }
            }
        }