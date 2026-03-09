""" YorkUrban dataset for VP estimation evaluation. """

import os
import csv
import numpy as np
import torch
import cv2
import scipy.io
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from ..config.project_config import Config as cfg


def unproject_vp_to_world(vp, K):
    """ Convert the VPs from homogenous format in the image plane
        to world direction. """
    proj_vp = (np.linalg.inv(K) @ vp.T).T
    proj_vp[:, 1] *= -1
    proj_vp /= np.linalg.norm(proj_vp, axis=1, keepdims=True)
    return proj_vp

class NYU(torch.utils.data.Dataset):
    def __init__(self, mode='test', config=None):

        # assert mode in ['val', 'test']

        # Extract the image names
        num_imgs = 1449
        val_size = -49
        
        self.root_dir = cfg.nyu_dataroot
        self.img_paths = [os.path.join(self.root_dir, 'images', 'nyu_rgb_'+str(i+1).zfill(4) + '.png')
                          for i in range(num_imgs)]
        self.vps_paths = [os.path.join(self.root_dir, 'vps', 'vps_' + str(i).zfill(4) + '.csv')
            for i in range(num_imgs)]
        self.lines_paths = [os.path.join(self.root_dir, 'labelled_lines', 'labelled_lines_' + str(i).zfill(4) + '.csv')
            for i in range(num_imgs)]
        self.img_names = [str(i).zfill(4) for i in range(num_imgs)]

        # Separate validation and test
        if mode == 'val':
            self.img_paths = self.img_paths[-val_size:]
            self.vps_paths = self.vps_paths[-val_size:]
            self.lines_paths = self.lines_paths[-val_size:]
            self.img_names = self.img_names[-val_size:]
        elif mode == 'test':
            self.img_paths = self.img_paths[:-val_size]
            self.vps_paths = self.vps_paths[:-val_size]
            self.lines_paths = self.lines_paths[:-val_size]
            self.img_names = self.img_names[:-val_size]

        # Load the intrinsics
        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        self.K = torch.tensor([[fx_rgb, 0, cx_rgb],
                               [0, fy_rgb, cy_rgb],
                               [0, 0, 1]])

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = str(Path(img_path).stem)
        img = cv2.imread(img_path)

        # Load the GT VPs
        vps = []
        with open(self.vps_paths[idx]) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for ri, row in enumerate(reader):
                if ri == 0:
                    continue
                vps.append([float(row[1]), float(row[2]), 1.])
        vps = unproject_vp_to_world(np.array(vps), self.K.numpy())

        lines = []
        with open(self.lines_paths[idx]) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for ri, row in enumerate(reader):
                if ri == 0:
                    continue
                lines.append([float(row[1]), float(row[2]), 1.])

        # Normalize the images in [0, 1]
        # img = img.astype(float) / 255.

        # Convert to torch tensors
        # img = torch.tensor(img[None], dtype=torch.float)
        vps = torch.tensor(vps, dtype=torch.float)
        lines = torch.tensor(lines, dtype=torch.float)

        data = {'image': img,
                'image_path': img_path,
                'name': name, 
                'gt_lines': lines,
                'vps': vps, 
                'K': self.K
                }

        return data      

    def __len__(self):
        return len(self.img_paths)

    # Overwrite the parent data loader to handle custom split
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['val', 'test', 'export']
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=False, pin_memory=True,
                          num_workers=num_workers)