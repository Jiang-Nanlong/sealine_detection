from pathlib import Path
import cv2
import PIL
import numpy as np
import torch
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from .transforms.homographic_transforms import sample_homography
from kornia.geometry import warp_perspective,transform_points


homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}

class Hybrid_Dataset(torch.utils.data.Dataset):
    def __init__(self, datacfg=None, images_root=None, overwrite=False):
        self.conf = datacfg
        self.root = images_root

        # torch.manual_seed(self.conf.seed)
        # np.random.seed(self.conf.seed)

        # # Extract images paths
        # self.files = [Path(self.root)/img for img in Path(self.root).iterdir()
        #                if img.with_suffix('.png') or img.with_suffix('.jpg')]
        self.files = glob.glob(f'{images_root}/*.png') + glob.glob(f'{images_root}/*.jpg')
        self.files.sort()

        self.npz_files = [] if overwrite else glob.glob(f'{images_root}/*.npz')

        self.size = (512, 512)

        self.overwrite = overwrite

        if len(self.files) == 0:
            raise ValueError(f'Could not find any images in the path of {self.root}. Please check the input images root path.')
        
        # Randomly generate the homography for each image to ensure reproducibility
        for file in tqdm(self.files):
            npz_file = Path(file).with_suffix('.npz')
            if not npz_file.exists() or self.overwrite:
                image = cv2.imread(file, 0)
                image = cv2.resize(image, self.size)
                image = np.array(image, dtype=np.float32)/255.0

                w, h = image.shape[:2]
                H = sample_homography(self.size, **homography_params)[0]
                warped_image = cv2.warpPerspective(image, H, self.size)
                warped_image = np.array(warped_image, dtype=np.float32)

                data = {
                    'ref_image': image,
                    'target_image': warped_image,
                    'homo_mat': H,
                }

                np.savez(npz_file, ref_image=image, target_image=warped_image, homo_mat=H)

                self.npz_files.append(npz_file)

    def get_dataset(self):
        return self.npz_files
    
    def get_images(self):
        return self.files

    def len_dataset(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        npz_file = self.npz_files(idx)
        data = np.load(npz_file)

        return data
