import glob
import os

from setuptools import find_packages
from setuptools import setup

setup(
    name="scalelsd",
    version="1.0",
    author="Zeran Ke and Nan Xue",
    description="Scalable Deep Line Segment Detection Streamlined",
    packages=find_packages(),
    install_requires=[
        "torch", 
        "torchvision",
        "accelerate",
        "tensorboard",
        "timm",
        "opencv-python-headless==4.8.1.78",
        "kornia",
        "cython",
        "matplotlib",
        "yacs",
        "scikit-image",
        "tqdm",
        "python-json-logger",
        "h5py",
        "shapely",
        "seaborn",
        "easydict",
    ],
    extras_require={
        "dev": [
            "pycolmap",
        ]
    }
)
