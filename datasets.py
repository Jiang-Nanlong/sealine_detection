# datasets.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class LowLightDataset(Dataset):
    """
    用于 train_zerodce.py
    这个数据集只需要一个文件夹路径，它会加载该文件夹下所有的低光照图像。
    """

    def __init__(self, root_dir, image_size=(256, 480)):
        self.root_dir = root_dir
        # 过滤掉非图片文件，确保健壮性
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


class NoisyCleanDataset(Dataset):
    """
    用于 train_denoiser.py
    这个数据集需要两个文件夹：一个放有噪/雨的输入图，一个放干净的目标图。
    !!! 核心要求: 两个文件夹下的对应图片的文件名必须完全一致 !!!
    例如:
    - noisy_dir/DSC_1234.JPG
    - clean_dir/DSC_1234.JPG
    """

    def __init__(self, noisy_dir, clean_dir, image_size=(256, 480)):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.image_files = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy_image = Image.open(noisy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        return self.transform(noisy_image), self.transform(clean_image)


class FinetuneDataset(NoisyCleanDataset):
    """
    用于 train_enhancement_finetune.py
    输入是 "低光照+有噪/雨" (input_dir)
    目标是 "正常光照+干净" (target_dir)
    它的结构和 NoisyCleanDataset 完全一样，我们直接继承它。
    """

    def __init__(self, input_dir, target_dir, image_size=(256, 480)):
        super().__init__(noisy_dir=input_dir, clean_dir=target_dir, image_size=image_size)