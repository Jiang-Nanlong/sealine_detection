# train_denoiser.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from denoise_unet import DenoiseUNet
from datasets import NoisyCleanDataset

def train_denoiser_standalone(
    noisy_dir, clean_dir, epochs=150, lr=1e-4, batch_size=4,
    image_height=256, image_width=480, save_path="denoiser_pretrained.pth"
):
    print("--- Starting Stage 1, Part B: Pre-training Denoise U-Net ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DenoiseUNet(pretrained_encoder=True).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Loading datasets from: [Noisy: {noisy_dir}], [Clean: {clean_dir}]")
    try:
        train_dataset = NoisyCleanDataset(noisy_dir=noisy_dir, clean_dir=clean_dir, image_size=(image_height, image_width))
        if not train_dataset:
            print(f"Error: No images found in directory: {noisy_dir}"); return
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Dataset loaded with {len(train_dataset)} image pairs.")
    except Exception as e:
        print(f"Error loading dataset: {e}"); return

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for noisy_imgs, clean_imgs in data_iterator:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            optimizer.zero_grad()
            denoised_outputs = model(noisy_imgs)
            loss = criterion(denoised_outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            data_iterator.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Average L1 Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining finished. Denoise U-Net pre-trained weights saved to: {save_path}")

if __name__ == '__main__':
    # !!! 你需要修改这里的路径 !!!
    YOUR_NOISY_DATA_PATH = r"D:\dataset\rainy_images"  # 例如：包含雨天图片的文件夹
    YOUR_CLEAN_DATA_PATH = r"D:\dataset\clean_images"  # 例如：对应的干净背景图片文件夹

    if not os.path.isdir(YOUR_NOISY_DATA_PATH) or not os.path.isdir(YOUR_CLEAN_DATA_PATH):
        print(f"Error: One or both directories do not exist.")
    else:
        train_denoiser_standalone(
            noisy_dir=YOUR_NOISY_DATA_PATH,
            clean_dir=YOUR_CLEAN_DATA_PATH,
            epochs=150,
            lr=1e-4,
            batch_size=2,
            image_height=256,
            image_width=480,
        )