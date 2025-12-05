# train_enhancement_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from enhancement_module import EnhancementModule
from losses import ZeroDCELoss
from datasets import FinetuneDataset


def train_finetune(
        input_dir, target_dir,
        zerodce_weights, denoiser_weights,
        epochs=50, lr=1e-5, batch_size=2,
        image_height=256, image_width=480,
        save_path="enhancement_module_final.pth"
):
    print("--- Starting Stage 2: Jointly Finetuning the Enhancement Module ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EnhancementModule(pretrained_unet_encoder=False).to(device)

    try:
        model.low_light_enhancer.load_state_dict(torch.load(zerodce_weights, map_location=device))
        model.denoiser.load_state_dict(torch.load(denoiser_weights, map_location=device))
        print("Pre-trained weights loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}");
        return

    reconstruction_loss_fn = nn.L1Loss()
    zerodce_aux_loss_fn = ZeroDCELoss()
    lambda1, lambda2 = 1.0, 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Loading datasets from: [Input: {input_dir}], [Target: {target_dir}]")
    try:
        train_dataset = FinetuneDataset(input_dir=input_dir, target_dir=target_dir,
                                        image_size=(image_height, image_width))
        if not train_dataset:
            print(f"Error: No images found in directory: {input_dir}");
            return
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Dataset loaded with {len(train_dataset)} image pairs.")
    except Exception as e:
        print(f"Error loading dataset: {e}");
        return

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for input_imgs, target_imgs in data_iterator:
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            optimizer.zero_grad()

            final_output, intermediate_clamped, A_maps = model(input_imgs)

            l_reconstruction = reconstruction_loss_fn(final_output, target_imgs)
            l_zerodce_aux = zerodce_aux_loss_fn(intermediate_clamped, A_maps)
            total_loss = lambda1 * l_reconstruction + lambda2 * l_zerodce_aux

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            data_iterator.set_postfix(total_loss=f"{total_loss.item():.4f}")
        print(f"Epoch [{epoch + 1}/{epochs}], Average Total Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nFinetuning finished. Final model saved to: {save_path}")


if __name__ == '__main__':
    # !!! 你需要修改这里的路径 !!!
    YOUR_FINETUNE_INPUT_PATH = r"D:\dataset\low_light_rainy"  # 例如：低光照+有雨的图片
    YOUR_FINETUNE_TARGET_PATH = r"D:\dataset\normal_light_clean"  # 例如：对应的正常光照+干净的图片

    # 预训练权重的路径，应与前两个脚本的save_path一致
    ZERODCE_PRETRAINED = "zerodce_pretrained.pth"
    DENOISER_PRETRAINED = "denoiser_pretrained.pth"

    if not os.path.isdir(YOUR_FINETUNE_INPUT_PATH) or not os.path.isdir(YOUR_FINETUNE_TARGET_PATH):
        print(f"Error: One or both finetune directories do not exist.")
    elif not os.path.exists(ZERODCE_PRETRAINED) or not os.path.exists(DENOISER_PRETRAINED):
        print("Error: Pre-trained weight files not found. Please run Stage 1 scripts first.")
    else:
        train_finetune(
            input_dir=YOUR_FINETUNE_INPUT_PATH,
            target_dir=YOUR_FINETUNE_TARGET_PATH,
            zerodce_weights=ZERODCE_PRETRAINED,
            denoiser_weights=DENOISER_PRETRAINED,
            epochs=50,
            lr=1e-5,  # 微调时使用更小的学习率
            batch_size=1,  # 微调时batch size可能需要更小
            image_height=256,
            image_width=480,
        )