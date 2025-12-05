# train_zerodce.py
# (此文件与上一轮回答中的最终版本一致，这里为保持完整性再次提供)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from zerodce import ZeroDCE
from losses import ZeroDCELoss
from datasets import LowLightDataset

def train_zerodce_standalone(
    dataset_dir, epochs=100, lr=1e-4, batch_size=4,
    image_height=256, image_width=480, save_path="zerodce_pretrained.pth"
):
    print("--- Starting Stage 1, Part A: Pre-training ZeroDCE ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ZeroDCE().to(device)
    criterion = ZeroDCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Loading dataset from: {dataset_dir}")
    try:
        train_dataset = LowLightDataset(root_dir=dataset_dir, image_size=(image_height, image_width))
        if not train_dataset:
            print(f"Error: No images found in directory: {dataset_dir}"); return
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Dataset loaded with {len(train_dataset)} images.")
    except Exception as e:
        print(f"Error loading dataset: {e}"); return

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for low_light_imgs in data_iterator:
            low_light_imgs = low_light_imgs.to(device)
            optimizer.zero_grad()
            enhanced_imgs, A_maps = model(low_light_imgs)
            loss = criterion(enhanced_imgs, A_maps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            data_iterator.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining finished. ZeroDCE pre-trained weights saved to: {save_path}")

if __name__ == '__main__':
    YOUR_DATASET_PATH = r"D:\dataset\low light"
    if not os.path.isdir(YOUR_DATASET_PATH):
        print(f"Error: Directory does not exist: {YOUR_DATASET_PATH}")
    else:
        train_zerodce_standalone(
            dataset_dir=YOUR_DATASET_PATH,
            epochs=100,
            lr=1e-4,
            batch_size=2,
            image_height=256,
            image_width=480,
        )