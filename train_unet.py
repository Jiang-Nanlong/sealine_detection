import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

from unet_model import RestorationGuidedHorizonNet

# STAGE A 训练图像增强，生成 stage_a.pth
# STAGE B 预训练分割头，保持编码器不变，只训练分割头，
# STAGE C 联合微调

def synthesize_rain_fog(image_rgb_u8):
    """
    合成退化（更接近“恶劣天气”的统计特性）：
      - 雨：雨丝 + 雨幕(veiling) + 轻微模糊
      - 雾：非均匀雾（垂直梯度/局部浓淡）+ 对比度下降
      - 低照：整体亮度衰减
      - 统一叠加轻噪声

    输入:
      image_rgb_u8: uint8, RGB, [H,W,3]
    输出:
      float32, RGB, [H,W,3] in [0,1]
    """
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape

    mode = random.choices(["rain", "fog", "dark", "none"], weights=[0.35, 0.35, 0.2, 0.1])[0]

    if mode == "rain":
        severity = random.uniform(0.3, 1.0)

        streak = np.zeros((h, w, 3), dtype=np.float32)
        n_lines = int(300 * severity)
        angle = random.uniform(-15, 15)
        tan_a = np.tan(np.deg2rad(angle))

        for _ in range(n_lines):
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            length = random.randint(int(10 + 20 * severity), int(30 + 50 * severity))
            x1 = int(x0 + length * tan_a)
            y1 = int(y0 + length)
            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            cv2.line(streak, (x0, y0), (x1, y1), (1.0, 1.0, 1.0), 1)

        streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=1.2, sigmaY=1.2)
        alpha = 0.25 * severity
        img = np.clip(img + alpha * streak, 0, 1)

        veil = random.uniform(0.05, 0.25) * severity
        img = img * (1 - veil) + veil * 1.0

        if random.random() < 0.6:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

    elif mode == "fog":
        severity = random.uniform(0.25, 0.9)

        ys = np.linspace(0, 1, h, dtype=np.float32)
        center = random.uniform(0.35, 0.65)
        sigma = random.uniform(0.15, 0.30)
        alpha_line = np.exp(-0.5 * ((ys - center) / sigma) ** 2)
        alpha_line = (alpha_line / alpha_line.max()) * severity
        alpha = np.repeat(alpha_line[:, None], w, axis=1)[:, :, None]

        airlight = np.ones_like(img)
        img = img * (1 - alpha) + airlight * alpha

        c = random.uniform(0.85, 0.98)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * c + mean
        img = np.clip(img, 0, 1)

    elif mode == "dark":
        factor = random.uniform(0.25, 0.7)
        img = np.clip(img * factor, 0, 1)

    noise_sigma = random.uniform(0.005, 0.02)
    img = np.clip(img + np.random.randn(h, w, 3).astype(np.float32) * noise_sigma, 0, 1)

    return img.astype(np.float32)


class HorizonImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=384, mode="joint", ignore_band_px: int = 10):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        self.ignore_band_px = ignore_band_px

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.data)

    def _read_image_rgb(self, img_name: str):
        img_path = os.path.join(self.img_dir, img_name)
        possible_exts = ["", ".JPG", ".jpg", ".png", ".jpeg"]
        image = None
        for ext in possible_exts:
            if os.path.exists(img_path + ext):
                image = cv2.imread(img_path + ext)
                break
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            # FIX: BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _make_sky_mask_from_two_points(self, p1, p2):
        H = W = self.img_size

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        if abs(x2 - x1) < 1e-6:
            x2 = x1 + 1e-3

        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1

        xs = np.arange(W, dtype=np.float32)
        y_line = k * xs + b
        y_line = np.clip(y_line, 0, H - 1)

        ys = np.arange(H, dtype=np.float32)[:, None]
        mask = (ys < y_line[None, :]).astype(np.uint8)  # sky=1

        if self.ignore_band_px and self.ignore_band_px > 0:
            p1i = (int(np.clip(x1, 0, W - 1)), int(np.clip(y1, 0, H - 1)))
            p2i = (int(np.clip(x2, 0, W - 1)), int(np.clip(y2, 0, H - 1)))
            cv2.line(mask, p1i, p2i, 255, self.ignore_band_px)

        return mask

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])

        try:
            x1, y1 = float(row.iloc[1]), float(row.iloc[2])
            x2, y2 = float(row.iloc[3]), float(row.iloc[4])
        except Exception:
            x1, y1, x2, y2 = 0, 0, 0, 0

        image = self._read_image_rgb(img_name)

        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        sx = self.img_size / max(orig_w, 1)
        sy = self.img_size / max(orig_h, 1)
        p1 = (x1 * sx, y1 * sy)
        p2 = (x2 * sx, y2 * sy)

        mask = self._make_sky_mask_from_two_points(p1, p2)
        mask_tensor = torch.from_numpy(mask).long()

        clean_img_np = image_resized.copy()
        target_clean = torch.from_numpy(clean_img_np.astype(np.float32) / 255.0).permute(2, 0, 1)

        degraded_img_np = synthesize_rain_fog(clean_img_np)  # float 0-1
        input_tensor = torch.from_numpy(degraded_img_np).permute(2, 0, 1)
        input_tensor = (input_tensor - self.mean) / self.std

        if self.mode == "restoration":
            return input_tensor, target_clean
        elif self.mode == "segmentation":
            clean_input = torch.from_numpy(clean_img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
            clean_input = (clean_input - self.mean) / self.std
            return clean_input, mask_tensor
        else:
            return input_tensor, target_clean, mask_tensor


def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _build_optimizer(model: RestorationGuidedHorizonNet, stage: str, lr: float):
    enc_params, rest_params, seg_params = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            enc_params.append(p)
        elif name.startswith(("rest_up", "rest_conv", "rest_out")):
            rest_params.append(p)
        else:
            seg_params.append(p)

    if stage == "A":
        _set_requires_grad(model, True)
        # freeze seg branch (optional)
        _set_requires_grad(model.strip_pool, False)
        _set_requires_grad(model.seg_lat5, False)
        _set_requires_grad(model.seg_lat4, False)
        _set_requires_grad(model.seg_lat3, False)
        _set_requires_grad(model.seg_conv_fuse, False)
        _set_requires_grad(model.injection_conv, False)
        _set_requires_grad(model.seg_head, False)
        _set_requires_grad(model.seg_final, False)

        return optim.AdamW(
            [{"params": enc_params, "lr": lr * 0.1}, {"params": rest_params, "lr": lr}],
            weight_decay=1e-4,
        )

    if stage == "B":
        # freeze encoder + restoration
        _set_requires_grad(model.encoder, False)
        for prefix in ["rest_up1", "rest_conv1", "rest_up2", "rest_conv2", "rest_up3", "rest_conv3", "rest_up4", "rest_out"]:
            _set_requires_grad(getattr(model, prefix), False)
        # enable seg
        _set_requires_grad(model.strip_pool, True)
        _set_requires_grad(model.seg_lat5, True)
        _set_requires_grad(model.seg_lat4, True)
        _set_requires_grad(model.seg_lat3, True)
        _set_requires_grad(model.seg_conv_fuse, True)
        _set_requires_grad(model.injection_conv, True)
        _set_requires_grad(model.seg_head, True)
        _set_requires_grad(model.seg_final, True)

        return optim.AdamW([{"params": seg_params, "lr": lr}], weight_decay=1e-4)

    _set_requires_grad(model, True)
    return optim.AdamW(
        [{"params": enc_params, "lr": lr * 0.1}, {"params": rest_params, "lr": lr}, {"params": seg_params, "lr": lr}],
        weight_decay=1e-4,
    )


def train_rghnet():
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"

    BATCH_SIZE = 16
    IMG_SIZE = 384
    LR = 1e-4
    EPOCHS = 30

    STAGE = "C"  # 'A'/'B'/'C'
    print(f"当前训练阶段: {STAGE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda")

    mode_map = {"A": "restoration", "B": "segmentation", "C": "joint"}
    dataset = HorizonImageDataset(CSV_PATH, IMG_DIR, img_size=IMG_SIZE, mode=mode_map[STAGE], ignore_band_px=10)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, _ = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = RestorationGuidedHorizonNet(num_classes=2).to(device)

    criterion_rest = nn.L1Loss()
    criterion_seg = nn.CrossEntropyLoss(ignore_index=255)  # FIX

    stage_a_ckpt = "stage_a.pth"
    if STAGE == "C" and os.path.exists(stage_a_ckpt):
        print(f"Stage C: 加载 {stage_a_ckpt} ...")
        state = torch.load(stage_a_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)

    optimizer = _build_optimizer(model, STAGE, LR)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if STAGE == "A":
                    img, target_clean = batch
                    img = img.to(device, non_blocking=True)
                    target_clean = target_clean.to(device, non_blocking=True)

                    pred_clean, _ = model(img)
                    loss = criterion_rest(pred_clean, target_clean)

                elif STAGE == "B":
                    img, mask = batch
                    img = img.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    _, pred_seg = model(img)
                    loss = criterion_seg(pred_seg, mask)

                else:
                    img, target_clean, mask = batch
                    img = img.to(device, non_blocking=True)
                    target_clean = target_clean.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    pred_clean, pred_seg = model(img)
                    loss_r = criterion_rest(pred_clean, target_clean)
                    loss_s = criterion_seg(pred_seg, mask)
                    loss = loss_r + 0.5 * loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        print(f"Epoch {epoch + 1} Loss: {running_loss / max(len(train_loader), 1):.4f}")

        ckpt_path = f"rghnet_stage_{STAGE.lower()}.pth"
        torch.save(model.state_dict(), ckpt_path)

    print("训练完成。")


if __name__ == "__main__":
    train_rghnet()
