import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from modules import DoubleConv, StripPooling


class RestorationGuidedHorizonNet(nn.Module):
    """
    Restoration-Guided Horizon Net (RGHNet)

    - Shared Encoder: MobileNetV3-Large (ImageNet pretrained)
    - Restoration branch: U-Net style decoder, predicts residual in [0,1] space
    - Segmentation branch: FPN-style + StripPooling + Restoration feature injection

    IMPORTANT FIX:
      训练时输入 x 做了 ImageNet Normalize（均值方差归一化），因此不能直接做：
          restored = clamp(x + residual, 0, 1)
      因为 x 不是 0~1 的图像域。

      本版本会在 forward 里把 x 反归一化回 x_raw（近似 0~1），再做残差相加：
          restored = clamp(x_raw + residual, 0, 1)
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # =========================================================
        # 1. 共享编码器 (MobileNetV3-Large)
        # =========================================================
        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)

        # c2: 1/4, c3: 1/8, c4: 1/16, c5: 1/32
        return_nodes = {
            "features.3": "c2",   # 24 ch
            "features.6": "c3",   # 40 ch
            "features.12": "c4",  # 112 ch
            "features.16": "c5",  # 960 ch
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        c2_ch, c3_ch, c4_ch, c5_ch = 24, 40, 112, 960

        # 保存 mean/std（用于 forward 里反归一化）
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        # =========================================================
        # 2. Restoration Branch (Derain/Dehaze)
        # =========================================================
        # 1/32 -> 1/16
        self.rest_up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rest_conv1 = DoubleConv(c5_ch + c4_ch, 256)

        # 1/16 -> 1/8
        self.rest_up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rest_conv2 = DoubleConv(256 + c3_ch, 128)

        # 1/8 -> 1/4
        self.rest_up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rest_conv3 = DoubleConv(128 + c2_ch, 64)  # r3_feat (注入特征)

        # 1/4 -> 1 (输出 residual)
        self.rest_up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.rest_out = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

        # =========================================================
        # 3. Segmentation Branch (Sea/Sky)
        # =========================================================
        self.strip_pool = StripPooling(c5_ch)

        self.seg_lat5 = nn.Conv2d(c5_ch, 256, 1)
        self.seg_lat4 = nn.Conv2d(c4_ch, 256, 1)
        self.seg_lat3 = nn.Conv2d(c3_ch, 256, 1)

        self.seg_conv_fuse = DoubleConv(256, 128)  # 1/8

        # Restoration Injection
        self.injection_conv = nn.Conv2d(64, 64, 1)

        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 1/8 -> 1/2
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.seg_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 1/2 -> 1
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B,3,H,W] normalized by ImageNet mean/std
        Returns:
            restored_img: [B,3,H,W] in [0,1]
            seg_logits:   [B,C,H,W]
        """
        input_size = x.shape[-2:]

        # 反归一化回近似 0~1 的图像域（用于残差相加）
        x_raw = torch.clamp(x * self.img_std + self.img_mean, 0.0, 1.0)

        # --- 1. Shared Encoder ---
        feats = self.encoder(x)
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        # --- 2. Restoration Branch ---
        r1 = self.rest_up1(c5)
        r1 = torch.cat([r1, c4], dim=1)
        r1 = self.rest_conv1(r1)  # 1/16

        r2 = self.rest_up2(r1)
        r2 = torch.cat([r2, c3], dim=1)
        r2 = self.rest_conv2(r2)  # 1/8

        r3 = self.rest_up3(r2)
        r3 = torch.cat([r3, c2], dim=1)
        r3_feat = self.rest_conv3(r3)  # 1/4, 64ch

        # residual output (add on x_raw)
        res_out = self.rest_out(self.rest_up4(r3_feat))
        restored_img = torch.clamp(x_raw + res_out, 0.0, 1.0)

        # --- 3. Segmentation Branch ---
        s5 = self.strip_pool(c5)

        p5 = self.seg_lat5(s5)
        p4 = self.seg_lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.seg_lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)

        seg_base = self.seg_conv_fuse(p3)  # 1/8

        seg_feat = self.seg_head(seg_base)  # 1/2, 64ch

        # injection: 1/4 -> 1/2
        rest_inject = self.injection_conv(r3_feat)
        rest_inject = F.interpolate(rest_inject, size=seg_feat.shape[-2:], mode="bilinear", align_corners=False)
        seg_fused = seg_feat + rest_inject

        seg_logits = self.seg_final(seg_fused)

        if seg_logits.shape[-2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)

        return restored_img, seg_logits
