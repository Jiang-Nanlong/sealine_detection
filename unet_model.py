import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import os
from collections import OrderedDict

from modules import DoubleConv, StripPooling
from zerodce import C_DCE_Net


class RestorationGuidedHorizonNet(nn.Module):
    def __init__(self, num_classes=2, dce_weights_path="Epoch99.pth"):
        super().__init__()

        # =========================================================
        # 0. Zero-DCE++ (Frozen)
        # =========================================================
        self.dce_net = C_DCE_Net()

        if os.path.exists(dce_weights_path):
            try:
                state = torch.load(dce_weights_path, map_location='cpu')

                # 兼容可能出现的 'module.' 前缀
                new_state = OrderedDict()
                for k, v in state.items():
                    nk = k.replace('module.', '')
                    new_state[nk] = v

                # 先尝试 strict=True（完全匹配最安全）；失败再退化到 strict=False
                try:
                    self.dce_net.load_state_dict(new_state, strict=True)
                    print("[RG-HNet] Zero-DCE++ 权重加载成功! 前置增强已开启。")
                    self.dce_enabled = True
                except RuntimeError as e:
                    missing, unexpected = self.dce_net.load_state_dict(new_state, strict=False)
                    total = len(self.dce_net.state_dict())

                    # 如果缺失/多余过多，宁可禁用，避免“半残权重”破坏输入
                    if (len(missing) / max(1, total) > 0.25) or (len(unexpected) / max(1, total) > 0.25):
                        print(f"[RG-HNet] 警告: Zero-DCE++ 权重匹配度过低，missing={len(missing)}/{total}, unexpected={len(unexpected)}。DCE 将被禁用。")
                        self.dce_enabled = False
                    else:
                        print(f"[RG-HNet] Zero-DCE++ 权重已加载 (strict=False). missing={len(missing)}, unexpected={len(unexpected)}")
                        self.dce_enabled = True

                # 冻结 DCE
                for param in self.dce_net.parameters():
                    param.requires_grad = False

            except Exception as e:
                print(f"[RG-HNet] 警告: Zero-DCE++ 权重加载失败 ({e})。DCE 将被禁用 (Identity Mapping)。")
                self.dce_enabled = False
        else:
            print("[RG-HNet] 警告: 未找到 DCE 权重文件。DCE 将被禁用。")
            self.dce_enabled = False

        # =========================================================
        # 1. 共享编码器 (MobileNetV3-Large)
        # =========================================================
        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)
        return_nodes = {'features.3': 'c2', 'features.6': 'c3', 'features.12': 'c4', 'features.16': 'c5'}
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)
        c2_ch, c3_ch, c4_ch, c5_ch = 24, 40, 112, 960

        # ImageNet 均值方差
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # =========================================================
        # 2. 复原分支 (UNet-like decoder) - 使用 GroupNorm
        # =========================================================
        norm_type = 'gn'

        self.rest_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.rest_conv1 = DoubleConv(c5_ch + c4_ch, 256, norm_type=norm_type)

        self.rest_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.rest_conv2 = DoubleConv(256 + c3_ch, 128, norm_type=norm_type)

        self.rest_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.rest_conv3 = DoubleConv(128 + c2_ch, 64, norm_type=norm_type)

        self.rest_up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.rest_out = nn.Conv2d(64, 3, kernel_size=1)

        # =========================================================
        # 3. 分割分支 (FPN-like + StripPooling)
        # =========================================================
        self.strip_pool = StripPooling(c5_ch)
        self.seg_lat5 = nn.Conv2d(c5_ch, 256, 1)
        self.seg_lat4 = nn.Conv2d(c4_ch, 256, 1)
        self.seg_lat3 = nn.Conv2d(c3_ch, 256, 1)

        self.seg_conv_fuse = DoubleConv(256, 128, norm_type='bn')

        # 注入复原特征 (r3: 64ch) -> seg_feat (64ch)
        self.injection_conv = nn.Conv2d(64, 64, 1)

        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.seg_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x, target_clean=None, enable_restoration: bool = True, enable_segmentation: bool = True):
        """Forward.

        Args:
            x: [B,3,H,W] range [0,1] (Rainy/Foggy)
            target_clean: [B,3,H,W] range [0,1] (Clean GT, optional)
            enable_restoration: 是否计算复原分支 (Stage B 设 False 可显著提速)
            enable_segmentation: 是否计算分割分支 (Stage A 设 False 可提速)

        Returns:
            restored_img (or None),
            seg_logits (or None),
            target_enhanced (or None)
        """
        input_size = x.shape[-2:]

        # --- Step 0: Zero-DCE++ Enhancement ---
        if self.dce_enabled:
            with torch.no_grad():
                x_enhanced = self.dce_net(x)
                if target_clean is not None:
                    target_enhanced = self.dce_net(target_clean)
                else:
                    target_enhanced = None
        else:
            x_enhanced = x
            target_enhanced = target_clean if target_clean is not None else None

        # Normalize for MobileNet (ImageNet stats)
        x_norm = (x_enhanced - self.img_mean) / self.img_std

        # --- Step 1: Encoder ---
        feats = self.encoder(x_norm)
        c2, c3, c4, c5 = feats['c2'], feats['c3'], feats['c4'], feats['c5']

        restored_img = None
        r3 = None

        # --- Step 2: Restoration Branch ---
        if enable_restoration:
            r1 = self.rest_conv1(torch.cat([self.rest_up1(c5), c4], dim=1))
            r2 = self.rest_conv2(torch.cat([self.rest_up2(r1), c3], dim=1))
            r3 = self.rest_conv3(torch.cat([self.rest_up3(r2), c2], dim=1))

            res_out = self.rest_out(self.rest_up4(r3))
            restored_img = torch.clamp(x_enhanced + res_out, 0.0, 1.0)

        # --- Step 3: Segmentation Branch ---
        seg_logits = None
        if enable_segmentation:
            s5 = self.strip_pool(c5)
            p5 = self.seg_lat5(s5)
            p4 = self.seg_lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
            p3 = self.seg_lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')

            seg_base = self.seg_conv_fuse(p3)
            seg_feat = self.seg_head(seg_base)

            # Feature Injection
            if enable_restoration and r3 is not None:
                rest_inject = self.injection_conv(r3)
                rest_inject = F.interpolate(rest_inject, size=seg_feat.shape[-2:], mode='bilinear', align_corners=False)
                seg_fused = seg_feat + rest_inject
            else:
                # Stage B 为了速度跳过复原分支：注入项置零
                seg_fused = seg_feat

            seg_logits = self.seg_final(seg_fused)

            if seg_logits.shape[-2:] != input_size:
                seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)

        return restored_img, seg_logits, target_enhanced
