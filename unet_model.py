import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from modules import DoubleConv, StripPooling
from zerodce import C_DCE_Net


class RestorationGuidedHorizonNet(nn.Module):
    """
    Zero-DCE++ (frozen) + MobileNetV3 shared encoder + dual-branch (restoration + segmentation)
    """

    def __init__(self, num_classes=2, dce_weights_path="Epoch99.pth", require_dce=True):
        super().__init__()

        # =========================================================
        # 0. Zero-DCE++ (Frozen)
        # =========================================================
        self.dce_net = C_DCE_Net()
        self.dce_enabled = False

        if not os.path.exists(dce_weights_path):
            msg = f"[RG-HNet] 未找到 Zero-DCE++ 权重文件: {dce_weights_path}"
            if require_dce:
                raise FileNotFoundError(msg)
            else:
                print(msg + " -> DCE 将被禁用")
                self.dce_enabled = False
        else:
            # 你这里的 Epoch99.pth 已经是 OrderedDict state_dict（你给过 keys）
            # 用 weights_only=True 也能消掉 pytorch 的 FutureWarning
            try:
                state = torch.load(dce_weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                # 兼容旧版本 pytorch 没有 weights_only 参数
                state = torch.load(dce_weights_path, map_location="cpu")

            # 安全：去掉可能存在的 module. 前缀（你当前这份看起来没有，但加上不亏）
            if isinstance(state, (dict, OrderedDict)):
                new_state = OrderedDict()
                for k, v in state.items():
                    nk = k.replace("module.", "")
                    new_state[nk] = v
                state = new_state

            # Stage A 需要 DCE：严格匹配，不匹配就报错（避免“悄悄禁用”白训练）
            try:
                self.dce_net.load_state_dict(state, strict=True)
                self.dce_enabled = True
                print("[RG-HNet] Zero-DCE++ 权重加载成功，DCE 已启用。")
            except Exception as e:
                msg = (
                    "[RG-HNet] Zero-DCE++ 权重加载失败（结构不匹配或权重非本模型）。\n"
                    f"  权重文件: {dce_weights_path}\n"
                    "  你这份权重的 key 形如 e_conv*.depth_conv / point_conv，"
                    "必须使用深度可分离卷积版本的 C_DCE_Net。\n"
                    f"  原始错误: {repr(e)}"
                )
                if require_dce:
                    raise RuntimeError(msg)
                else:
                    print(msg + "\n-> DCE 将被禁用（Identity Mapping）")
                    self.dce_enabled = False

        # 冻结 DCE
        for p in self.dce_net.parameters():
            p.requires_grad = False

        # =========================================================
        # 1. Shared Encoder (MobileNetV3-Large)
        # =========================================================
        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)
        return_nodes = {
            "features.3": "c2",   # ~1/4
            "features.6": "c3",   # ~1/8
            "features.12": "c4",  # ~1/16
            "features.16": "c5",  # ~1/32
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        # 这些 channel 对 mobilenet_v3_large 是固定的
        c2_ch, c3_ch, c4_ch, c5_ch = 24, 40, 112, 960

        # ImageNet norm
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # =========================================================
        # 2. Restoration Branch (UNet-like decoder)  - GroupNorm
        # =========================================================
        norm_type = "gn"

        self.rest_up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rest_conv1 = DoubleConv(c5_ch + c4_ch, 256, norm_type=norm_type)

        self.rest_up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rest_conv2 = DoubleConv(256 + c3_ch, 128, norm_type=norm_type)

        self.rest_up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rest_conv3 = DoubleConv(128 + c2_ch, 64, norm_type=norm_type)

        # 关键修复：r3 是 1/4 尺度（比如 96×96），要回到原图（384×384）必须 ×4
        self.rest_up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.rest_out = nn.Conv2d(64, 3, kernel_size=1)

        # =========================================================
        # 3. Segmentation Branch (FPN-like + StripPooling)
        # =========================================================
        self.strip_pool = StripPooling(c5_ch)
        self.seg_lat5 = nn.Conv2d(c5_ch, 256, 1)
        self.seg_lat4 = nn.Conv2d(c4_ch, 256, 1)
        self.seg_lat3 = nn.Conv2d(c3_ch, 256, 1)

        self.seg_conv_fuse = DoubleConv(256, 128, norm_type="bn")

        # inject restoration features (r3: 64ch -> 64ch)
        self.injection_conv = nn.Conv2d(64, 64, 1)

        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seg_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x, target_clean=None, enable_restoration=True, enable_segmentation=True):
        """
        Args:
            x: [B,3,H,W] in [0,1]
            target_clean: [B,3,H,W] in [0,1] (optional)
        Returns:
            restored_img (or None),
            seg_logits (or None),
            target_enhanced (or None)  # 用于 Stage A 的 GT 一致性监督
        """
        input_size = x.shape[-2:]

        # --- Step 0: Zero-DCE++ Enhancement ---
        if self.dce_enabled:
            with torch.no_grad():
                x_enhanced = self.dce_net(x)
                target_enhanced = self.dce_net(target_clean) if target_clean is not None else None
        else:
            # require_dce=True 时这里一般不会走到；保留是为了 stage 兼容
            x_enhanced = x
            target_enhanced = target_clean if target_clean is not None else None

        # Normalize for MobileNet
        x_norm = (x_enhanced - self.img_mean) / self.img_std

        # --- Step 1: Encoder ---
        feats = self.encoder(x_norm)
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        restored_img = None
        r3 = None

        # --- Step 2: Restoration Branch ---
        if enable_restoration:
            r1 = self.rest_conv1(torch.cat([self.rest_up1(c5), c4], dim=1))
            r2 = self.rest_conv2(torch.cat([self.rest_up2(r1), c3], dim=1))
            r3 = self.rest_conv3(torch.cat([self.rest_up3(r2), c2], dim=1))

            res_out = self.rest_out(self.rest_up4(r3))

            # 兜底：确保与输入同分辨率（未来你改 IMG_SIZE 或输入非整除时也不炸）
            if res_out.shape[-2:] != input_size:
                res_out = F.interpolate(res_out, size=input_size, mode="bilinear", align_corners=False)

            restored_img = torch.clamp(x_enhanced + res_out, 0.0, 1.0)

        # --- Step 3: Segmentation Branch ---
        seg_logits = None
        if enable_segmentation:
            s5 = self.strip_pool(c5)
            p5 = self.seg_lat5(s5)
            p4 = self.seg_lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
            p3 = self.seg_lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

            seg_base = self.seg_conv_fuse(p3)
            seg_feat = self.seg_head(seg_base)

            if enable_restoration and r3 is not None:
                rest_inject = self.injection_conv(r3)
                rest_inject = F.interpolate(rest_inject, size=seg_feat.shape[-2:], mode="bilinear", align_corners=False)
                seg_fused = seg_feat + rest_inject
            else:
                seg_fused = seg_feat

            seg_logits = self.seg_final(seg_fused)

            if seg_logits.shape[-2:] != input_size:
                seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)

        return restored_img, seg_logits, target_enhanced
