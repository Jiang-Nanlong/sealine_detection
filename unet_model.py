import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from zerodce import C_DCE_Net


# ============ 小工具：尽量兼容不同 torch 版本的 torch.load ============
def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class StripPooling(nn.Module):
    """
    轻量 StripPooling：增强长条结构上下文（海天线/细长雾带等）
    """
    def __init__(self, channels, mid=None):
        super().__init__()
        mid = mid or max(8, channels // 4)
        self.reduce = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, mid, (3, 1), padding=(1, 0), bias=False)
        self.conv_w = nn.Conv2d(mid, mid, (1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.expand = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        y = F.relu(self.bn1(self.reduce(x)), inplace=True)

        y_h = F.adaptive_avg_pool2d(y, (h, 1))
        y_h = self.conv_h(y_h)
        y_h = F.interpolate(y_h, size=(h, w), mode="bilinear", align_corners=False)

        y_w = F.adaptive_avg_pool2d(y, (1, w))
        y_w = self.conv_w(y_w)
        y_w = F.interpolate(y_w, size=(h, w), mode="bilinear", align_corners=False)

        y = y_h + y_w
        y = F.relu(self.bn2(y), inplace=True)
        y = self.bn3(self.expand(y))
        return F.relu(x + y, inplace=True)


class RestorationGuidedHorizonNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        dce_weights_path="Epoch99.pth",
        dce_scale_factor=1,
        require_dce=True,
    ):
        super().__init__()
        self.require_dce = bool(require_dce)

        # ------------------ 1) DCE（必须成功加载） ------------------
        self.dce_net = C_DCE_Net(scale_factor=dce_scale_factor, n=32, enhance_iters=8)

        if dce_weights_path is not None:
            if not os.path.isabs(dce_weights_path):
                cand = os.path.join(os.path.dirname(__file__), dce_weights_path)
                if os.path.exists(cand):
                    dce_weights_path = cand

            if not os.path.exists(dce_weights_path):
                raise FileNotFoundError(
                    f"[RG-HNet] 找不到 Zero-DCE++ 权重文件: {dce_weights_path}\n"
                    f"请把 Epoch99.pth 放到 unet_model.py 同目录，或传入绝对路径。"
                )

            state = _safe_torch_load(dce_weights_path)
            # 有些 checkpoint 会包一层
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]

            try:
                self.dce_net.load_state_dict(state, strict=True)
            except Exception as e:
                raise RuntimeError(
                    "[RG-HNet] Zero-DCE++ 权重加载失败（结构不匹配或权重非本模型）。\n"
                    f"  权重文件: {dce_weights_path}\n"
                    f"  原始错误: {repr(e)}"
                )

            # Stage A 里 DCE 固定，不参与训练（你想要“只做亮度增强”，一般就冻结）
            self.dce_net.eval()
            for p in self.dce_net.parameters():
                p.requires_grad = False
        else:
            if self.require_dce:
                raise RuntimeError("[RG-HNet] require_dce=True 但 dce_weights_path=None")
            self.dce_net = None

        # ------------------ 2) Encoder: MobileNetV3-Large ------------------
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        return_nodes = {
            "features.3": "c2",   # 1/4
            "features.6": "c3",   # 1/8
            "features.12": "c4",  # 1/16
            "features.16": "c5",  # 1/32
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        c2_ch, c3_ch, c4_ch, c5_ch = 24, 40, 112, 960

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ------------------ 3) Restoration Branch (Stage A 用) ------------------
        r_ch = 64
        self.rest_lat2 = nn.Conv2d(c2_ch, r_ch, 1)
        self.rest_lat3 = nn.Conv2d(c3_ch, r_ch, 1)
        self.rest_lat4 = nn.Conv2d(c4_ch, r_ch, 1)
        self.rest_lat5 = nn.Conv2d(c5_ch, r_ch, 1)

        self.rest_fuse = DoubleConv(r_ch, r_ch)
        self.rest_strip = StripPooling(r_ch)
        self.rest_out = nn.Conv2d(r_ch, 3, 1)  # 输出 residual (RGB)

        # ------------------ 4) Segmentation Branch（Stage A 可关掉） ------------------
        s_ch = 128
        self.seg_lat3 = nn.Conv2d(c3_ch, s_ch, 1)
        self.seg_lat4 = nn.Conv2d(c4_ch, s_ch, 1)
        self.seg_lat5 = nn.Conv2d(c5_ch, s_ch, 1)

        self.seg_fuse = DoubleConv(s_ch, s_ch)
        self.seg_strip = StripPooling(s_ch)
        self.seg_final = nn.Conv2d(s_ch, num_classes, 1)

        # restoration -> segmentation 的注入（你后面 stage B 可能用到）
        self.inject = nn.Conv2d(r_ch, s_ch, 1)

    @torch.no_grad()
    def _dce_enhance(self, x: torch.Tensor) -> torch.Tensor:
        if self.dce_net is None:
            return x
        return self.dce_net(x)

    def forward(self, x, target=None, enable_restoration=True, enable_segmentation=True):
        """
        返回：
          restored_img: [B,3,H,W]
          seg_logits:   [B,C,H,W] 或 None
          target_dce:   DCE(target)（Stage A 用它做监督；若 target=None 则返回 None）
        """
        input_size = x.shape[-2:]

        # -------- DCE 增强（必须有） --------
        if self.dce_net is None:
            if self.require_dce:
                raise RuntimeError("[RG-HNet] DCE 被禁用，但你 require_dce=True（Stage A 必须依赖 DCE）。")
            x_enh = x
            target_dce = target
        else:
            x_enh = self._dce_enhance(x)
            target_dce = self._dce_enhance(target) if target is not None else None

        # -------- Encoder（对增强图做 ImageNet normalize）--------
        enc_in = (x_enh - self.img_mean) / (self.img_std + 1e-6)
        feats = self.encoder(enc_in)
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        # -------- Restoration Branch --------
        if enable_restoration:
            p5 = self.rest_lat5(c5)
            p4 = self.rest_lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
            p3 = self.rest_lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
            p2 = self.rest_lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

            r = self.rest_strip(self.rest_fuse(p2))
            residual = self.rest_out(r)

            # 关键：强制对齐到原图尺寸，避免任何 384 vs 192 的 mismatch
            if residual.shape[-2:] != input_size:
                residual = F.interpolate(residual, size=input_size, mode="bilinear", align_corners=False)

            restored_img = torch.clamp(x_enh + residual, 0.0, 1.0)
        else:
            restored_img = x_enh

        # -------- Segmentation Branch --------
        seg_logits = None
        if enable_segmentation:
            q5 = self.seg_lat5(c5)
            q4 = self.seg_lat4(c4) + F.interpolate(q5, size=c4.shape[-2:], mode="nearest")
            q3 = self.seg_lat3(c3) + F.interpolate(q4, size=c3.shape[-2:], mode="nearest")

            s = self.seg_strip(self.seg_fuse(q3))

            # 注入 restoration 的信息（用 p2/r 的尺度对齐）
            if enable_restoration:
                inj = self.inject(r)
                inj = F.interpolate(inj, size=s.shape[-2:], mode="bilinear", align_corners=False)
                s = s + inj

            seg_logits = self.seg_final(s)
            if seg_logits.shape[-2:] != input_size:
                seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)

        return restored_img, seg_logits, target_dce
