import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise separable conv with exact submodule names:
    depth_conv + point_conv  (to match checkpoint keys)
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=True
        )
        self.point_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class C_DCE_Net(nn.Module):
    """
    Depthwise-separable Zero-DCE++ style network.
    Output is typically 24 channels (8 curves * 3 RGB).
    This forward applies the curve enhancement internally and returns enhanced RGB image.
    """
    def __init__(self, base_ch=32, out_ch=24):
        super().__init__()
        self.e_conv1 = DSConv(3, base_ch)
        self.e_conv2 = DSConv(base_ch, base_ch)
        self.e_conv3 = DSConv(base_ch, base_ch)
        self.e_conv4 = DSConv(base_ch, base_ch)
        self.e_conv5 = DSConv(base_ch * 2, base_ch)
        self.e_conv6 = DSConv(base_ch * 2, base_ch)
        self.e_conv7 = DSConv(base_ch * 2, out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    @staticmethod
    def enhance(img, r):
        # img: [B,3,H,W] in [0,1], r: [B,3,H,W]
        return img + r * (img * img - img)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x7 = self.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # typical: 24 channels -> 8 curves
        if x7.shape[1] == 24:
            rs = torch.split(x7, 3, dim=1)
            out = x
            for r in rs:
                out = self.enhance(out, r)
            return torch.clamp(out, 0.0, 1.0)

        # fallback: if some weight outputs 3 directly
        if x7.shape[1] == 3:
            return torch.clamp(x + x7, 0.0, 1.0)

        raise RuntimeError(f"Unexpected DCE output channels: {x7.shape[1]}")
