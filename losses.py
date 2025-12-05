# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorConstancyLoss(nn.Module):
    """色彩恒常性损失 L_col"""

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced_image):
        mean_rgb = torch.mean(enhanced_image, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        color_loss = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return torch.mean(color_loss)


class ExposureLoss(nn.Module):
    """曝光损失 L_exp"""

    def __init__(self, patch_size=16, E=0.6):
        super(ExposureLoss, self).__init__()
        self.patch_size = patch_size
        self.E = E  # 理想的平均曝光水平
        self.pool = nn.AvgPool2d(self.patch_size)

    def forward(self, enhanced_image):
        # 计算每个patch的平均亮度
        mean_intensity = self.pool(torch.mean(enhanced_image, 1, keepdim=True))
        # 计算与理想曝光水平E的L2距离
        exposure_loss = torch.mean(torch.pow(mean_intensity - self.E, 2))
        return exposure_loss


class IlluminationSmoothnessLoss(nn.Module):
    """光照平滑度损失 L_tvA"""

    def __init__(self, tv_loss_weight=1.0):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, A):  # A是Zero-DCE网络输出的曲线参数图
        batch_size = A.size()[0]
        h_x = A.size()[2]
        w_x = A.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class ZeroDCELoss(nn.Module):
    """
    将Zero-DCE的多个无参考损失组合起来
    """

    def __init__(self, w_col=0.5, w_exp=10.0, w_tvA=200.0):
        super(ZeroDCELoss, self).__init__()
        self.w_col = w_col
        self.w_exp = w_exp
        self.w_tvA = w_tvA
        self.color_loss = ColorConstancyLoss()
        self.exposure_loss = ExposureLoss()
        self.smoothness_loss = IlluminationSmoothnessLoss()

    def forward(self, enhanced_image, A):
        l_col = self.color_loss(enhanced_image)
        l_exp = self.exposure_loss(enhanced_image)
        l_tvA = self.smoothness_loss(A)

        total_loss = self.w_col * l_col + self.w_exp * l_exp + self.w_tvA * l_tvA
        return total_loss