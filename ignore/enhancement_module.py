import torch
import torch.nn as nn

# 从本地文件导入 ZeroDCE 和 DenoiseUNet
from zerodce_plusplus import ZeroDCE
from denoise_unet import DenoiseUNet


class EnhancementModule(nn.Module):
    """
    将 Zero-DCE 和 U-Net Denoise 串联起来的完整增强模块.
    """

    def __init__(self, n_iter_dce=8, pretrained_unet_encoder=True):
        super(EnhancementModule, self).__init__()
        print("Initializing Enhancement Module...")
        self.low_light_enhancer = ZeroDCE(n_iter=n_iter_dce)
        print("-> Zero-DCE++ part initialized.")
        self.denoiser = DenoiseUNet(pretrained_encoder=pretrained_unet_encoder)
        print("-> Denoise U-Net (MobileNetV3) part initialized.")
        print("Module ready.")

    # def forward(self, x):
    #     # 第一步：低光照增强
    #     enhanced_img = self.low_light_enhancer(x)
    #
    #     # 第二步：去噪/去伪影 (例如雨痕)
    #     # Zero-DCE的输出可能不在[0,1]范围，可以加一个clamp
    #     enhanced_img_clamped = torch.clamp(enhanced_img, 0, 1)
    #     denoised_img = self.denoiser(enhanced_img_clamped)
    #
    #     return denoised_img

    def forward(self, x):
        # 第一步：低光照增强，并接收所有返回值
        intermediate_enhanced, A_maps = self.low_light_enhancer(x)

        # 第二步：去噪/去伪影 (例如雨痕)
        # 对中间结果进行clamp，准备送入去噪器
        intermediate_clamped = torch.clamp(intermediate_enhanced, 0, 1)
        final_output = self.denoiser(intermediate_clamped)

        # 返回所有需要的值，以支持微调阶段的复合损失计算
        return final_output, intermediate_clamped, A_maps