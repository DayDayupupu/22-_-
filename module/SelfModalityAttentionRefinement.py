import torch
import torch.nn as nn
from module.BaseBlock import BaseConv2d


class ChannelAttention(nn.Module):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    """
    The implementation of spatial attention mechanism.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        weight_map = self.sigmoid(x)
        return weight_map

class SelfModalityAttentionRefinement(nn.Module):
    def __init__(self):
        """
        Self-Modality Attention Refinement Module

        Args:
            in_channels: 输入特征的通道数
            out_channels: 输出特征的通道数
        """
        super(SelfModalityAttentionRefinement, self).__init__()
        # 空间注意力模块
        # self-modality attention refinement
        self.ca_rgb = ChannelAttention(512)
        self.ca_depth = ChannelAttention(512)
        self.ca_rgbd = ChannelAttention(512)

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)
        self.sa_rgbd = SpatialAttention(kernel_size=7)
        self.conv_rgb = BaseConv2d(512, 512, kernel_size = 3, padding=1)
        self.conv_depth = BaseConv2d(512,512, kernel_size = 3, padding=1)
        self.conv_rgbd = BaseConv2d(512, 512, kernel_size = 3, padding=1)
        # cross-modality weighting refinement


    def forward(self, conv5_r, conv5_d, conv5_rgbd):
        """
        Args:
            conv5_r: RGB 模态输入特征 [B, C, H, W]
            conv5_d: 深度模态输入特征 [B, C, H, W]
            conv5_rgbd: RGB-D 模态输入特征 [B, C, H, W]
        Returns:
            rgb_smAR: 优化后的 RGB 特征
            depth_smAR: 优化后的深度特征
            rgbd_smAR: 优化后的 RGB-D 特征
        """
        B, C, H, W = conv5_r.size()
        P = H * W

        # 1. 计算空间注意力
        rgb_SA = self.sa_rgb(conv5_r).view(B, -1, P)  # [B, 1, P]
        depth_SA = self.sa_depth(conv5_d).view(B, -1, P)  # [B, 1, P]
        rgbd_SA = self.sa_rgbd(conv5_rgbd).view(B, -1, P)  # [B, 1, P]

        # 2. 计算通道注意力
        rgb_CA = self.ca_rgb(conv5_r).view(B, C, -1)  # [B, C, 1]
        depth_CA = self.ca_depth(conv5_d).view(B, C, -1)  # [B, C, 1]
        rgbd_CA = self.ca_rgbd(conv5_rgbd).view(B, C, -1)  # [B, C, 1]

        # 3. 融合通道注意力和空间注意力
        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W)  # RGB 模态权重 [B, C, H, W]
        depth_M = torch.bmm(depth_CA, depth_SA).view(B, C, H, W)  # Depth 模态权重 [B, C, H, W]
        rgbd_M = torch.bmm(rgbd_CA, rgbd_SA).view(B, C, H, W)  # RGBD 模态权重 [B, C, H, W]

        # 4. 特征优化（使用注意力权重）
        rgb_smAR = conv5_r * rgb_M + conv5_r  # 残差连接
        depth_smAR = conv5_d * depth_M + conv5_d
        rgbd_smAR = conv5_rgbd * rgbd_M + conv5_rgbd

        # 5. 优化后的特征进一步处理
        rgb_smAR = self.conv_rgb(rgb_smAR)  # [B, out_channels, H, W]
        depth_smAR = self.conv_depth(depth_smAR)  # [B, out_channels, H, W]
        rgbd_smAR = self.conv_rgbd(rgbd_smAR)  # [B, out_channels, H, W]

        return rgb_smAR, depth_smAR, rgbd_smAR


# 测试模块
if __name__ == "__main__":

    B, C, H, W = 16, 512, 11, 11
    conv5_r = torch.randn(B, C, H, W)
    conv5_d = torch.randn(B, C, H, W)
    conv5_rgbd = torch.randn(B, C, H, W)

    model = SelfModalityAttentionRefinement(in_channels=C, out_channels=128)
    rgb_smAR, depth_smAR, rgbd_smAR = model(conv5_r, conv5_d, conv5_rgbd)

    print(f"RGB Feature Shape: {rgb_smAR.shape}")
    print(f"Depth Feature Shape: {depth_smAR.shape}")
    print(f"RGBD Feature Shape: {rgbd_smAR.shape}")
