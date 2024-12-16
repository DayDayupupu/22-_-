import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class cmWR_CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        CBAM (Convolutional Block Attention Module) for cross-modality weighting refinement

        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(cmWR_CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, rgb, depth, rgbd):
        """
        Args:
            rgb: RGB feature map
            depth: Depth feature map
            rgbd: RGBD fused feature map

        Returns:
            rgb_final: Refined RGB feature map
            depth_final: Refined Depth feature map
            rgbd_final: Refined RGBD feature map
        """
        # Apply CBAM on RGB
        rgb = self.channel_attention(rgb) * rgb
        rgb = self.spatial_attention(rgb) * rgb

        # Apply CBAM on Depth
        depth = self.channel_attention(depth) * depth
        depth = self.spatial_attention(depth) * depth

        # Apply CBAM on RGBD
        rgbd = self.channel_attention(rgbd) * rgbd
        rgbd = self.spatial_attention(rgbd) * rgbd

        return rgb, depth, rgbd


# 测试 CBAM 模块
B, C, H, W = 4, 64, 32, 32
rgb = torch.randn(B, C, H, W)
depth = torch.randn(B, C, H, W)
rgbd = torch.randn(B, C, H, W)

model = cmWR_CBAM(in_channels=C)
rgb_out, depth_out, rgbd_out = model(rgb, depth, rgbd)

print(f"rgb_out shape: {rgb_out.shape}, depth_out shape: {depth_out.shape}, rgbd_out shape: {rgbd_out.shape}")
