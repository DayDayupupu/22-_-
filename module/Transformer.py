import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, squeeze_ratio=2):
        """
        Transformer-based Cross-Modality Attention

        Args:
            in_channels: Number of input channels for RGB, Depth, and RGBD
            num_heads: Number of attention heads
            squeeze_ratio: Channel reduction ratio
        """
        super(TransformerAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.inter_channels = in_channels // squeeze_ratio

        # Linear transformations for Query, Key, Value
        self.query_rgb = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_depth = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_rgbd = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # Output projection layer
        self.out_proj = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # Multi-head Attention configuration
        self.num_heads = num_heads
        assert self.inter_channels % num_heads == 0, "Inter_channels must be divisible by num_heads"
        self.head_dim = self.inter_channels // num_heads

    def forward(self, rgb, depth, rgbd):
        """
        Args:
            rgb: RGB feature map [B, C, H, W]
            depth: Depth feature map [B, C, H, W]
            rgbd: RGBD fused feature map [B, C, H, W]
        Returns:
            rgb_final, depth_final, rgbd_final: Refined features
        """
        B, C, H, W = rgb.size()
        P = H * W

        # Generate Query, Key, Value
        Q = self.query_rgb(rgb).view(B, self.num_heads, self.head_dim, P).permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]
        K = self.key_depth(depth).view(B, self.num_heads, self.head_dim, P)                   # [B, num_heads, head_dim, HW]
        V = self.value_rgbd(rgbd).view(B, self.num_heads, self.head_dim, P).permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]

        # Attention mechanism
        attn_weights = F.softmax(torch.matmul(Q, K) / (self.head_dim ** 0.5), dim=-1)  # [B, num_heads, HW, HW]
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, HW, head_dim]
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(B, self.inter_channels, H, W)

        # Project back to the original channel size
        refined_features = self.out_proj(attn_output)  # [B, C, H, W]

        # Residual connections
        rgb_final = rgb + refined_features
        depth_final = depth + refined_features
        rgbd_final = rgbd + refined_features

        return rgb_final, depth_final, rgbd_final


# 测试模块
B, C, H, W = 4, 64, 32, 32
rgb = torch.randn(B, C, H, W)
depth = torch.randn(B, C, H, W)
rgbd = torch.randn(B, C, H, W)

# 构建 Transformer Attention 模块
model = TransformerAttention(in_channels=C, num_heads=4, squeeze_ratio=2)
rgb_out, depth_out, rgbd_out = model(rgb, depth, rgbd)

print(f"rgb_out shape: {rgb_out.shape}, depth_out shape: {depth_out.shape}, rgbd_out shape: {rgbd_out.shape}")
