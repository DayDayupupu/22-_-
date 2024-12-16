import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=128, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.attend = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape  # B: batch size, N: num of tokens, D: embedding dimension
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into query, key, value
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        attn = self.attend(torch.einsum("bhid,bhjd->bhij", q, k) * self.scale)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Add & Norm
        x = self.norm1(x + self.dropout(out))
        # Feedforward & Norm
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x

class cmWR_Transformer(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=2, embed_dim=64, num_heads=8, mlp_dim=128):
        """
        The implementation of cross-modality weighting refinement using Transformer

        Args:
            in_channels: Number of channels for the three inputs
            squeeze_ratio: Ratio to compute intermediate channels
            embed_dim: Embedding dimension for Transformer
            num_heads: Number of attention heads in Transformer
            mlp_dim: Dimension of the feedforward network in Transformer
        """
        super(cmWR_Transformer, self).__init__()
        inter_channels = in_channels // squeeze_ratio

        self.conv_r = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_d = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_rd = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        self.transformer = TransformerBlock(dim=inter_channels, heads=num_heads, dim_head=embed_dim // num_heads, mlp_dim=mlp_dim)
        self.channel_project = nn.Conv2d(inter_channels, in_channels, kernel_size=1)  # Project back to in_channels

    def forward(self, rgb, depth, rgbd):
        B, C, H, W = rgb.size()
        P = H * W

        # Linear embeddings
        rgb_emb = self.conv_r(rgb).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C']
        depth_emb = self.conv_d(depth).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C']
        rgbd_emb = self.conv_rd(rgbd).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C']

        # Concatenate embeddings
        combined_emb = torch.cat([rgb_emb, depth_emb, rgbd_emb], dim=1)  # [B, 3HW, C']

        # Apply Transformer
        refined_emb = self.transformer(combined_emb)  # [B, 3HW, C']

        # Split refined embeddings
        refined_rgb, refined_depth, refined_rgbd = torch.split(refined_emb, P, dim=1)

        # Reshape back to image dimensions
        refined_rgb = refined_rgb.permute(0, 2, 1).view(B, C // 2, H, W)
        refined_depth = refined_depth.permute(0, 2, 1).view(B, C // 2, H, W)
        refined_rgbd = refined_rgbd.permute(0, 2, 1).view(B, C // 2, H, W)

        # Project back to the original channel dimensions
        refined_rgb = self.channel_project(refined_rgb)
        refined_depth = self.channel_project(refined_depth)
        refined_rgbd = self.channel_project(refined_rgbd)

        # Add original inputs
        rgb_final = rgb + refined_rgb
        depth_final = depth + refined_depth
        rgbd_final = rgbd + refined_rgbd

        return rgb_final, depth_final, rgbd_final


# Test the model
B, C, H, W = 16, 512, 11, 11
rgb = torch.randn(B, C, H, W)
depth = torch.randn(B, C, H, W)
rgbd = torch.randn(B, C, H, W)

model = cmWR_Transformer(in_channels=C)
rgb_out, depth_out, rgbd_out = model(rgb, depth, rgbd)

print(f"rgb_out shape: {rgb_out.shape}, depth_out shape: {depth_out.shape}, rgbd_out shape: {rgbd_out.shape}")
