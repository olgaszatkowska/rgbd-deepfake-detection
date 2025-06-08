import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GuidedCBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(GuidedCBAM, self).__init__()

        # Channel attention (guided by RGB)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention (applied to depth)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, depth_feat, rgb_feat):
        # Channel attention guided by RGB
        rgb_pool = self.channel_attn(rgb_feat)        # [B, C, 1, 1]
        depth_feat = depth_feat * rgb_pool            # Apply to depth

        # Spatial attention on updated depth features
        avg_pool = torch.mean(depth_feat, dim=1, keepdim=True)
        max_pool, _ = torch.max(depth_feat, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_pool, max_pool], dim=1)
        scale = self.spatial_attn(spatial_map)        # [B, 1, H, W]

        return depth_feat * scale


class GuidedSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block with external guidance.
    Uses features from a guidance branch RGB to modulate another DEPTH
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super(GuidedSEBlock, self).__init__()
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.rgb_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, depth_feat: Tensor, rgb_feat: Tensor) -> Tensor:
        rgb_attn = self.rgb_pool(rgb_feat)
        scale = self.rgb_fc(rgb_attn)
        return depth_feat * scale
