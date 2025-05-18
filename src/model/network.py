import torch
import torch.nn as nn
import torchvision.models as models


class GuidedSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block with external guidance.
    Uses features from a guidance branch (e.g. RGB) to modulate another (e.g. Depth).
    """
    def __init__(self, channels, reduction=16):
        super(GuidedSEBlock, self).__init__()
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.rgb_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, depth_feat, rgb_feat):
        # Generate attention from RGB features
        rgb_attn = self.rgb_pool(rgb_feat)       # [B, C, 1, 1]
        scale = self.rgb_fc(rgb_attn)            # [B, C, 1, 1]

        # Apply to depth features
        return depth_feat * scale


class RGBDNet(nn.Module):
    """
    Dual-branch RGBD network with attention from RGB guiding the Depth branch.
    """
    def __init__(self, cfg, num_classes=2, pretrained=True):
        super(RGBDNet, self).__init__()

        # Load config
        self.cfg = cfg

        # RGB backbone
        rgb_backbone = models.resnet18(pretrained=pretrained)
        self.rgb_base = nn.Sequential(*list(rgb_backbone.children())[:-2])  # Remove avgpool and fc

        # Depth backbone
        depth_backbone = models.resnet18(pretrained=pretrained)
        depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_base = nn.Sequential(*list(depth_backbone.children())[:-2])

        # Guided attention module: RGB guides Depth
        self.guided_attn = GuidedSEBlock(channels=512)

        # Fusion + classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # Extract features
        rgb_feat = self.rgb_base(rgb)
        depth_feat = self.depth_base(depth)

        # Use RGB features to guide attention in Depth features
        if self.cfg.model.attention:
            depth_feat = self.guided_attn(depth_feat, rgb_feat)

        # Global pooling
        rgb_feat = self.pool(rgb_feat)
        depth_feat = self.pool(depth_feat)

        # Fuse and classify
        fused = torch.cat((rgb_feat, depth_feat), dim=1)  # [B, 1024, 1, 1]
        return self.classifier(fused)
