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
            nn.Sigmoid(),
        )

    def forward(self, depth_feat, rgb_feat):
        rgb_attn = self.rgb_pool(rgb_feat)
        scale = self.rgb_fc(rgb_attn)
        return depth_feat * scale


class DualBranchRGBDNet(nn.Module):
    """
    Dual-branch RGBD network with attention from RGB guiding the Depth branch.
    """

    def __init__(self, cfg, num_classes=2, pretrained=True):
        super(DualBranchRGBDNet, self).__init__()

        # Load config
        self.cfg = cfg

        # RGB MobileNetV2
        rgb_model = models.mobilenet_v2(pretrained=pretrained).features
        self.rgb_base = rgb_model

        # Depth MobileNetV2 (adapted for 1-channel input)
        depth_model = models.mobilenet_v2(pretrained=pretrained)
        depth_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.depth_base = depth_model.features

        # Use GuidedSEBlock if enabled
        self.attention = cfg.model.attention
        self.guided_attn = GuidedSEBlock(channels=1280) if self.attention else None

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
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
