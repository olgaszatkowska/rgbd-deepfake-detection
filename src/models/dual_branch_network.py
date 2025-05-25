import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from typing import Any

from models.dehydrate import dehydrate_classifier_head


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


class DualBranchRGBDNet(nn.Module):
    """
    Dual-branch RGBD network with attention from RGB guiding the Depth branch.
    """

    def __init__(self, cfg: Any, num_classes: int = 2, pretrained: bool = True) -> None:
        super(DualBranchRGBDNet, self).__init__()

        # Load config
        self.cfg = cfg

        # RGB MobileNetV2
        rgb_model = models.mobilenet_v2(pretrained=pretrained).features
        self.rgb_base = rgb_model

        # Depth MobileNetV2 (adapted for 1-channel input)
        depth_model = models.mobilenet_v2(pretrained=pretrained)
        depth_model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.depth_base = depth_model.features

        # Use GuidedSEBlock if enabled
        self.use_rgb_guided_attention: bool = cfg.model.use_rgb_guided_attention
        self.use_bidirectional_attention: bool = cfg.model.use_bidirectional_attention

        use_rgb_guided_block = (
            self.use_rgb_guided_attention or self.use_bidirectional_attention
        )
        uses_depth_guided_block = self.use_bidirectional_attention

        self.rgb_guides_depth: nn.Module | None = (
            GuidedSEBlock(channels=1280) if use_rgb_guided_block else None
        )
        self.depth_guides_rgb: nn.Module | None = (
            GuidedSEBlock(channels=1280) if uses_depth_guided_block else None
        )

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = dehydrate_classifier_head(cfg, num_classes)

        if self.cfg.model.init_weights_method == "kaiming":
            self._init_kaiming_weights()

    def _init_kaiming_weights(self) -> None:
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # Extract features
        rgb_feat = self.rgb_base(rgb)
        depth_feat = self.depth_base(depth)

        if self.use_bidirectional_attention:
            # Use original (unmodulated) features to avoid feedback loop
            depth_feat = self.rgb_guides_depth(depth_feat, rgb_feat.detach())
            rgb_feat = self.depth_guides_rgb(rgb_feat, depth_feat.detach())
        elif self.use_rgb_guided_attention:
            depth_feat = self.rgb_guides_depth(depth_feat, rgb_feat)
        
        if return_features:
            return rgb_feat, depth_feat

        # Global pooling
        rgb_feat = self.pool(rgb_feat)
        depth_feat = self.pool(depth_feat)

        # Fuse and classify
        fused = torch.cat((rgb_feat, depth_feat), dim=1)  # [B, 1024, 1, 1]
        return self.classifier(fused)
