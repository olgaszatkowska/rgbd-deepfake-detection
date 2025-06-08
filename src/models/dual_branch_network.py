import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from typing import Any

from models.dehydrate import dehydrate_classifier_head
from models.attention import GuidedCBAM


class DualBranchRGBDNet(nn.Module):
    """
    Dual-branch RGBD network with attention from RGB guiding the Depth branch.
    """

    def __init__(self, cfg: Any, num_classes: int = 2, pretrained: bool = True) -> None:
        super(DualBranchRGBDNet, self).__init__()

        # Load config
        self.cfg = cfg

        # RGB MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        rgb_model = models.mobilenet_v2(weights=weights).features

        self.rgb_base = rgb_model

        # Depth MobileNetV2 (adapted for 1-channel input)
        depth_model = models.mobilenet_v2(weights=weights)
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

        self.rgb_guides_depth = GuidedCBAM(channels=1280) if use_rgb_guided_block else None
        self.depth_guides_rgb = GuidedCBAM(channels=1280) if uses_depth_guided_block else None

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = dehydrate_classifier_head(cfg, num_classes)
        self.depth_aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 2)  # 2 = num_classes
        )

        # Learnable modulation weights
        self.rgb_to_depth_weight = nn.Parameter(torch.tensor(0.2))
        self.depth_to_rgb_weight = nn.Parameter(torch.tensor(0.1)) if self.use_bidirectional_attention else None

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

        if self.training:
            if torch.rand(1).item() < self.cfg.model.drop_rgb_prob:
                rgb = torch.zeros_like(rgb)

        if self.use_bidirectional_attention:
            # Use original (unmodulated) features to avoid feedback loop
            modulated_depth = self.rgb_guides_depth(depth_feat, rgb_feat.detach())
            modulated_rgb = self.depth_guides_rgb(rgb_feat, depth_feat.detach())

            depth_feat = depth_feat + self.rgb_to_depth_weight * modulated_depth
            rgb_feat = rgb_feat + self.depth_to_rgb_weight * modulated_rgb

        elif self.use_rgb_guided_attention:
            modulated_depth = self.rgb_guides_depth(depth_feat, rgb_feat)
            depth_feat = depth_feat + self.rgb_to_depth_weight * modulated_depth

        if return_features:
            return rgb_feat, depth_feat

        # Global pooling
        rgb_feat = self.pool(rgb_feat)
        depth_feat = self.pool(depth_feat)

        # Auxiliary logits before fusion
        depth_logits_aux = self.depth_aux_classifier(depth_feat)

        # Fusion
        fused = torch.cat((rgb_feat, depth_feat), dim=1)
        fused_logits = self.classifier(fused)

        return fused_logits, depth_logits_aux
