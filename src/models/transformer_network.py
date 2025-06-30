import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from typing import Any

from models.dehydrate import dehydrate_classifier_head
from models.attention import GuidedCBAM


class TransformerRGDBNet(nn.Module):
    def __init__(self, cfg: Any, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg

        # Load weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        # RGB and Depth backbones
        self.rgb_backbone = models.resnet18(weights=weights)
        self.depth_backbone = models.resnet18(weights=weights)
        self.depth_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.rgb_base = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_base = nn.Sequential(*list(self.depth_backbone.children())[:-2])

        self.token_dim = 512

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation="relu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_dim))

        # Attention
        self.use_rgb_guided_attention: bool = cfg.model.use_rgb_guided_attention
        self.use_bidirectional_attention: bool = cfg.model.use_bidirectional_attention
        use_rgb_guided_block = self.use_rgb_guided_attention or self.use_bidirectional_attention
        uses_depth_guided_block = self.use_bidirectional_attention

        self.rgb_guides_depth = GuidedCBAM(channels=512) if use_rgb_guided_block else None
        self.depth_guides_rgb = GuidedCBAM(channels=512) if uses_depth_guided_block else None

        # Modulation weights
        self.rgb_to_depth_weight = nn.Parameter(torch.tensor(0.2))
        self.depth_to_rgb_weight = nn.Parameter(torch.tensor(0.1)) if uses_depth_guided_block else None

        # Classifier
        self.classifier = dehydrate_classifier_head(cfg=cfg, num_classes=num_classes)
        self.depth_aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2)  # Use actual num_classes if needed
        )

        if self.cfg.model.init_weights_method == "kaiming":
            self._init_kaiming_weights()

        self.pool = nn.AdaptiveAvgPool2d(1)

    def _init_kaiming_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False, return_cls_token=False, return_fused_spatial=False):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        rgb_feat = self.rgb_base(rgb)   # shape [B, 512, H, W]
        depth_feat = self.depth_base(depth)  # same shape

        if self.training and torch.rand(1).item() < self.cfg.model.drop_rgb_prob:
            rgb = torch.zeros_like(rgb)

        if self.use_bidirectional_attention:
            modulated_depth = self.rgb_guides_depth(depth_feat, rgb_feat.detach())
            modulated_rgb = self.depth_guides_rgb(rgb_feat, depth_feat.detach())
            depth_feat = depth_feat + self.rgb_to_depth_weight * modulated_depth
            rgb_feat = rgb_feat + self.depth_to_rgb_weight * modulated_rgb
        elif self.use_rgb_guided_attention:
            modulated_depth = self.rgb_guides_depth(depth_feat, rgb_feat)
            depth_feat = depth_feat + self.rgb_to_depth_weight * modulated_depth

        if return_fused_spatial:
            fused_spatial = torch.cat([rgb_feat, depth_feat], dim=1)  # shape [B, 1024, H, W]
            return fused_spatial

        if return_features:
            return rgb_feat, depth_feat

        # Flatten spatial dimensions into tokens
        rgb_tokens = rearrange(rgb_feat, "b c h w -> b (h w) c")
        depth_tokens = rearrange(depth_feat, "b c h w -> b (h w) c")

        # Prepend CLS token
        tokens = torch.cat([rgb_tokens, depth_tokens], dim=1)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Transformer encoding
        encoded = self.transformer(tokens)
        cls_output = encoded[:, 0]

        # Auxiliary classification from depth
        pooled_depth = self.pool(depth_feat)
        depth_logits_aux = self.depth_aux_classifier(pooled_depth)

        return self.classifier(cls_output), depth_logits_aux
