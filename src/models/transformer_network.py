import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

from models.dehydrate import dehydrate_classifier_head


class TransformerRGDBNet(nn.Module):
    def __init__(self, cfg, num_classes=2, pretrained=True):
        super().__init__()
        self.cfg = cfg

        # RGB and Depth feature extractors (ResNet18)
        # RGB and Depth feature extractors (ResNet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.rgb_backbone = models.resnet18(weights=weights)
        self.depth_backbone = models.resnet18(weights=weights)
        self.depth_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Truncate before avgpool and fc
        self.rgb_base = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_base = nn.Sequential(*list(self.depth_backbone.children())[:-2])

        self.token_dim = 512  # Final feature map has 512 channels

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_dim))

        # Final classification head
        self.classifier = dehydrate_classifier_head(cfg=cfg, num_classes=num_classes)

        if self.cfg.model.init_weights_method == "kaiming":
            self._init_kaiming_weights()

    def _init_kaiming_weights(self) -> None:
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # Extract feature maps
        rgb_feat = self.rgb_base(rgb)  # [B, 512, H, W]
        depth_feat = self.depth_base(depth)  # [B, 512, H, W]

        # Flatten spatial dimensions to tokens: [B, C, H, W] -> [B, HW, C]
        rgb_tokens = rearrange(rgb_feat, "b c h w -> b (h w) c")
        depth_tokens = rearrange(depth_feat, "b c h w -> b (h w) c")

        # Concatenate and prepend [CLS] token
        tokens = torch.cat([rgb_tokens, depth_tokens], dim=1)  # [B, 2*HW, C]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, C]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 1 + 2*HW, C]

        # Transformer encoding
        encoded = self.transformer(tokens)
        cls_output = encoded[:, 0]  # [B, C]

        return self.classifier(cls_output)
