from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim import Optimizer, lr_scheduler


def dehydrate_model(cfg: DictConfig) -> nn.Module:
    from models import DualBranchRGBDNet, TransformerRGDBNet

    model_type = cfg.model.type

    if model_type == "transformer":
        return TransformerRGDBNet(cfg)

    if model_type == "dual_branch":
        return DualBranchRGBDNet(cfg)

    raise Exception("Unknown network type")


def dehydrate_scheduler_config(
    cfg: DictConfig, optimizer: Optimizer
) -> dict[str, lr_scheduler.LRScheduler | dict[str, lr_scheduler.LRScheduler | str]]:
    scheduler_type = cfg.training.scheduler.type

    if scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.scheduler.cosine_T_max
        )

        return {"lr_scheduler": scheduler}

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.training.scheduler.decay_factor,
            patience=2,
        )

        return {
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": cfg.training.scheduler.monitor,
            }
        }

    return {}


def dehydrate_classifier_head(cfg: DictConfig, num_classes: int):
    name = cfg.model.classifier_head.name

    if name == "dual_branch_v1":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 2, 256),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(256, num_classes),
        )

    if name == "dual_branch_v2":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    raise Exception("Unknown classifier head")
