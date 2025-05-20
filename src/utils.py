from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim import Optimizer, lr_scheduler

from models import DualBranchRGBDNet, TransformerRGDBNet


def dehydrate_model(cfg: DictConfig) -> nn.Module:
    model_type = cfg.model.type

    if model_type == "transformer":
        return TransformerRGDBNet(cfg)

    if model_type == "dual_branch":
        return DualBranchRGBDNet(cfg)

    raise Exception("Unknown network type")


def dehydrate_scheduler_config(
    cfg: DictConfig, optimizer: Optimizer
) -> dict[str, lr_scheduler.LRScheduler | dict[str, lr_scheduler.LRScheduler | str]]:
    scheduler_type = cfg.training.scheduler_type

    if scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.cosine_T_max
        )

        return {"lr_scheduler": scheduler}

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.training.decay_factor, patience=2
        )

        return {
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": cfg.training.monitor
            }
        }

    raise Exception("Unknown scheduler type")
