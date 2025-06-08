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

    if scheduler_type == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.training.learning_rate,
            steps_per_epoch=cfg.training.scheduler.steps_per_epoch,
            epochs=cfg.training.max_epochs,
            pct_start=cfg.training.scheduler.get("pct_start", 0.3),
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

        return {
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    return {}


def dehydrate_classifier_head(cfg: DictConfig, num_classes: int) -> nn.Sequential:
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

    if name == "dual_branch_v3":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    if name == "transformer_v1":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    raise Exception("Unknown classifier head")


def dehydrate_loss(cfg: DictConfig, weights) -> nn.CrossEntropyLoss:
    if cfg.training.label_smoothing:
        return torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing, weight=weights)

    return torch.nn.CrossEntropyLoss(weight=weights)
