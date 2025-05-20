from omegaconf import DictConfig

from models import DualBranchRGBDNet, TransformerRGDBNet
import torch.nn as nn

def dehydrate_model(cfg: DictConfig) -> nn.Module:
    model_type = cfg.model.type

    if model_type == "transformer":
        return TransformerRGDBNet(cfg)

    if model_type == "dual_branch":
        return DualBranchRGBDNet(cfg)

    raise Exception("Unknown network type")
