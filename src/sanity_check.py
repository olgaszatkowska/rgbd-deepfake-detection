import sys
from typing import Optional

import torch
from torchsummary import summary
import hydra
from omegaconf import DictConfig

from data.data_loader import FaceForensicsPlusPlus
from models.dehydrate import dehydrate_model

config_name: Optional[str] = None
check_type: Optional[str] = None

if len(sys.argv) > 1:
    config_name = sys.argv[1]
if len(sys.argv) > 2:
    check_type = sys.argv[2]

sys.argv = sys.argv[:1]

# Validation
assert config_name, "Missing config_name"
assert check_type in ["model", "data"], "Invalid type_check: must be 'model' or 'data'"


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_data(cfg: DictConfig):
    data_module = FaceForensicsPlusPlus(cfg)
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Estimated steps per epoch: {len(train_loader)}")


def check_model(cfg: DictConfig):
    model = dehydrate_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nModel Summary (input: 4x224x224):")
    summary(model, input_size=(4, 224, 224), device=str(device))
    print(f"\nTotal trainable parameters: {count_params(model):,}")


@hydra.main(config_path="../conf", config_name=config_name, version_base="1.3")
def main(cfg: DictConfig):
    if check_type == "data":
        check_data(cfg)
    elif check_type == "model":
        check_model(cfg)


if __name__ == "__main__":
    main()
