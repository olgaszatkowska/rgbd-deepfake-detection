import torch
from torchsummary import summary
import hydra
from omegaconf import DictConfig

from models import DualBranchRGBDNet
from data.data_loader import FaceForensicsPlusPlus
from utils import dehydrate_model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(
    config_path="../conf", config_name="dual_branch_attention", version_base="1.3"
)
def check(cfg: DictConfig):
    # Initialize datamodule
    data_module = FaceForensicsPlusPlus(cfg)
    data_module.prepare_data()
    data_module.setup()

    train_size = len(data_module.train_dataloader().dataset)
    val_size = len(data_module.val_dataloader().dataset)

    print(f"Train set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # Initialize model
    model = dehydrate_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nModel Summary (input: 4x224x224):")
    summary(model, input_size=(4, 224, 224), device=str(device))

    print(f"\nTotal trainable parameters: {count_params(model):,}")


if __name__ == "__main__":
    check()
