import sys
from typing import Optional
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.nn import Module
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from models import RGBDDetector
from data.data_loader import FaceForensicsPlusPlus
from models.dehydrate import dehydrate_model

config_name: Optional[str] = None
ckpt_filename: Optional[str] = None
input_type: Optional[str] = None

if len(sys.argv) > 1:
    config_name = sys.argv[1]
if len(sys.argv) > 2:
    ckpt_filename = sys.argv[2]
if len(sys.argv) > 3:
    input_type = sys.argv[3]

sys.argv = sys.argv[:1]

assert input_type in ["depth", "rgb"], "Invalid or missing input type"
assert config_name, "Missing config name"
assert ckpt_filename, "Missing checkpoint filename"


@hydra.main(config_path="../conf", config_name=config_name, version_base="1.3")
def load_model(cfg: DictConfig):
    network = dehydrate_model(cfg)
    detector = RGBDDetector.load_from_checkpoint(
        f"checkpoints/{ckpt_filename}", cfg=cfg, model=network
    )

    return detector.model


def generate_grad_cam(
    model: Module, input_tensor: Tensor, target_class: Optional[int] = None
) -> Tensor:
    model.eval()
    input_tensor.requires_grad_()
    rgb_feat, depth_feat = model(input_tensor, return_features=True)

    if input_type == "depth":
        feature_map = depth_feat
    if input_type == "rgb":
        feature_map = rgb_feat

    feature_map.retain_grad()

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1)

    class_score = output[range(output.size(0)), target_class].sum()
    class_score.backward()

    gradients: Tensor = feature_map.grad  # [B, C, H, W]
    weights: Tensor = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam: Tensor = (weights * feature_map).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)  # Normalize to [0,1]
    return cam


@hydra.main(config_path="../conf", config_name=config_name, version_base="1.3")
def main(cfg: DictConfig):
    model = load_model(cfg, ckpt_filename)
    device = torch.device(cfg.training.accelerator)
    model = model.to(device)

    datamodule = FaceForensicsPlusPlus(cfg)
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()

    save_dir = Path(f"heat_maps/{cfg.model.name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    labels = batch["label"]

    cams = generate_grad_cam(model, images)

    for i in range(min(10, images.size(0))):
        image = images[i].detach().cpu()
        cam = cams[i].squeeze().detach().cpu()

        # Take only RGB channels for visualization
        if image.size(0) > 3:
            image = image[:3]

        image = F.to_pil_image(image)
        cam = F.to_pil_image(cam)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.axis("off")

        fig.savefig(
            save_dir / f"sample_{i}_label_{labels[i].item()}.png", bbox_inches="tight"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
