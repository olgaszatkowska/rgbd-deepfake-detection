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

CONFIG_NAME: Optional[str] = None
CKPT_FILENAME: Optional[str] = None
INPUT_TYPE: Optional[str] = None

if len(sys.argv) > 1:
    CONFIG_NAME = sys.argv[1]
if len(sys.argv) > 2:
    CKPT_FILENAME = sys.argv[2]
if len(sys.argv) > 3:
    INPUT_TYPE = sys.argv[3]

sys.argv = sys.argv[:1]

assert CONFIG_NAME, "Missing config name"
assert CKPT_FILENAME, "Missing checkpoint filename"
assert INPUT_TYPE in ["depth", "rgb"], "Invalid or missing input type"


def load_model(cfg: DictConfig):
    network = dehydrate_model(cfg)
    detector = RGBDDetector.load_from_checkpoint(
        f"checkpoints/{CKPT_FILENAME}.ckpt", cfg=cfg, model=network
    )

    return detector.model


def generate_grad_cam(
    model: Module,
    input_tensor: Tensor,
    input_type: str,
    target_class: Optional[int] = None
) -> Tensor:
    model.eval()

    activations = []
    gradients = []

    # Choose branch
    if input_type == "rgb":
        target_layer = model.rgb_base[-1]  # Last conv layer of RGB branch
    elif input_type == "depth":
        target_layer = model.depth_base[-1]  # Last conv layer of Depth branch
    else:
        raise ValueError("Invalid input_type")

    def forward_hook(module, input, output):
        activations.append(output)
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    input_tensor.requires_grad_()
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1)

    class_score = output[range(output.size(0)), target_class].sum()
    model.zero_grad()
    class_score.backward()

    # Cleanup hooks
    handle_fwd.remove()
    handle_bwd.remove()

    if not gradients or not activations:
        raise RuntimeError("Failed to capture gradients or activations for Grad-CAM.")

    grad = gradients[0]       # [B, C, H, W]
    activation = activations[0]  # [B, C, H, W]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    # Normalize per image
    cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
    cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam



@hydra.main(config_path="../conf", config_name=CONFIG_NAME, version_base="1.3")
def main(cfg: DictConfig):
    model = load_model(cfg=cfg)
    device = torch.device(cfg.training.accelerator)
    model = model.to(device)

    datamodule = FaceForensicsPlusPlus(cfg)
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()

    save_dir = Path(f"heat_maps/{cfg.model.name}/{INPUT_TYPE}")
    save_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    labels = batch["label"]

    cams = generate_grad_cam(model, images, INPUT_TYPE)

    for i in range(min(10, images.size(0))):
        image_tensor = images[i].detach().cpu()
        cam_tensor = cams[i].squeeze().detach().cpu()

        # Take only the RGB channels
        if image_tensor.size(0) > 3:
            image_tensor = image_tensor[:3]

        # Normalize to [0, 1] using dynamic min/max
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        image_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)

        # Convert to PIL image
        input_img = F.to_pil_image(image_tensor)
        cam_img = F.to_pil_image(cam_tensor)

        sample_dir = save_dir / f"sample_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save input image
        input_img.save(sample_dir / "input.png")

        # Save heatmap overlay
        fig, ax = plt.subplots()
        ax.imshow(input_img)
        ax.imshow(cam_img, cmap="jet", alpha=0.5)
        ax.axis("off")
        fig.savefig(sample_dir / "map.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
