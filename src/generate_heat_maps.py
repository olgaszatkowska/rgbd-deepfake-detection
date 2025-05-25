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

if len(sys.argv) > 1:
    CONFIG_NAME = sys.argv[1]
if len(sys.argv) > 2:
    CKPT_FILENAME = sys.argv[2]

sys.argv = sys.argv[:1]

assert CONFIG_NAME, "Missing config name"
assert CKPT_FILENAME, "Missing checkpoint filename"


def load_model(cfg: DictConfig):
    network = dehydrate_model(cfg)
    detector = RGBDDetector.load_from_checkpoint(
        f"checkpoints/{CKPT_FILENAME}.ckpt", cfg=cfg, model=network
    )
    return detector.model


def generate_grad_cam(model: Module, input_tensor: Tensor, input_type: str, target_class: Optional[int] = None) -> Tensor:
    model.eval()

    activations = []
    gradients = []

    target_layer = {
        "rgb": model.rgb_base[-1],
        "depth": model.depth_base[-1],
    }[input_type]

    def forward_hook(module, input, output):
        activations.append(output)
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    input_tensor.requires_grad_()
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1)

    class_score = output[range(output.size(0)), target_class].sum()
    model.zero_grad()
    class_score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grad = gradients[0]
    activation = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

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

    save_dir = Path(f"heat_maps/{cfg.model.name}/combined")
    save_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    labels = batch["label"]

    cams_rgb = generate_grad_cam(model, images, "rgb")
    cams_depth = generate_grad_cam(model, images, "depth")

    for i in range(min(10, images.size(0))):
        image_tensor = images[i].detach().cpu()
        cam_rgb = cams_rgb[i].detach().cpu()
        cam_depth = cams_depth[i].detach().cpu()

        cam_rgb = torch.nn.functional.interpolate(
            cam_rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze()

        cam_depth = torch.nn.functional.interpolate(
            cam_depth.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze()

        rgb_img = image_tensor[:3]
        depth_img = image_tensor[3:4]

        # Normalize both inputs
        def normalize(t):
            return (t - t.min()) / (t.max() - t.min() + 1e-8)

        rgb_img = normalize(rgb_img)
        depth_img = normalize(depth_img)

        input_rgb = F.to_pil_image(rgb_img)
        input_depth = F.to_pil_image(depth_img.squeeze(0))

        heatmap_rgb = F.to_pil_image(cam_rgb)
        heatmap_depth = F.to_pil_image(cam_depth)

        # Create figure with 2 columns: RGB and Depth
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(input_rgb)
        axs[0].imshow(heatmap_rgb, cmap="jet", alpha=0.5)
        axs[0].axis("off")

        axs[1].imshow(input_depth, cmap="gray")
        axs[1].imshow(heatmap_depth, cmap="jet", alpha=0.5)
        axs[1].axis("off")

        fig.tight_layout()
        fig.savefig(save_dir / f"sample_{i}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
