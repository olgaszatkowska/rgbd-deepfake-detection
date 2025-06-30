import sys
from typing import Optional
from pathlib import Path

import re
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

def extract_config_name(filename):
    match = re.match(r"(.+?)-epoch=\d+-val_acc=\d+\.\d+", filename)
    if match:
        return match.group(1)
    return None

CKPT_FILENAME: Optional[str] = None
if len(sys.argv) > 1:
    CKPT_FILENAME = sys.argv[1]
sys.argv = sys.argv[:1]
assert CKPT_FILENAME, "Missing checkpoint filename"
CONFIG_NAME = extract_config_name(CKPT_FILENAME)

def load_model(cfg: DictConfig):
    network = dehydrate_model(cfg)
    detector = RGBDDetector.load_from_checkpoint(
        f"checkpoints/{CKPT_FILENAME}.ckpt", cfg=cfg, model=network
    )
    return detector.model

def generate_fused_spatial_grad_cam(model: Module, input_tensor: Tensor, target_class: Optional[int] = None) -> Tensor:
    model.eval()
    gradients = []

    # Get spatial fused features
    fused_spatial = model(input_tensor, return_fused_spatial=True)
    fused_spatial.retain_grad()

    def grad_hook(grad):
        gradients.append(grad)

    fused_spatial.register_hook(grad_hook)

    # Manually classify with a temporary linear layer compatible with fused features
    pooled = torch.nn.functional.adaptive_avg_pool2d(fused_spatial, output_size=(1, 1))
    flattened = torch.flatten(pooled, 1)

    # Create temporary classifier for Grad-CAM
    temp_classifier = torch.nn.Linear(flattened.shape[1], model.classifier[-1].out_features).to(flattened.device)
    output = temp_classifier(flattened)

    if target_class is None:
        target_class = output.argmax(dim=1)

    class_score = output[range(output.size(0)), target_class].sum()
    model.zero_grad()
    class_score.backward()

    grad = gradients[0]
    activation = fused_spatial

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
    datamodule.setup("test")
    val_loader = datamodule.test_dataloader()

    save_dir = Path(f"heat_maps/{cfg.model.name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    num_batches = 10  # Number of batches to process
    sample_counter = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"]
        outputs, _ = model(images)
        preds = outputs.argmax(dim=1).detach().cpu()

        # Generate Grad-CAM from fused spatial features
        cams_fused = generate_fused_spatial_grad_cam(model, images)

        for i in range(images.size(0)):
            image_tensor = images[i].detach().cpu()
            cam = cams_fused[i].detach().cpu()

            # Resize CAM to match image
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze()

            # Normalize RGB image and CAM
            def normalize(t):
                return (t - t.min()) / (t.max() - t.min() + 1e-8)

            rgb_img = normalize(image_tensor[:3])
            input_rgb = F.to_pil_image(rgb_img)
            heatmap = F.to_pil_image(cam)

            # Plot: original and overlayed heatmap
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))

            axs[0].imshow(input_rgb)
            axs[0].set_title("Input RGB")
            axs[0].axis("off")

            axs[1].imshow(input_rgb)
            axs[1].imshow(heatmap, cmap="jet", alpha=0.5)
            axs[1].set_title("Fused Grad-CAM")
            axs[1].axis("off")

            fig.tight_layout()

            pred = preds[i].item()
            label = labels[i].item()
            correctness = "correct" if pred == label else "wrong"
            filename = f"sample_{sample_counter}_{correctness}_{label}.png"

            fig.savefig(save_dir / filename, bbox_inches="tight")
            sample_counter += 1
            plt.close(fig)

if __name__ == "__main__":
    main()
