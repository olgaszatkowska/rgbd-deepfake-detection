from collections import Counter
import sys
from typing import Optional

import torch
from torchsummary import summary
import hydra
from omegaconf import DictConfig

from data.faceforensics import FaceForensics
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
    for split in ["train", "val", "test"]:
        print(f"\n=== Checking split: {split} ===")
        dataset = FaceForensics(conf=cfg, split=split)

        # Count class indices (0 = real, 1 = fake in binary mode)
        label_counts = Counter(dataset.dataset["classes"])
        label_names = cfg.data.real + cfg.data.attacks

        print("Class balance (numeric):")
        for k, v in label_counts.items():
            print(f"  Class {k}: {v} samples")

        # Count original labels (source names)
        source_counts = Counter(dataset.dataset["labels"])
        print("\nClass balance (label names):")
        for k, v in source_counts.items():
            print(f"  {k}: {v} samples")

        total = sum(source_counts.values())
        real_total = sum(source_counts[r] for r in cfg.data.real if r in source_counts)
        fake_total = sum(source_counts[a] for a in cfg.data.attacks if a in source_counts)

        print(f"\nTotal samples: {total}")
        print(f"Real samples: {real_total}")
        print(f"Fake samples: {fake_total}")
        print("-" * 40)


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
