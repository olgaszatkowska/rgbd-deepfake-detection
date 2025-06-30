import sys
import re
from typing import Optional
from pathlib import Path

import hydra
import torch
import pandas as pd
from omegaconf import DictConfig

from data.data_loader import FaceForensicsPlusPlus
from models import RGBDDetector
from models.dehydrate import dehydrate_model
from sklearn.metrics import fbeta_score, log_loss

# Expected CLI: python evaluate_checkpoints.py CHECKPOINT_NAME
CKPT_FILENAME: Optional[str] = None

if len(sys.argv) > 1:
    CKPT_FILENAME = sys.argv[1]

sys.argv = sys.argv[:1]

assert CKPT_FILENAME, "Missing checkpoint filename"

def extract_config_name(filename: str) -> Optional[str]:
    match = re.match(r"(.+?)-epoch=\d+-val_acc=\d+\.\d+", filename)
    return match.group(1) if match else None

CONFIG_NAME = extract_config_name(CKPT_FILENAME)
assert CONFIG_NAME, f"Could not parse config name from {CKPT_FILENAME}"

ATTACKS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
REALS = ["youtube"]

COL_NAMES = ATTACKS + REALS

RESEARCH_DIR = Path("checkpoint_eval")
RESEARCH_DIR.mkdir(exist_ok=True, parents=True)

def load_model(cfg: DictConfig) -> RGBDDetector:
    model_arch = dehydrate_model(cfg)
    model = RGBDDetector.load_from_checkpoint(
        f"checkpoints/{CKPT_FILENAME}.ckpt", cfg=cfg, model=model_arch
    )
    return model

def evaluate_model(model: RGBDDetector, cfg: DictConfig):
    records = {g: {"acc": [], "f2": [], "loss": [], "conf": []} for g in COL_NAMES}
    for col_name in COL_NAMES:

        if col_name in ATTACKS:
            cfg.data.attacks = [col_name]
            cfg.data.real = []

        if col_name in REALS:
            cfg.data.attacks = []
            cfg.data.real = [col_name]

        datamodule = FaceForensicsPlusPlus(cfg)
        datamodule.setup("test")
        
        device = torch.device(cfg.training.accelerator)
        model.eval().to(device)

        val_loader = datamodule.test_dataloader()

        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)[0]
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            for i in range(len(x)):
                true = y[i].item()
                pred = preds[i].item()
                prob = probs[i].detach().cpu().numpy()

                record = records[col_name]
                record["acc"].append(int(pred == true))
                record["f2"].append(fbeta_score([true], [pred], beta=2.0, average="macro"))
                record["loss"].append(log_loss([true], [prob], labels=[0, 1]))
                record["conf"].append(max(prob))

    return records

def aggregate(records: dict) -> pd.DataFrame:
    total = {"acc": [], "f2": [], "loss": [], "conf": []}
    rows = []

    for group, metrics in records.items():
        row = {
            "attack": group,
            "accuracy": sum(metrics["acc"]) / len(metrics["acc"]) if metrics["acc"] else 0,
            "f2_score": sum(metrics["f2"]) / len(metrics["f2"]) if metrics["f2"] else 0,
            "loss": sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else 0,
            "confidence": sum(metrics["conf"]) / len(metrics["conf"]) if metrics["conf"] else 0,
        }
        rows.append(row)

        for key in total:
            total[key].extend(metrics[key])

    all_row = {
        "attack": "All",
        "accuracy": sum(total["acc"]) / len(total["acc"]) if total["acc"] else 0,
        "f2_score": sum(total["f2"]) / len(total["f2"]) if total["f2"] else 0,
        "loss": sum(total["loss"]) / len(total["loss"]) if total["loss"] else 0,
        "confidence": sum(total["conf"]) / len(total["conf"]) if total["conf"] else 0,
    }
    rows.append(all_row)
    return pd.DataFrame(rows)

@hydra.main(config_path="../conf", config_name=CONFIG_NAME, version_base="1.3")
def main(cfg: DictConfig):
    print(f"Running evaluation for checkpoint: {CKPT_FILENAME}")
    model = load_model(cfg)
    
    records = evaluate_model(model, cfg)
    df = aggregate(records)
    out_path = RESEARCH_DIR / f"{CKPT_FILENAME}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved CSV to {out_path}")

if __name__ == "__main__":
    main()
