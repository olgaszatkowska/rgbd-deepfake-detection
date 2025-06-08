import logging
import sys
from typing import Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from models import RGBDDetector
from data.data_loader import FaceForensicsPlusPlus
from models.dehydrate import dehydrate_model

logger = logging.getLogger(__name__)

config_name: Optional[str] = None

if len(sys.argv) > 1:
    config_name = sys.argv[1]
    sys.argv = sys.argv[:1]


def train_from_config(cfg: DictConfig):
    data_module = FaceForensicsPlusPlus(cfg)
    data_module.setup()

    network = dehydrate_model(cfg)
    model = RGBDDetector(cfg=cfg, model=network, lr=cfg.training.learning_rate)
    model_name = cfg.model.name

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"{cfg.model.name}" + "-{epoch:02d}-{val_acc:.2f}",
        dirpath="checkpoints/",
    )

    callbacks = [checkpoint_callback]

    if cfg.training.early_stopping_patience:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

    logger_tb = TensorBoardLogger("lightning_logs_tf", name=model_name)
    logger_csv = CSVLogger("lightning_logs_csv", name=model_name)

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        callbacks=callbacks,
        devices=1,
        logger=[logger_tb, logger_csv],
        gradient_clip_val=0.5,
    )

    trainer.fit(model, datamodule=data_module)


@hydra.main(config_path="../conf", config_name=config_name, version_base="1.3")
def train_dual_branch_attention(cfg: DictConfig):
    train_from_config(cfg=cfg)


if __name__ == "__main__":
    train_dual_branch_attention()
