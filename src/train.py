import logging

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch.nn as nn

from models import DualBranchRGBDNet, TransformerRGDBNet, RGBDDetector
from data.data_loader import FaceForensicsPlusPlus

logger = logging.getLogger(__name__)


def dehydrate_model(cfg: DictConfig) -> nn.Module:
    model_type = cfg.model.type

    if model_type == "transformer":
        return TransformerRGDBNet(cfg)
    
    if model_type == "dual_branch":
        return DualBranchRGBDNet(cfg)
    
    raise Exception("Unknown network type")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
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

    logger_tb = TensorBoardLogger("lightning_logs_tf", name=model_name)
    logger_csv = CSVLogger("lightning_logs_csv", name=model_name)

    trainer = Trainer(
        max_epochs=20,
        accelerator=cfg.training.accelerator,
        callbacks=[checkpoint_callback],
        devices=1,
        logger=[logger_tb, logger_csv],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
