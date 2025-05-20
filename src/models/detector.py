import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from utils import dehydrate_scheduler_config


class RGBDDetector(pl.LightningModule):
    def __init__(self, cfg, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=2)

        self.train_loss_epoch = []
        self.val_loss_epoch = []

        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        train_acc = self.trainer.callback_metrics.get("train_acc")
        val_acc = self.trainer.callback_metrics.get("val_acc")

        if train_loss and val_loss:
            self.log("loss_gap", train_loss - val_loss)
        if train_acc and val_acc:
            self.log("acc_gap", train_acc - val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )

        scheduler_config = dehydrate_scheduler_config(self.cfg, optimizer)

        return {
            "optimizer": optimizer,
            **scheduler_config
        }
