import pytorch_lightning as pl
import torch
from torch import Tensor
from torchmetrics import Accuracy
from typing import Any, Dict

from models.dehydrate import dehydrate_scheduler_config, dehydrate_loss


class RGBDDetector(pl.LightningModule):
    def __init__(self, cfg: Any, model: torch.nn.Module, lr: float = 1e-4) -> None:
        super().__init__()
        self.cfg = cfg

        self.model = model
        self.lr = lr
        self.criterion = dehydrate_loss(cfg=cfg)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.confidences = []

        self.train_loss_epoch: list[float] = []
        self.val_loss_epoch: list[float] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_accuracy(logits, y)
        preds = torch.argmax(logits, dim=1)
        f2 = self.compute_f2_score(y, preds)

        conf = torch.softmax(logits, dim=1).max(dim=1).values
        avg_conf = conf.mean()
        self.confidences.append(avg_conf.item())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f2", f2)
        self.log("val_confidence", avg_conf)
        return {"val_loss": loss, "val_acc": acc, "val_f2": f2, "val_conf": avg_conf}

    def on_validation_epoch_end(self) -> None:
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        train_acc = self.trainer.callback_metrics.get("train_acc")
        val_acc = self.trainer.callback_metrics.get("val_acc")

        if self.confidences:
            self.log(
                "epoch_avg_confidence", sum(self.confidences) / len(self.confidences)
            )
            self.confidences.clear()

        if train_loss and val_loss:
            self.log("loss_gap", train_loss - val_loss)
        if train_acc and val_acc:
            self.log("acc_gap", train_acc - val_acc)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self) -> Dict[str, Any]:
        lr = self.cfg.training.learning_rate

        uses_rgb_guided_block = (
            self.model.use_rgb_guided_attention
            or self.model.use_bidirectional_attention
        )

        uses_depth_guided_block = self.model.use_bidirectional_attention

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.rgb_base.parameters(), "lr": lr * 0.01},
                {"params": self.model.depth_base.parameters(), "lr": lr * 0.01},
                {"params": self.model.classifier.parameters(), "lr": lr},
                *(
                    [
                        {
                            "params": self.model.rgb_guides_depth.parameters(),
                            "lr": lr * 0.5,
                        }
                    ]
                    if uses_rgb_guided_block
                    else []
                ),
                *(
                    [
                        {
                            "params": self.model.depth_guides_rgb.parameters(),
                            "lr": lr * 0.5,
                        }
                    ]
                    if uses_depth_guided_block
                    else []
                ),
            ],
            weight_decay=1e-4,
        )

        scheduler_config = dehydrate_scheduler_config(self.cfg, optimizer)

        return {"optimizer": optimizer, **scheduler_config}

    @classmethod
    def compute_f2_score(cls, y_true, y_pred):
        from sklearn.metrics import fbeta_score

        return fbeta_score(y_true.cpu(), y_pred.cpu(), beta=2.0, average="macro")
