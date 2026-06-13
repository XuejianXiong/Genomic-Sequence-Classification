import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision
)


class DNAClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        tuning: int = -2,
        lr: float = 2e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr

        # -------------------------
        # Backbone
        # -------------------------
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # -------------------------
        # Classifier head
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

        # -------------------------
        # Loss
        # -------------------------
        self.loss_fn = nn.CrossEntropyLoss()

        # -------------------------
        # Metrics (MULTICLASS ONLY)
        # -------------------------
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_pr_auc = MulticlassAveragePrecision(num_classes=num_classes, average="macro")

        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.test_pr_auc = MulticlassAveragePrecision(num_classes=num_classes, average="macro")

        # -------------------------
        # Apply tuning strategy
        # -------------------------
        self.set_tuning_layers(tuning)

        # -------------------------
        # Best tracking
        # -------------------------
        self.best_val_metrics = {
            "val_loss": float("inf"),
            "val_accu": 0.0,
            "val_roc_auc": 0.0,
            "val_pr_auc": 0.0,
        }

    # =========================================================
    # TUNING STRATEGY
    # =========================================================
    def set_tuning_layers(self, tuning: int):
        """
        tuning:
            0  -> freeze entire transformer
            1  -> full fine-tuning
            -k -> unfreeze last k encoder layers
        """

        # Freeze everything first
        for param in self.transformer.parameters():
            param.requires_grad = False

        if tuning == 0:
            return

        if tuning == 1:
            for param in self.transformer.parameters():
                param.requires_grad = True
            return

        if tuning < 0:
            n_layers = len(self.transformer.encoder.layer)
            k = abs(tuning)

            for layer in self.transformer.encoder.layer[n_layers - k:]:
                for param in layer.parameters():
                    param.requires_grad = True
            return

        raise ValueError("tuning must be 0, 1, or negative integer (e.g., -2, -4)")

    # =========================================================
    # FORWARD
    # =========================================================
    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = output.last_hidden_state

        # CLS token embedding (stable for DNA transformers)
        cls_emb = hidden[:, 0, :]

        logits = self.classifier(cls_emb)
        return logits

    # =========================================================
    # TRAIN
    # =========================================================
    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # =========================================================
    # VALIDATION
    # =========================================================
    def on_validation_start(self):
        self.val_accuracy.reset()
        self.val_auroc.reset()
        self.val_pr_auc.reset()

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.val_accuracy.update(preds, batch["labels"])
        self.val_auroc.update(probs, batch["labels"])
        self.val_pr_auc.update(probs, batch["labels"])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accu", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_roc_auc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val_pr_auc", self.val_pr_auc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return

        if float(val_loss) < self.best_val_metrics["val_loss"]:
            self.best_val_metrics = {
                "val_loss": float(val_loss),
                "val_accu": float(metrics["val_accu"]),
                "val_roc_auc": float(metrics["val_roc_auc"]),
                "val_pr_auc": float(metrics["val_pr_auc"]),
            }

    # =========================================================
    # TEST
    # =========================================================
    def on_test_start(self):
        self.test_accuracy.reset()
        self.test_auroc.reset()
        self.test_pr_auc.reset()

        self.test_probs = []
        self.test_labels = []
        self.test_ids = []

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.test_accuracy.update(preds, batch["labels"])
        self.test_auroc.update(probs, batch["labels"])
        self.test_pr_auc.update(probs, batch["labels"])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accu", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log("test_roc_auc", self.test_auroc, on_epoch=True, prog_bar=True)
        self.log("test_pr_auc", self.test_pr_auc, on_epoch=True, prog_bar=True)

        self.test_probs.append(probs.cpu())
        self.test_labels.append(batch["labels"].cpu())
        self.test_ids.extend(batch["id"])

    def on_test_end(self):
        self.test_probs = torch.cat(self.test_probs)
        self.test_labels = torch.cat(self.test_labels)

    # =========================================================
    # OPTIMIZER
    # =========================================================
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)