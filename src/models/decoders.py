import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import global_mean_pool
from torchmetrics.classification import MulticlassAccuracy

from base_models import GAT, TypeEncoding

class GraphClassifier(pl.LightningModule):
    def __init__(
        self,
        # model,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.encoder = TypeEncoding(num_types, channels, existing_embedding)
        self.model = GAT
        self.classifier = torch.nn.Linear(model.out_channels, num_classes)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, data):
        """
        data: torch_geometric.data.Data
        """
        x = self.model(data.x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        logits = self.classifier(x)
        return logits

    def _step(self, batch, stage: str):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)

        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.train_acc(preds, batch.y)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", self.train_acc, prog_bar=True)

        elif stage == "val":
            self.val_acc(preds, batch.y)
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)

        elif stage == "test":
            self.test_acc(preds, batch.y)
            self.log("test_loss", loss)
            self.log("test_acc", self.test_acc)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
