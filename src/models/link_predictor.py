import torch
import torch.nn as nn
import pytorch_lightning as pl

from .gat import GAT
from .initial_embeddings import TypeEncoding
from .losses import MaxMarginLoss


class LinkPredictor(pl.LightningModule):
    def __init__(
        self,
        num_types,
        n_layers,
        in_channels,
        hidden_channels,
        out_channels,
        n_heads,
        dropout=0.0,
        residual=False,
        existing_embedding=None,
        margin=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = TypeEncoding(num_types, in_channels, existing_embedding=existing_embedding)
        self.encoder = GAT(
            n_layers=n_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_heads=n_heads,
            dropout=dropout,
            residual=residual,
        )
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_fn = MaxMarginLoss(margin=margin)

    def forward(self, x, edge_index, src_index, dst_index):
        z = self.encoder(x, edge_index)
        return self.similarity(z[src_index], z[dst_index])

    def training_step(self, batch, batch_idx):
        batch = batch.cuda()

        x = self.embedding(batch.y.long(), batch.x)

        pos_logits = self.forward(
            x,
            batch.edge_index,
            batch.src_index,
            batch.dst_pos_index,
        )

        neg_logits = self.forward(
            x,
            batch.edge_index,
            batch.src_index,
            batch.dst_neg_index,
        )

        loss = self.loss_fn(pos_logits, neg_logits)

        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0], sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
