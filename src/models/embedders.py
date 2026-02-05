
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric as pyg
# import BatchNorm, LayerNorm, GATv2Conv

# class GAT(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.gat = nn.models.GAT(config.in_channels, config.hidden_channels, config.num_layers, norm=nn.LayerNorm)

class GAT_layer(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, dropout, residual):
        super().__init__()
        self.gat = pyg.nn.GATv2Conv(in_channels=in_channels, out_channels=out_channels, heads=n_heads, dropout=dropout, residual=residual)
        self.ln = pyg.nn.LayerNorm(out_channels * n_heads)

    def forward(self, x, edge_index):
        assert x.dim() == 2
        assert edge_index.dim() == 2
        assert edge_index.size(0) == 2
        assert edge_index.dtype == torch.long
        assert edge_index.min() >= 0
        assert edge_index.max() < x.size(0)
        assert torch.isfinite(x).all()
        
        # print(f"device x: {x.get_device()}")
        # print(f"device edge_index: {edge_index.get_device()}")
        x = self.gat(x, edge_index)
        x = self.ln(x)
        return x

class GAT(pl.LightningModule):
    def __init__(self, n_layers, in_channels, hidden_channels, out_channels,n_heads, dropout, residual):
        super().__init__()
        self.n_layers = n_layers
        self.in_layer = GAT_layer(
            in_channels=in_channels, 
            out_channels=hidden_channels, # if I only had 1 layer this should prob. be out_channels
            n_heads=n_heads, 
            dropout=dropout, 
            residual=residual
            )
        self.layers = nn.ModuleList([GAT_layer(
            in_channels=hidden_channels * n_heads, 
            out_channels=hidden_channels, 
            n_heads=n_heads, 
            dropout=dropout, 
            residual=residual
            ) for _ in range(n_layers - 2)])
        if self.n_layers > 1:
            self.out_layer = GAT_layer(
            in_channels=hidden_channels * n_heads, 
            out_channels=out_channels, 
            n_heads=n_heads, 
            dropout=dropout, 
            residual=residual
            )

    def forward(self, x, edge_index):
        x = self.in_layer(x, edge_index)
        if self.n_layers > 1:
            for layer in self.layers:
                x = layer(x, edge_index)
            x = self.out_layer(x, edge_index)
        return x

class MaxMarginLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
        
    def forward(self, logits_pos, logits_neg):
        loss = torch.clamp(self.margin - logits_pos + logits_neg, min=0.0)
        return loss.mean() 


class TypeEmbedder(nn.Module):
    def __init__(self, num_types, channels, existing_embedding=None):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, channels)
        self.existing_embedding = existing_embedding

    def forward(self, types, x=None):
        assert (self.existing_embedding is None) == (x is None), f"This model expects precomputed {self.existing_embedding} embeddings. Please pass them for correct computation."

        type_embedding = self.type_embedding(types)
        if self.existing_embedding:
            return x + type_embedding
        else:
            return type_embedding


class LinkPredictor(pl.LightningModule):
    def __init__(self, num_types, n_layers, in_channels, hidden_channels, out_channels, n_heads, dropout=0.0, residual=False, existing_embedding=None, margin=1):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = TypeEmbedder(num_types, in_channels, existing_embedding=existing_embedding)
        self.encoder = GAT(
            n_layers=n_layers, 
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            n_heads= n_heads, 
            dropout=dropout, 
            residual=residual)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_fn = MaxMarginLoss(margin=margin) # torch.nn.BCEWithLogitsLoss() 

    def forward(self, x, edge_index, src_index, dst_index):
        # n = src_index.shape[0]
        # x = self.embedding((torch.cat((src_index,dst_index))))
        # x = self.encoder(x, edge_index)
        # # reshaping to broadcast similarity calculation to multiple destinations per source node (needed during training)
        # d = x.shape[-1]
        # x_src = x[:n].unsqueeze(0)
        # x_dst = x[n:].view(-1, n, d) 
        # logits = self.similarity(x_src, x_dst) 
        # return logits.flatten()
        z = self.encoder(x, edge_index)
        return self.similarity(z[src_index], z[dst_index])
    
    def training_step(self, batch, batch_idx):
        # edge_index, src_index, dst_pos_index, dst_neg_index = batch.edge_index, batch.src_index, batch.dst_pos_index, batch.dst_neg_index
        # # Data(edge_index=[2, 22], y=[24], edge_label=[22], num_nodes=24, n_id=[24], e_id=[22], num_sampled_nodes=[2], num_sampled_edges=[1], input_id=[2], src_index=[2], dst_pos_index=[2], dst_neg_index=[2])
        # num_pos = dst_pos_index.shape[-1] 
        # dst_index = torch.cat((dst_pos_index, dst_neg_index))
        # logits = self(edge_index, src_index, dst_index)
        # loss = self.loss_fn(logits_pos=logits[:num_pos], logits_neg=logits[num_pos:]) 
        # print(batch)
        batch = batch.cuda()

        x = self.embedding(batch.y.long(), batch.x) #any chance this exists..?

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
    