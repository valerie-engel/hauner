
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
    def __init__(self, config):
        super().__init__()
        self.gat = pyg.nn.GATv2Conv(config.in_channels, config.hidden_channels, config.n_heads) # residuals?
        self.ln = pyg.nn.LayerNorm(config.hidden_channels)
        # dropout?

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.ln(x)
        return x

class GAT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.GAT = nn.ModuleList([GAT_layer(config) for _ in range(config.n_layers)])

    def forward(self, x, edge_index):
        for layer in self.GAT:
            x = layer(x, edge_index)
        return x

# class MaxMarginClassifier(nn.model):
#     def max_margin_loss(self, sim_pos, sim_neg, margin=1):
#         error = margin - sim_pos + sim_neg
#         loss = torch.max(torch.tensor(0), error)
#         return torch.mean(loss, axis=1) #which axis?
    
#     def training_step(self, x, src_index, dst_index):

class MaxMarginLoss(pl.LightningModule):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
        
    def forward(self, logits_pos, logits_neg):
        loss = torch.clamp(self.margin - logits_pos + logits_neg, min=0.0)
        return loss.mean() 

class LinkPredictor(pl.LightningModule):
    # more elegant with standard sampler and handling neg. sampling myself?
    def __init__(self, num_nodes, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, config.in_channels)
        self.encoder = GAT(config)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_fn = MaxMarginLoss() #(margin=config.margin) # torch.nn.BCEWithLogitsLoss() 

    def forward(self, edge_index, src_index, dst_index):
        n = src_index.shape[0]
        x = self.embedding((torch.cat((src_index,dst_index))))
        x = self.encoder(x, edge_index)
        # reshaping to broadcast similarity calculation to multiple destinations per source node (needed during training)
        d = x.shape[-1]
        x_src = x[:n].unsqueeze(0)
        x_dst = x[n:].view(-1, n, d) 
        logits = self.similarity(x_src, x_dst) 
        return logits.flatten()
    
    def training_step(self, batch, batch_idx):
        edge_index, src_index, dst_pos_index, dst_neg_index = batch.edge_index, batch.src_index, batch.dst_pos_index, batch.dst_neg_index
        # Data(edge_index=[2, 22], y=[24], edge_label=[22], num_nodes=24, n_id=[24], e_id=[22], num_sampled_nodes=[2], num_sampled_edges=[1], input_id=[2], src_index=[2], dst_pos_index=[2], dst_neg_index=[2])
        num_pos = dst_pos_index.shape[-1] 
        dst_index = torch.cat((dst_pos_index, dst_neg_index))
        logits = self(edge_index, src_index, dst_index)
        loss = self.loss_fn(logits_pos=logits[:num_pos], logits_neg=logits[num_pos:]) 
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    