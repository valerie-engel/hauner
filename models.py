
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric.nn as nn
# import BatchNorm, LayerNorm, GATv2Conv

class GAT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gat = nn.models.GAT(config.in_channels, config.hidden_channels, config.num_layers, norm=nn.LayerNorm)


# class GAT_layer_normed(nn.Module):
#     def __init__(self, dim_in, dim_out, n_heads):
#         super().__init__()
#         self.gat = nn.GATv2Conv(dim_in, dim_out, n_heads) # residuals?
#         self.ln = nn.LayerNorm(dim_out)
#         # dropout?

#     def forward(self, x, edge_index):
#         x = self.gat(x)
#         x = self.ln(x)
#         return x

# class GAT_normed(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.GAT = nn.ModuleList([GAT_normed for _ in config.n_layers])

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
        return loss.mean() # axis?

class LinkPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.encoder = GAT(config)
        self.similarity = F.cosine_similarity
        self.loss_fn = MaxMarginLoss() #(margin=config.margin) # torch.nn.BCEWithLogitsLoss() 

    def forward(self, x, edge_index, src_index, dst_index):
        x = self.encoder(x, edge_index)
        logits = self.similarity(x[src_index], x[dst_index])
        return logits
    
    def training_step(self, batch, batch_idx):
        print(batch.x)
        x, edge_index, src_index, dst_pos_index, dst_neg_index = batch.x, batch.edge_index, batch.src_index, batch.dst_pos_index, batch.dst_neg_index
        # Data(edge_index=[2, 22], y=[24], edge_label=[22], num_nodes=24, n_id=[24], e_id=[22], num_sampled_nodes=[2], num_sampled_edges=[1], input_id=[2], src_index=[2], dst_pos_index=[2], dst_neg_index=[2])
        # x = self.encoder(x, edge_index)
        print(dst_pos_index.shape)
        num_pos = dst_pos_index.shape[-1] # correct?
        dst_index = torch.stack((dst_pos_index, dst_neg_index))
        # labels = 
        logits = self(x, edge_index, src_index, dst_index)
        print(logits.shape)
        loss = self.loss_fn(logits_pos=logits[:num_pos], logits_neg=logits[num_pos:]) #which axis?
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    


        
