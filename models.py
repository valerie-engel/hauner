
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


class LinkPredictor(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.encoder = model(config)
        self.sim = F.cosine_similarity()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss() # max margin...

    def max_margin_loss(self, sim_pos, sim_neg, margin=1):
        error = margin - sim_pos + sim_neg
        loss = torch.max(torch.tensor(0), error)
        return torch.mean(loss, axis=1) #which axis?

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def predict(self, x, src_index, dst_index):
        logits = self.sim(x[src_index], x[dst_index])
        return logits
    
    def compute_loss(self, x, src_index, dst_pos_index, dst_neg_index):
        logits_pos = self.predict(x, src_index, dst_pos_index)
        logits_neg = self.predict(x, src_index, dst_neg_index)
        loss = self.max_margin_loss(logits_pos, logits_neg)
        return loss
    
    
    


        
