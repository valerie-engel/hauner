
import torch
import torch.nn as nn
# import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric as pyg


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
    def __init__(self, n_layers, in_channels, hidden_channels, out_channels, n_heads, dropout, residual):
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


