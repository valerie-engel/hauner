import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import LinkNeighborLoader
from pytorch_lightning.loggers import CSVLogger

from data import import_knowledge_graph
from plotting import plot_loss
import configs.cfg_pretrain as cfg
import utils
from models.embedders import LinkPredictor


pl.seed_everything(42)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu' #, graph_device = utils.get_devices(cfg.num_gpu)

KG, _ = import_knowledge_graph(
    path=cfg.KG_path, 
    device='cpu', 
    undirected=cfg.undirected, 
    embedding=cfg.embedding, 
    drop_labels=cfg.drop_labels, 
    drop_unembedded=cfg.drop_unembedded
    ) #graph_device
print(f'KG.x exists: {KG.x is not None}')

train_loader = LinkNeighborLoader(
    KG, 
    [cfg.num_sampled_neighbors] * cfg.n_layers, 
    batch_size=cfg.batch_size, 
    edge_label_index=KG.edge_index, 
    neg_sampling='triplet', 
    shuffle=True
    ) # binary?
train_loader.num_neighbors = [cfg.num_sampled_neighbors] * cfg.n_layers
# sampled_data = next(iter(train_loader))

num_node_types = len(KG.y.unique()) 
in_channels = cfg.in_channels if KG.x is None else KG.x.shape[-1]
model = LinkPredictor(
    num_types=num_node_types, 
    n_layers=cfg.n_layers, 
    in_channels=in_channels, 
    hidden_channels=cfg.hidden_channels, 
    out_channels=cfg.out_channels, 
    n_heads=cfg.n_heads, 
    dropout=cfg.dropout, 
    existing_embedding=cfg.embedding,
    residual=cfg.residual, 
    margin=cfg.margin
    )
# x = model.embedding(sampled_data.y).cuda()
# print(model(sampled_data.edge_index.cuda(), sampled_data.src_indexcuda(), sampled_data.dst_pos_indexcuda()))

csv_logger = CSVLogger("results/pretrain", name=cfg.save_as)
trainer = Trainer(
    accelerator=accelerator, 
    devices=torch.cuda.device_count(), 
    strategy="ddp", 
    max_epochs=cfg.max_epochs, 
    precision="16-mixed", # 
    logger=csv_logger
    )
trainer.fit(model, train_loader)

plot_loss(csv_logger.log_dir)

