import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loaders import LinkNeighborLoader
from pytorch_lightning.loggers import CSVLogger

from data.import_data import import_knowledge_graph
from plotting import plot_loss
import cfg
import utils
from models import LinkPredictor


pl.seed_everything(42)
accelerator, graph_device = utils.get_devices()

KG = import_knowledge_graph(path=cfg.KG_path, device=graph_device, drop_labels=cfg.drop_labels)

train_loader = LinkNeighborLoader(KG, [cfg.num_sampled_neighbors] * cfg.n_layers, batch_size=cfg.batch_size, edge_label_index=KG.edge_index, neg_sampling='triplet', shuffle=True) # binary?
# sampled_data = next(iter(train_loader))
# print(sampled_data)
model = LinkPredictor(KG.num_nodes, cfg.num_nodes, cfg.n_layers, cfg.in_channels, cfg.hidden_channels, cfg.n_heads, cfg.dropout, cfg.residual, cfg.margin)
# print(model(sampled_data.edge_index, sampled_data.src_index, sampled_data.dst_pos_index))
csv_logger = CSVLogger("results/pretrain", name=cfg.save_as)
trainer = Trainer(accelerator=accelerator, devices=cfg.num_gpu, strategy="ddp", max_epochs=cfg.max_epochs, precision="16-mixed", logger=csv_logger)
trainer.fit(model, train_loader)

plot_loss(csv_logger.log_dir)

