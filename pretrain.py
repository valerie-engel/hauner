import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import LinkNeighborLoader

from data.import_data import import_knowledge_graph
from plotting import plot_loss
import cfg
from models import LinkPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu' #torch.device()
print(f"Device: {device}")

KG = import_knowledge_graph(path=cfg.tiny_KG_path, device=device) #, drop_labels=['Biological_sample', 'Subject']
# print(KG.keys())
pl.seed_everything(42)
train_loader = LinkNeighborLoader(KG, [5] * cfg.n_layers, batch_size=3, edge_label_index=KG.edge_index, neg_sampling='triplet') # binary?
sampled_data = next(iter(train_loader))
# print(sampled_data)
model = LinkPredictor(KG.num_nodes, cfg)
# print(model(sampled_data.edge_index, sampled_data.src_index, sampled_data.dst_pos_index))

csv_logger = CSVLogger("results", name="pretrain")
trainer = Trainer(accelerator=device, max_epochs=cfg.max_epochs, logger=csv_logger)
trainer.fit(model, train_loader)

plot_loss(csv_logger.log_dir)

