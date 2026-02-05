import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import CSVLogger

from data.importing import import_patients
from plotting import plot_loss
import configs.cfg_decode as cfg
import utils

pl.seed_everything(42)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu' #, graph_device = utils.get_devices(cfg.num_gpu)

patients = import_patients(
    path=cfg.KG_path, 
    num_hops = cfg.num_hops,
    device='cpu', 
    undirected=cfg.undirected, 
    embedding=cfg.embedding, 
    drop_labels=cfg.drop_labels, 
    drop_unembedded=cfg.drop_unembedded
    )

# train_loader = DataLoader(patients, batch_size=cfg.batch_size, shuffle=True)
# # VAL 
# model = GATGraphClassifier(
#     gat_model=gat,
#     num_classes=num_classes,
#     lr=1e-3,
# )

# csv_logger = CSVLogger("results/decode", name=cfg.save_as)
# trainer = Trainer(
#     accelerator=accelerator, 
#     devices=torch.cuda.device_count(), 
#     strategy="ddp", 
#     max_epochs=cfg.max_epochs, 
#     precision="16-mixed", # 
#     logger=csv_logger
# )

# trainer.fit(model, train_loader, val_loader)