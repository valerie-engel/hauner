import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import LinkNeighborLoader
from pytorch_lightning.loggers import CSVLogger

from data import import_patients
from plotting import plot_loss
import cfg
import utils

pl.seed_everything(42)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu' #, graph_device = utils.get_devices(cfg.num_gpu)

patients, _ = import_patients(
    path=cfg.KG_path, 
    num_hops = 1,
    device='cpu', 
    undirected=cfg.undirected, 
    embedding=cfg.embedding, 
    drop_labels=cfg.drop_labels, 
    drop_unembedded=cfg.drop_unembedded
    )
