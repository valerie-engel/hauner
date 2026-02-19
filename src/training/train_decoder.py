import torch
import torch_geometric as pyg
import os 
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader, NeighborLoader
from pytorch_lightning.loggers import CSVLogger

from ..data.knowledge_graph import HaunerGraph
from ..data.patients import PatientData
from ..plotting import plot_loss
# from ..configs import cfg_decode as args 
from .. import utils
from ..models.link_predictor import LinkPredictor
from ..inference.embed import embed_graph
from ..models.losses import ContrastiveLoss

pl.seed_everything(42)
# accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

args = utils.decoder_args()
pretrain_args = utils.load_args(os.path.join(args.model_path, "config.json")) # version_0...

Patients = PatientData(pretrain_args, discard_undiagnosed=True)
nodes_to_embed = Patients.nodes_of_labels(pretrain_args.select_labels)
embeddings = embed_graph(
    Patients.subgraph(nodes_to_embed), 
    model_class=LinkPredictor, 
    checkpoint_path=os.path.join(args.model_path, "checkpoints"), 
    device=device) 
Patients.assign_embeddings(embeddings, nodes_to_embed) 

# loss = ContrastiveLoss()
# loss(Patients.patients)
# train_loader = DataLoader(patients, batch_size=args.batch_size, shuffle=True)
# # VAL 
# model = GraphClassifier(
#     gat_model=gat,
#     num_classes=num_classes,
#     lr=1e-3,
# )

# csv_logger = CSVLogger("results/decode", name=args.save_as)
# trainer = Trainer(
#     accelerator=accelerator, 
#     devices=torch.cuda.device_count(), 
#     strategy="ddp", 
#     max_epochs=args.max_epochs, 
#     precision="16-mixed", # 
#     logger=csv_logger
# )

# trainer.fit(model, train_loader, val_loader)