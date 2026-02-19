import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import LinkNeighborLoader
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path

from ..data.knowledge_graph import HaunerGraph
from ..models.link_predictor import LinkPredictor
from ..plotting import plot_loss
# from ..configs import cfg_pretrain as args
from ..utils import pretrain_args

pl.seed_everything(42)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu' 

args = pretrain_args()
KG = HaunerGraph(
    path=args.KG_path, 
    device='cpu', 
    undirected=args.undirected, 
    embedding=args.embedding, 
    select_labels=args.select_labels, 
    drop_selected_labels=args.drop_selected_labels,
    drop_unembedded=args.drop_unembedded
    ) 

# put somewhere else
# num_neighbors = [args.num_sampled_neighbors//2] * (n_layers - 1)
# num_neighbors.insert(0, args.num_sampled_neighbors)
train_loader = LinkNeighborLoader(
    KG, 
    args.num_neighbors, #[10, 5, 5]
    batch_size=args.batch_size, 
    edge_label_index=KG.edge_index, 
    neg_sampling='triplet', 
    shuffle=True
    ) # binary?
train_loader.num_neighbors = [10, 5, 5]
# sampled_data = next(iter(train_loader))

num_node_types = len(KG.y.unique()) 
in_channels = args.in_channels if KG.x is None else KG.x.shape[-1]
model = LinkPredictor(
    num_types=num_node_types, 
    n_layers=args.n_layers, 
    in_channels=in_channels, 
    hidden_channels=args.hidden_channels, 
    out_channels=args.out_channels, 
    n_heads=args.n_heads, 
    dropout=args.dropout, 
    existing_embedding=args.embedding,
    residual=args.residual, 
    margin=args.margin
    )
# x = model.embedding(sampled_data.y).cuda()
# print(model(sampled_data.edge_index.cuda(), sampled_data.src_indexcuda(), sampled_data.dst_pos_indexcuda()))

out_path = Path("results/pretrain")
csv_logger = CSVLogger(out_path, name=args.save_as)
trainer = Trainer(
    accelerator=accelerator, 
    devices=torch.cuda.device_count(), 
    strategy="ddp", 
    max_epochs=args.max_epochs, 
    precision="16-mixed", # 
    logger=csv_logger
    )
trainer.fit(model, train_loader)

#path should point to right version!
utils.save_args(args, out_path / Path(args.save_as))
plot_loss(csv_logger.log_dir)
# immediately save embeddings? where? 

