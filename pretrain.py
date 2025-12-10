import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import LinkNeighborLoader
from data.utils import load_graph
import cfg
# from models import GAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    KG = load_graph(path=cfg.KG_path, device=device)
    print(KG.keys())
    pl.seed_everything(42)
    train_loader = LinkNeighborLoader(KG, [10], batch_size=2, edge_label_index=KG.edge_index, neg_sampling='triplet') # binary?
    sampled_data = next(iter(train_loader))
    # for k, v in sampled_data:
    #     print(k,v)
    print(sampled_data)
    # print(KG.keys())
    # model = GAT(cfg)
    # trainer = Trainer(accelerator=device)
    # trainer.fit(model)


if __name__ == "__main__":
    main()
