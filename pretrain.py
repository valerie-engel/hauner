import torch
import torch_geometric as pyg
# import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.loader import LinkNeighborLoader

from data.import_data import import_knowledge_graph
import cfg
from models import LinkPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu' #torch.device()
print(device)


# def main():
KG = import_knowledge_graph(path=cfg.KG_path, device=device, drop_labels=['Biological_sample', 'Subject'])
print(KG.keys())
pl.seed_everything(42)
train_loader = LinkNeighborLoader(KG, [10] * cfg.num_layers, batch_size=2, edge_label_index=KG.edge_index, neg_sampling='triplet') # binary?
sampled_data = next(iter(train_loader))
# for k, v in sampled_data:
#     print(k,v)
print(sampled_data)
model = LinkPredictor(KG.num_nodes, cfg)
# print(model(sampled_data))

trainer = Trainer(accelerator=device, max_epochs=1)
trainer.fit(model, train_loader)


# if __name__ == "__main__":
#     main()
