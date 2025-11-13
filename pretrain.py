import torch
import torch_geometric as pyg
import pandas as pd

def load_KG(path):
    KG = torch.load(path)
    return KG

def main():
    KG_path = './pyG_graph_data/knowledge_graph_patients.pt'
    KG = load_KG(KG_path)
    print(KG['x'])

if __name__ == "__main__":
    main()