import torch
import os
import pandas as pd
import numpy as np

def load_graph(path, device='cpu', embedding=None):
    file_name = os.path.join(path, 'knowledge_graph_patients.pt')
    print(f"Loading Knowledge graph from {os.path.abspath(file_name)}\n")
    
    KG = torch.load(file_name, weights_only=False, map_location=device)
    print("Done!")

    if embedding:
        KG['x'] = load_embedding(embedding_type=embedding, num_nodes=KG.num_nodes, path=path)

    return KG

def load_embedding(embedding_type, num_nodes, path):
    assert embedding_type in ['fastrp', 'node2vec'], f"{embedding_type} is not available as node embedding"
    
    print(f"Loading {embedding} node embeddings...")

    file_extension = '.parquet'
    file_name = os.path.join(path, embedding_type + file_extension)
    df = pd.read_parquet(file_name)
    available_embeddings = torch.tensor(list(df['embedding'])).float()
    emb_size = available_embeddings.shape[1:] 
    embedding = torch.full((num_nodes, *emb_size), float('nan'))
    embedding[df['pyg_id']] = available_embeddings 

    print("Done!")

    return embedding