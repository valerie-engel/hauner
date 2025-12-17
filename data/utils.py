import torch
import os
import pandas as pd
import json
from torch_geometric.utils import to_undirected


def load_graph(path, device='cpu', undirected=False, embedding=None, drop_labels=None):
    file_name = os.path.join(path, 'knowledge_graph_patients.pt')
    print(f"Loading Knowledge graph from {os.path.abspath(file_name)}")
    
    graph = torch.load(file_name, weights_only=False, map_location=device)
    print("Done!\n")

    if drop_labels:
        graph = drop_labels(graph, drop_labels, path)

    if undirected:
        print("Making graph undirected")
        graph.edge_index = to_undirected(graph.edge_index)

    if embedding:
        graph['x'] = load_embedding(embedding_type=embedding, num_nodes=graph.num_nodes, path=path)

    return graph

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

    print("Done!\n")

    return embedding

## more elegant to do this from parquet files...? Or directly when downloading?
# def drop_labels(graph, labels, path):
#     types = map_labels_to_types(labels, path)
#     keep_nodes_mask = torch.isin(graph.y, types)

#     # nodes_to_discard = discard_mask.nonzero(as_tuple=False).view(-1)
    
def map_labels_to_types(labels, path):
    # adapt to new filename, save dict other way around and saving without [\\]
    types_to_labels = json.load(os.path.join(path, 'node_labels.json')) 
    types = []
    for t, label in types_to_labels.items():
        label = eval(label)[0]  # to strip string of [\\]
        if label in labels:
            types.append(eval(t))
    assert len(types) == len(labels), f"Not all labels in {labels} could be translated to types"
    return types

