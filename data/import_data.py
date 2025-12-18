
import torch
import os
import pandas as pd
import json
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.data import Data

# disentangling load_graph and load_patients is slightly slower, but more flexible... extra fct. that does both?
def import_patients(path, device='cpu'):
    file_name = os.path.join(path, 'knowledge_graph_patients.pt')
    graph = load_graph(file_name, device)
    # represent each biological sample as set of their neighbours in KG... as pyg Data?
    

def import_knowledge_graph(path, device='cpu', undirected=False, embedding=None, drop_labels=None):
    file_name = os.path.join(path, 'knowledge_graph_patients.pt')
    KG = load_graph(file_name, device)

    if embedding:
        KG['x'] = load_embedding(embedding_type=embedding, num_nodes=KG.num_nodes, path=path)

    if drop_labels:
        KG, full_graph_ids = drop_nodes_by_label(KG, drop_labels, path)

    if undirected:
        print("Making graph undirected")
        KG.edge_index = to_undirected(KG.edge_index)

    return KG #, full_graph_ids


def load_graph(file_name, device='cpu'):
    print(f"Loading data from {os.path.abspath(file_name)}")
    graph = torch.load(file_name, weights_only=False, map_location=device)
    print("Done!\n")
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


## more elegant to do this from parquet files...? Or directly when downloading? yeah, but this works for us...
def drop_nodes_by_label(graph, drop_labels, path):
    print(f"Dropping all nodes of labels {drop_labels}")

    drop_types = map_labels_to_types(drop_labels, path)
    keep_nodes_mask = ~(torch.isin(graph.y, drop_types))
    subgraph = filter_graph_by_nodes(graph, keep_nodes_mask)
    full_graph_ids = torch.nonzero(keep_nodes_mask)
    # if not cfg.debug:
    #     torch.save(torch.where(keep_nodes_mask), os.path.join(path, )) ## SAVE SOMEWHERE IN RUN RESULTS...
    return subgraph, full_graph_ids

    
def map_labels_to_types(labels, path):
    # adapt to new filename, save dict other way around and saving without [\\]
    with open(os.path.join(path, 'node_labels.json')) as f:
        types_to_labels = json.load(f) 

    types = []
    for t, label in types_to_labels.items():
        label = eval(label)[0]  # to strip string of [\\]
        if label in labels:
            types.append(eval(t))

    assert len(types) == len(labels), f"Not all labels in {labels} could be translated to types. Please check spelling"
    return torch.tensor(types)


def filter_graph_by_nodes(graph, keep_nodes):
    # subgraph edges
    edge_index, edge_label, keep_edges = subgraph(
        keep_nodes, 
        graph.edge_index, 
        edge_attr=getattr(graph, 'edge_label', None),
        relabel_nodes=True,
        return_edge_mask=True
    )

    # number of subgraph nodes
    num_nodes = sum(keep_nodes) if keep_nodes.dtype == torch.bool else len(keep_nodes)
    
    # create a new Data object, filtering all keys with values of length num_nodes or num_edges
    subgraph_dict = {'edge_index': edge_index, 'num_nodes': num_nodes, 'edge_label': edge_label}
    for key, value in graph.items():
        if key not in subgraph_dict.keys():
            if len(value) == graph.num_nodes:
                print(f"Filtering {key} by nodes")
                subgraph_dict[key] = value[keep_nodes]
            elif len(value) == graph.num_edges:
                print(f"Filtering {key} by edges")
                subgraph_dict[key] = value[keep_edges]
            else:
                print(f"Not filtering {key}")
                subgraph_dict[key] = value
    
    return Data(**subgraph_dict)