
import torch
import os
import pandas as pd
import json
from torch_geometric.utils import to_undirected, k_hop_subgraph
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.data import Data

def import_patients(path, num_hops=0, device='cpu', undirected=False, embedding=None, drop_labels=None, drop_unembedded=False):
    KG, types_to_labels = import_knowledge_graph(
        path=path, 
        device=device, 
        undirected=undirected, 
        embedding=embedding, 
        drop_labels=drop_labels, 
        drop_unembedded=drop_unembedded)
    # return a set of subgraphs
    patients = get_nodes_of_labels(KG, ['Biological_sample'], types_to_labels, device)
    subgraphs = [extract_k_hops(KG, num_hops, center_nodes=[patient])[0] for patient in patients]   # keep mappings?
    return subgraphs


def extract_k_hops(graph, k, center_nodes):
    # compute keep nodes 
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, k, graph.edge_index, relabel_nodes=True)
    # or use get_subgraph logic...? This assumes exact attributes
    subgraph = Data(
        x=graph.x[subset],
        edge_index=edge_index,
        edge_labels=graph.edge_labels[edge_mask],
        y=graph.y[subset]   # keep node labels
    )
    return subgraph, mapping

    

def import_knowledge_graph(path, device='cpu', undirected=False, embedding=None, drop_labels=None, drop_unembedded=False):
    file_name = os.path.join(path, 'knowledge_graph_patients.pt')
    KG = load_graph(file_name, device)

    # I NEED TO ADAPT THIS FOR WHEN DROPPING LABELS AND MAKING CONSECUTIVE AGAIN
    types_to_labels = load_types_to_labels_map(path)    # maps numeric node_types to string node labels
    if embedding:
        KG.x, ids_with_embedding = load_embedding(embedding_type=embedding, num_nodes=KG.num_nodes, path=path, device=device)
        if drop_unembedded:
            KG = get_subgraph(KG, keep_nodes=ids_with_embedding)
            types = KG.y.unique()
            labels = map_types_to_labels(types, types_to_labels)
            # THIS RETURNS DIFFERENT LABELS ON CLUSTER!!
            print(f"Remaining node labels: {labels}")

    if drop_labels:
        KG, full_graph_ids = drop_nodes_by_label(KG, drop_labels, types_to_labels, device)

    if undirected:
        print("Making graph undirected")
        KG.edge_index = to_undirected(KG.edge_index)

    KG.y = remap_to_consecutive(KG.y)
    return KG, types_to_labels #, full_graph_ids


def load_graph(file_name, device='cpu'):
    print(f"Loading data from {os.path.abspath(file_name)}")
    graph = torch.load(file_name, weights_only=False, map_location=device)
    print("Done!\n")
    return graph 


def load_embedding(embedding_type, num_nodes, path, device='cpu'):
    assert embedding_type in ['fastrp', 'node2vec'], f"{embedding_type} is not available as node embedding"
    
    print(f"Loading {embedding_type} node embeddings...")

    file_extension = '.parquet'
    file_name = os.path.join(path, embedding_type + file_extension)
    df = pd.read_parquet(file_name)
    available_embeddings = torch.tensor(list(df['embedding'])).float().to(device=device)
    ids_with_embedding = torch.tensor(df['pyg_id']).to(device=device)

    emb_size = available_embeddings.shape[1:] 
    embedding = torch.full((num_nodes, *emb_size), float('nan'))
    embedding[ids_with_embedding] = available_embeddings 

    print("Done!\n")

    return embedding, ids_with_embedding

def get_nodes_of_labels(graph, labels, types_to_labels, device):
    types = map_labels_to_types(labels, types_to_labels, device)
    node_ids = (torch.isin(graph.y, types)).nonzero(as_tuple=True)[0]
    return node_ids

def keep_nodes_of_label(graph, labels, types_to_labels, device='cpu'):
    print(f"Only keeping nodes of labels {labels}")
    keep_nodes = get_nodes_of_labels(graph, labels, types_to_labels, device)
    subgraph = get_subgraph(graph, keep_nodes)
    # if not cfg.debug:
    #     torch.save(torch.where(keep_nodes_mask), os.path.join(path, )) ## SAVE SOMEWHERE IN RUN RESULTS...
    return subgraph, keep_nodes

## more elegant to do this from parquet files...? Or directly when downloading? yeah, but this works for us...
def drop_nodes_by_label(graph, drop_labels, types_to_labels, device='cpu'):
    print(f"Dropping all nodes of labels {drop_labels}")
    drop_types = map_labels_to_types(drop_labels, types_to_labels, device)
    # keep_types = Use this and get nodes of label fct.?
    keep_nodes = (~torch.isin(graph.y, drop_types)).nonzero(as_tuple=True)[0]
    subgraph = get_subgraph(graph, keep_nodes)
    # if not cfg.debug:
    #     torch.save(torch.where(keep_nodes_mask), os.path.join(path, )) ## SAVE SOMEWHERE IN RUN RESULTS...
    return subgraph, keep_nodes

def get_subgraph(graph, keep_nodes):
    # subgraph edges
    edge_index, edge_label, keep_edges = pyg_subgraph(
        keep_nodes, 
        graph.edge_index, 
        edge_attr=getattr(graph, 'edge_label', None),
        relabel_nodes=True,
        return_edge_mask=True
    )

    # number of subgraph nodes
    num_nodes = int(sum(keep_nodes)) if keep_nodes.dtype == torch.bool else len(keep_nodes)
    
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
    
    subgraph = Data(**subgraph_dict)
    print(f"Remaining: {subgraph.num_nodes} nodes and {subgraph.num_edges} edges\n")
    return subgraph    # on correct device?


def load_types_to_labels_map(path): 
    # adapt to new filename, save dict other way around and saving without [\\]
    with open(os.path.join(path, 'node_labels.json')) as f:
        types_to_labels = json.load(f) 
    return types_to_labels


def map_labels_to_types(labels, types_to_labels, device):
    types = []
    for t, label in types_to_labels.items():
        label = eval(label)[0]  # to strip string of [\\]
        if label in labels:
            types.append(eval(t))

    assert len(types) == len(labels), f"Not all labels in {labels} could be translated to types. Please check spelling"
    return torch.tensor(types).to(device=device)


def map_types_to_labels(types, types_to_labels):
    types = types.to(device='cpu')
    labels = [types_to_labels[str(int(t))] for t in types]
    return labels   #types_to_ ???


def remap_to_consecutive(x): #, path, name
    unique = x.unique(sorted=True)
    if len(unique) == unique[-1] + 1:
        # universe is already consecutive
        return x

    # torch.save # when in doubt I can just reconstruct, so dont save unique ...
    remap = torch.empty(x.max() + 1, dtype=torch.long)
    remap[unique] = torch.arange(len(unique), device=x.device)
    return remap[x]


def get_neo4j_id(ids, neo4j_ids=None, full_graph_ids=None):
    if full_graph_ids:
        ids = full_graph_ids[ids]

    if neo4j_ids is None:
        neo4j_ids = torch.load(os.path.join(cfg.path, 'neo4j_ids.pt'))
        
    return neo4j_ids[ids]