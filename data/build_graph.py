import json
import pandas as pd
import torch
from torch_geometric.data import Data
import os


def read_parquet_shards_to_df(directory):
    print(f"Reading Dataset from {directory}")

    dfs = [pd.read_parquet(os.path.join(directory, f)) for f in sorted(os.listdir(directory)) if f.endswith(".parquet")]
    df = pd.concat(dfs, ignore_index=True)
    return df


def save_graph(graph, nids, node_label_to_type, edge_label_to_type, fastrp, node2vec, path):    
    def save_json(data, file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_parquet(df, file_name):
        df.to_parquet(path=file_name)

    torch.save(graph, os.path.join(path, 'knowledge_graph_patients.pt'))
    torch.save(nids, os.path.join(path, 'neo4j_ids.pt'))
    save_json(node_label_to_type, os.path.join(path, 'node_label_to_type.json'))
    save_json(edge_label_to_type, os.path.join(path, 'edge_label_to_type.json'))
    # save_json(pyg_to_nid, os.path.join(path, 'pyg_to_nid.json'))
    save_parquet(fastrp, os.path.join(path, 'fastrp.parquet'))
    save_parquet(node2vec, os.path.join(path, 'node2vec.parquet'))
    print(f"Graph with accompanying index maps and node embeddings have been saved to {os.path.abspath(path)}")

    
def remap_edges(nids, edge_index):
    # i is new id of i-th node by definition as nodes.index
    # src_ids = torch.tensor(src_ids)
    # dst_ids = torch.tensor(dst_ids)
    print("Mapping edge index to pyg indices...")

    max_id = int(nids.max().item())
    table = torch.full((max_id + 1,), -1, dtype=torch.long) #, device=nids.device
    table[nids] = torch.arange(len(nids))    #, device=nids.device
    edge_index_new = table[edge_index]

    # edge_index_new = edge_index.detach().clone()
    # for i, nid in enumerate(nids):
    #     edge_index_new[edge_index == nid] = i 
    #     # src_ids[src_ids == nid] = i
    #     # dst_ids[dst_ids == nid] = i
    
    # assert all(nids[edge_index_new] == edge_index), "Not all edges have been mapped to pyg indices correctly"
    # assert all(src_ids < len(nids)) and all(dst_ids < len(nids)), ""
    return edge_index_new


def map_labels_to_types(labels):
    print("Mapping label strings to integer types...")

    types, labels = pd.factorize(labels)
    label_to_type = dict(zip(list(range(len(labels))), labels))
    # all_labels = set(labels)  # all unique types
    # label_to_type = {l: i for i, l in enumerate(all_labels)}
    # types = list(labels)
    # for i, t in enumerate(types):
    #     types[i] = label_to_type[t]

    # assert all(type(item) is int for item in types), "Not all types could be converted to integer labels"
    return torch.tensor(types), label_to_type


def build_embedding_df(props, embedding_type):
    assert embedding_type in ['fastrp', 'node2vec'], "{embedding_type} is not available as node embedding"

    print(f"Extracting {embedding_type} embeddings...")

    has_emb = props.str.contains(embedding_type, regex=False)
    props_with_emb = props.loc[has_emb]
    ids = props_with_emb.index # still uses index of original df -> fine
    props_with_emb = props_with_emb.apply(json.loads)
    embeddings = [eval(prop[embedding_type]) for prop in props_with_emb] #props_with_emb[embedding_type]
    # embeddings = [None] * has_emb.sum()
    # for i, prop in enumerate(props_with_emb):
    #     prop = json.loads(prop)
    #     embedding = prop[embedding_type]
    #     embeddings[i] = embedding

    df = pd.DataFrame({'pyg_id': ids, 'embedding': embeddings})
    return df


def load_all_parquet_files(parquet_dir):
    node_dir = os.path.join(parquet_dir, 'nodes')
    edge_dir = os.path.join(parquet_dir, 'edges')

    nodes = read_parquet_shards_to_df(node_dir)
    print(f"Imported {len(nodes)} nodes")
    edges = read_parquet_shards_to_df(edge_dir)
    print(f"Imported {len(edges)} edges")
    return nodes, edges


def build_graph_from_df(nodes, edges): 
    print("Processing nodes: ")
    nids = torch.tensor(nodes.nid)   # pyg indices will be consecutive -> nids[pyg_index] returns correct nid
    # pyg_to_nid = dict(zip(nodes.index, nodes.nid))  # pyg indices will be consecutive aka. nodes.index 
    # convention here: labels are stings, types numeric class indices
    node_types, node_label_to_type = map_labels_to_types(nodes.labels)
    print(f"Found {len(node_label_to_type)} unique node types")
    # build embedding dfs
    fastrp = build_embedding_df(nodes.props, 'fastrp')
    node2vec = build_embedding_df(nodes.props, 'node2vec')

    print("Processing edges: ")
    edge_index = torch.tensor((edges.src, edges.dst))
    edge_index = remap_edges(nids, edge_index)
    edge_types, edge_label_to_type = map_labels_to_types(edges.etype)
    print(f"Found {len(edge_label_to_type)} unique edge types")

    graph = Data(y=node_types, edge_index=edge_index, edge_label=edge_types, num_nodes=len(nodes))
    return graph, nids, node_label_to_type, edge_label_to_type, fastrp, node2vec


# def build_subgraph(parquet_dir):
#     nodes_file = os.path.join(parquet_dir, "nodes/nodes_00000.parquet")
#     edges_file = os.path.join(parquet_dir, "edges/edges_00000.parquet")

#     nodes = pd.read_parquet(nodes_file)
#     print(f"Imported {len(nodes)} nodes")
#     edges = pd.read_parquet(edges_file)
#     print(f"Imported {len(edges)} edges")

#     return build_graph_from_df(nodes, edges)


def build_full_graph(parquet_dir):
    nodes, edges = load_all_parquet_files(parquet_dir)
    return build_graph_from_df(nodes, edges)

parquet_dir = "/home/vagrant/dev/variant_ranking/parquet_export"
out_dir = "data/knowledge_graph"
graph, pyg_to_nid, node_label_to_type, edge_label_to_type, fastrp, node2vec = build_full_graph(parquet_dir)


