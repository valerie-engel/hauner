import os
from torch_geometric.utils import to_undirected
from data.utils import *

def import_patients(path, num_hops=0, device='cpu', undirected=False, embedding=None, drop_labels=None, drop_unembedded=False):
    KG, types_to_labels = import_knowledge_graph(
        path=path, 
        device=device, 
        undirected=undirected, 
        embedding=embedding, 
        drop_labels=drop_labels, 
        drop_unembedded=drop_unembedded)

    patients = get_nodes_of_labels(KG, ['Biological_sample'], types_to_labels, device)
    print(f"Found {len(patients)} biological samples")
    subgraphs = [extract_k_hops(KG, num_hops, center_nodes=[patient])[0] for patient in patients]   # keep mappings?
    subgraph_sizes = [list(subgraph.num_nodes, subgraph.num_edges) for subgraph in subgraphs]
    print(f"Patient subgraph sizes:")
    node_counts = subgraph_sizes[:][0]
    print(f"node counts:\n smallest: {min(node_counts)}, average: {sum(node_counts)/len(node_counts)}, largest: {max(node_counts)}")
    edge_counts = subgraph_sizes[:][1]
    print(f"edge counts:\n smallest: {min(edge_counts)}, average: {sum(edge_counts)/len(edge_counts)}, largest: {max(edge_counts)}")
    return subgraphs


def import_knowledge_graph(path, device='cpu', undirected=False, embedding=None, drop_labels=None, drop_unembedded=False):
    file_name = os.path.join(path, 'knowledge_graph_patients.pt')
    KG = load_graph(file_name, device)

    types_to_labels = load_types_to_labels_map(path)    # maps numeric node_types to string node labels
    if embedding:
        KG.x, ids_with_embedding = load_embedding(embedding_type=embedding, num_nodes=KG.num_nodes, path=path, device=device)
        if drop_unembedded:
            KG = get_subgraph(KG, keep_nodes=ids_with_embedding)
            types = KG.y.unique()
            labels = map_types_to_labels(types, types_to_labels)
            print(f"Remaining node labels: {labels}")

    if drop_labels:
        KG, full_graph_ids = drop_nodes_by_label(KG, drop_labels, types_to_labels, device)

    if undirected:
        print("Making graph undirected")
        KG.edge_index = to_undirected(KG.edge_index)

    KG.y, types_to_labels = make_types_consecutive(KG.y, types_to_labels)
    return KG, types_to_labels #, full_graph_ids

