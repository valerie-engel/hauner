import torch
import os
import pandas as pd
import json
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from .utils import get_complement, remap_to_consecutive     #put in src.utils...? or just here?


class HaunerGraph(Data):
    def __init__(
        self, 
        path, 
        device='cpu', 
        undirected=False, 
        embedding=None, 
        select_labels=None, 
        drop_selected_labels=False, 
        drop_unembedded=False
        ):
        super().__init__()
        self.path = path 
        self.device = device

        graph = self.load_graph() 
        self.edge_index, self.y, self.edge_label, self.num_nodes = graph.edge_index, graph.y.long(), graph.edge_label, graph.num_nodes
        self.labels = self.load_labels()
        
        if embedding:   # KEEP THIS...? 
            self.load_embedding(embedding_type=embedding, drop_unembedded=drop_unembedded)

        # only keep nodes of selected labels
        if drop_selected_labels:
            select_labels = get_complement(select_labels, self.labels)
        if select_labels:
            print(f"Only keeping nodes of labels {select_labels}")
            selection = self.nodes_of_labels(select_labels)
            self.substitute_by_subgraph(keep_nodes=selection)

        if undirected:
            print("Making graph undirected")
            self.edge_index = to_undirected(self.edge_index)

        self.make_y_consecutive()


    def load_graph(self):
        file_name = os.path.join(self.path, 'knowledge_graph_patients.pt')
        print(f"Loading data from {os.path.abspath(file_name)}")
        graph = torch.load(file_name, weights_only=False, map_location=self.device)
        # print(data) #Data(edge_index=[2, 209220742], y=[14730350], edge_label=[209220742], num_nodes=14730350)
        print("Done!\n")
        return graph

    def load_labels(self):
        with open(os.path.join(self.path, 'node_labels.json')) as f:
            ids_to_labels = json.load(f) 
        return [eval(label)[0] for label in ids_to_labels.values()]


    def load_embedding(self, embedding_type, drop_unembedded=False):
        assert embedding_type in ['fastrp', 'node2vec'], f"{embedding_type} is not available as node embedding"
        
        print(f"Loading {embedding_type} node embeddings...")
        file_extension = '.parquet'
        file_name = os.path.join(self.path, embedding_type + file_extension)
        df = pd.read_parquet(file_name)
        available_embeddings = torch.tensor(list(df['embedding'])).float().to(device=self.device)
        ids_with_embedding = torch.tensor(df['pyg_id']).to(device=self.device)
        self.assign_embeddings(available_embeddings, ids_with_embedding)
        print("Done!\n")
        if drop_unembedded:
            print("Dropping all nodes without embedding")
            self.substitute_by_subgraph(keep_nodes=ids_with_embedding)


    def assign_embeddings(self, available_embeddings, node_ids):
        emb_size = available_embeddings.shape[1:] 
        embedding = torch.full((self.num_nodes, *emb_size), float('nan'))
        embedding[node_ids] = available_embeddings 
        self.x = embedding.to(device=self.device)


    def substitute_by_subgraph(self, keep_nodes):
        # use .subgraph() instead?
        # subgraph edges
        self.y = self.y[keep_nodes]
        if self.x:
            self.x = self.x[keep_nodes]
        self.edge_index, self.edge_label = subgraph(
            keep_nodes, 
            self.edge_index, 
            edge_attr=self.edge_label,
            relabel_nodes=True  # REALLY? IF I NEED TO STORE NODE_IDS INSTEAD, MIGHT JUST NOT...
        )
        self.num_nodes = int(sum(keep_nodes)) if keep_nodes.dtype == torch.bool else len(keep_nodes)
        # MAKE CONSECUTIVE?  
        print(f"Remaining: {self.num_nodes} nodes and {self.num_edges} edges\n")


    def nodes_of_labels(self, labels):
        label_ids = self.ids_of_labels(labels)
        return (torch.isin(self.y, label_ids)).nonzero(as_tuple=True)[0]
        

    def ids_of_labels(self, labels):
        if not isinstance(labels, list):
            labels = [labels] 
        assert all([label in self.labels for label in labels]), f"Not all labels in {labels} could be found"
        
        label_ids = [self.labels.index(label) for label in labels]
        return torch.tensor(label_ids, device=self.device)


    def make_y_consecutive(self):
        print("Discarding unused node labels...")
        self.y, label_ids = remap_to_consecutive(self.y)
        self.labels = [self.labels[i] for i in label_ids.tolist()]
        print(f"Remaining node labels: {self.labels}")


    # def get_neo4j_id(self, ids, neo4j_ids=None, full_graph_ids=None):
    #     if full_graph_ids:
    #         ids = full_graph_ids[ids]

    #     if neo4j_ids is None:
    #         neo4j_ids = torch.load(os.path.join(cfg.path, 'neo4j_ids.pt'))
            
    #     return neo4j_ids[ids]

# def import_knowledge_graph(path, device='cpu', undirected=False, embedding=None, select_labels=None, drop_selected_labels=False, drop_unembedded=False):
#     file_name = os.path.join(path, 'knowledge_graph_patients.pt')
#     KG = load_graph(file_name, device)

#     # just storing list of labels would be a lot more elegant...
#     types_to_labels = load_types_to_labels_map(path)    # maps numeric node_types to string node labels
#     if embedding:
#         KG.x, ids_with_embedding = load_embedding(embedding_type=embedding, num_nodes=KG.num_nodes, path=path, device=device)
#         if drop_unembedded:
#             KG = get_subgraph(KG, keep_nodes=ids_with_embedding)
#             types = KG.y.unique()
#             labels = map_types_to_labels(types, types_to_labels)
#             print(f"Remaining node labels: {labels}")

#     # only keep nodes of selected labels
#     if drop_selected_labels:
#         select_labels = get_complement(select_labels, list(types_to_labels.values()))
#     if select_labels:
#         KG, full_graph_ids = get_subgraph_with_labels(KG, select_labels, types_to_labels, device=device)

#     if undirected:
#         print("Making graph undirected")
#         KG.edge_index = to_undirected(KG.edge_index)

#     KG.y, types_to_labels = make_types_consecutive(KG.y, types_to_labels)
#     return KG, types_to_labels #, full_graph_ids

