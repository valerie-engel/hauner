import torch
import os

def get_neo4j_id(ids, path, full_graph_ids=None):
    if full_graph_ids:
        ids = full_graph_ids[ids]
    neo4j_ids = torch.load(os.path.join(path, 'neo4j_ids.pt'))
    return neo4j_ids[ids]
