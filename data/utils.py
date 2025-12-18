import torch
import os
import cfg

def get_neo4j_id(ids, neo4j_ids=None, full_graph_ids=None):
    if full_graph_ids:
        ids = full_graph_ids[ids]

    if neo4j_ids is None:
        neo4j_ids = torch.load(os.path.join(cfg.path, 'neo4j_ids.pt'))
        
    return neo4j_ids[ids]
