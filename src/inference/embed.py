import torch 
from torch_geometric.loader import NeighborLoader
from ..configs import cfg_pretrain as cfg ## REPLACE THIS BY LOADING STORED ARGS!!
from ..utils import load_model


def embed_graph(graph, model_class, checkpoint_path, device='cpu'):
    print(f"Embedding graph with {model_class}...")
    loader = NeighborLoader(
        graph, #.edge_index
        num_neighbors=[10, 5, 5],  # number of neighbors per layer
        batch_size=50000,
        shuffle=False,
        num_workers=4
    )

    # checkpoint path in cfg 
    embedder = load_model(model_class=model_class, checkpoint_path=checkpoint_path, device=device)
    embedder.freeze()

    all_node_embeddings = []
    for batch in loader:
        # abstract embedder class?
        batch = batch.to(device) 
        out = embedder.embedding(batch.y, batch.x)
        out = embedder.encoder(out, batch.edge_index)[:batch.batch_size]    # first batch_size nodes are target nodes 
        all_node_embeddings.append(out.cpu())  # move to CPU to save GPU memory
    print("Done!")
    return torch.cat(all_node_embeddings, dim=0)

