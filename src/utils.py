import torch
import argparse
from pathlib import Path
import json
import os 

def pretrain_args():
    parser = argparse.ArgumentParser(description="Pretraining config")

    # Data
    parser.add_argument("--KG_path", type=str, default="data/knowledge_graph")
    parser.add_argument("--select_labels", type=str, nargs='+',
                        default=['Disease', 'Phenotype', 'Gene', 'Protein'])
    parser.add_argument("--drop_selected_labels", action="store_true", help="If set, nodes of all labels except selected will appear")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--embedding", type=str, choices=["fastrp", 'node2vec', None], default=None)
    parser.add_argument("--drop_unembedded", action="store_true")

    # Model
    parser.add_argument("--in_channels", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--margin", type=float, default=1)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_sampled_neighbors", type=int, nargs='+', default=[10, 5, 5])

    # experiment management
    parser.add_argument("--save_as", type=str, default="pretrain")

    args = parser.parse_args()
    print("Training with args:", args)
    
    return args


def decoder_args():
    parser = argparse.ArgumentParser(description="Decoder config")

    # Embedder model
    parser.add_argument("--model_path", type=str, default="results/pretrain/small_GAT")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=20)

    args = parser.parse_args()
    print("Training with args:", args)
    
    return args


def save_args(args, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = Path(path) / Path("config.json")
    with open(file_name, "w") as f:
        json.dump(vars(args), f)


def load_args(path):
    file_name = Path(path) / Path("config.json")
    with open(file_name, "r") as f:
        data = json.load(f)
    args = argparse.Namespace(**data)
    return args


def load_model(model_class, checkpoint_path, device):
    checkpoint_dir = Path(checkpoint_path)
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    assert len(checkpoint_files) == 1, f"Expected 1 checkpoint, found {len(checkpoint_files)}"

    model = model_class.load_from_checkpoint(checkpoint_files[0], map_location=device)
    return model.eval()    

