import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training config")

    # Data
    parser.add_argument("--KG_path", type=str, default="data/knowledge_graph")
    parser.add_argument("--drop_labels", type=str, nargs='+',
                        default=['Biological_sample', 'Subject', 'Chromosome', 'Publication', 'GWAS_study', 'Project'])

    # Model
    parser.add_argument("--in_channels", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--margin", type=float, default=1)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_sampled_neighbors", type=int, default=10)

    args = parser.parse_args()
    print("Training with args:", args)
    
    return args


# def get_devices(num_gpu):
#     if num_gpu > 1:
#         graph_device = 'cpu'  # keep graph on the CPU for distributed training
#     else:
#         graph_device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#     accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
#     print(f"Accelerator: {accelerator}, Graph on device: {graph_device}")
#     return accelerator, graph_device