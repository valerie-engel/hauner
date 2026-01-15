#!/bin/sh
#
#SBATCH --account=core-kind     # use this account as well
#SBATCH --gres=gpu:a100_10gb:1       # gpu:type:number
#SBATCH --mem=16GB
#SBATCH --time=00:05:00
#SBATCH --partition=testing  # which queue?
#SBATCH -o logs/slurm%j.log
#SBATCH -e logs/slurm%j.err
#SBATCH -J pretrain       # job name

# singularity exec --nv container/container.sif bash -c "
# source /opt/conda/etc/profile.d/conda.sh
# conda activate hauner
# python - << 'EOF'
# import torch_sparse
# from torch_geometric.loader import NeighborSampler
# print('NeighborSampler OK')
# EOF
# "


# singularity exec --nv container/container2.sif bash -c "
# source /opt/conda/etc/profile.d/conda.sh
# conda activate hauner
# export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
# python pretrain.py
# "

#/opt/conda/bin/conda run -n hauner 

singularity exec --nv container/container.sif python pretrain.py