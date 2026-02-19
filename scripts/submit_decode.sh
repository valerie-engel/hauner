#!/bin/sh
#
#SBATCH --account=core-kind     # use this account as well
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100_10gb:1    # gpu:type:number
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=testing #jobs-gpu #jobs-cpu #
#SBATCH -o logs/decode%j.log
#SBATCH -e logs/decode%j.err
#SBATCH -J decode       # job name

CUDA_LAUNCH_BLOCKING=1 singularity exec --nv container/container_pyg.sif python -m src.training.train_decoder
