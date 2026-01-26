#!/bin/sh
#
#SBATCH --account=core-kind     # use this account as well
#SBATCH --gres=gpu:b200:1     # gpu:type:number a100_10gb
#SBATCH --mem=64GB
#SBATCH --cpus-per-gpu=1
#SBATCH --partition=jobs-gpu-long #testing #
#SBATCH -o logs/pretrain%j.log
#SBATCH -e logs/pretrain%j.err
#SBATCH -J pretrain       # job name

CUDA_LAUNCH_BLOCKING=1 singularity exec --nv container/container_pyg.sif python src/pretrain.py