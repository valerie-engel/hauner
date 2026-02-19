#!/bin/sh
#
#SBATCH --account=core-kind     # use this account as well
#SBATCH --gres=gpu:a100:2     # gpu:type:number 
#SBATCH --mem=64GB
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=jobs-gpu-long #testing #
#SBATCH -o logs/pretrain%j.log
#SBATCH -e logs/pretrain%j.err
#SBATCH -J pretrain       # job name

CUDA_LAUNCH_BLOCKING=1 singularity exec --nv container/container_pyg.sif python -m src.pretrain