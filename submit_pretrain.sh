#!/bin/sh
#
#SBATCH --account=core-kind     # use this account as well
#SBATCH --gres=gpu:b200:2       # gpu:type:number
#SBATCH --mem=128GB
#SBATCH --cpus-per-gpu=2
#SBATCH --partition=jobs-gpu-long  # which queue?
#SBATCH -o logs/pretrain%j.log
#SBATCH -e logs/pretrain%j.err
#SBATCH -J pretrain       # job name

singularity exec --nv container/containerB200.sif python pretrain.py