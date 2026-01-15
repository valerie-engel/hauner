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

singularity exec --nv container/container.sif python pretrain.py