#!/bin/bash
#
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --account=core-kind
#SBATCH -o logs/build_container%j.log
#SBATCH -e logs/build_container%j.err
#SBATCH -J build_container


#### Adapt variables here ####

IMAGE_PATH="./nightly.sif" # could also be something like IMAGE_PATH=/data/core-xxx/containers/...
DEFINITION_FILE_PATH="./nightly.def"

#### End definitions ####


## Prepare build environments ##

# Within tmpfs container, we are properly isolated from other users
# Fakeroot build need write permissions for namespace-mapped UIDs
chmod 777 /tmp

export SINGULARITY_CACHEDIR=/tmp/.sing_cache
export SINGULARITY_TMPDIR=/tmp/.sing_tmp

mkdir -p $SINGULARITY_TMPDIR

## Build the container
singularity build --fakeroot --nv /tmp/mycontainer.sif ${DEFINITION_FILE_PATH}

mv /tmp/mycontainer.sif ${IMAGE_PATH}
