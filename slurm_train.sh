#!/bin/bash
#SBATCH --nodes 1              # Request 2 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:4          # Request 2 GPU "generic resources”. You will get 2 per node.

#SBATCH --tasks-per-node=4   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      
#SBATCH --time=0-00:20
#SBATCH --output=%N-%j.out

CONFIG=configs/stylegan_lsunchurch256.py
WORK_DIR=work_dirs/stylegan_lsunchurch256_train
PY_ARGS=${@:5}

module load python/3.7
source $HOME/genforce-env/bin/activate

echo $SLURM_PROCID
echo $SLURM_NTASKS
echo $SLURM_NODELIST

srun --kill-on-bad-exit=1 python -u ./train.py ${CONFIG} --work_dir=${WORK_DIR} --launcher="slurm" --adain_type BlockwiseAdaIN