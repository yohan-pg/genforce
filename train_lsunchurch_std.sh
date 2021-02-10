#! /bin/bash

#SBATCH --nodes=1
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:4               # Number of GPU(s) per node
#SBATCH --cpus-per-task=8          # CPU cores/threads
#SBATCH --mem=32000M            # memory per node
#SBATCH --time=3-00:00           # time (DD-HH:MM)

module load python/3.7
source $HOME/genforce-env/bin/activate

GPUS=4
CONFIG=configs/stylegan_lsunchurch256.py
WORK_DIR=work_dirs/stylegan_lsunchurch256_train_std
./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR} --adain_type StandardizationAdaIN
