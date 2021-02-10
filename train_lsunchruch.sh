#! /bin/bash

source $HOME/genforce-env/bin/activate

GPUS=2
CONFIG=configs/stylegan_lsunchruch256.py
WORK_DIR=work_dirs/stylegan_lsunchruch256_train
./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR}
