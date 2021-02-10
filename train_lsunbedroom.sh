#! /bin/bash

GPUS=1
CONFIG=configs/stylegan_lsunbedroom256.py
WORK_DIR=work_dirs/stylegan_lsunbedroom256_train
./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR}
