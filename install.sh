#! /bin/bash

conda create -n genforce python=3.7
conda activate genforce
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
pip install -r requirements.txt
pip install -e ../adaiw