#! /bin/bash

set -e

module load python/3.7

virtualenv --no-download ~/genforce-env
source $HOME/genforce-env/bin/activate

# conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
pip install -r requirements_slurm.txt
pip install -e ../adaiw
pip install --no-index torch torchvision 
