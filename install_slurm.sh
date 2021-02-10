#! /bin/bash

set -e

module load python/3.7

virtualenv --no-download ~/genforce-env
source $HOME/genforce-env/bin/activate


# get inception for fid metric
FILE=$HOME/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth
if [ -f "$FILE" ]; then
    echo "$FILE already exists."
else 
    wget https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth -P $HOME/.cache/torch/hub/checkpoints
fi

# conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
pip install -r requirements_slurm.txt
pip install -e ../adaiw
pip install --no-index torch torchvision 
