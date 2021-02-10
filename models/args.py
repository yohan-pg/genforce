import argparse
from utils.misc import DictAction

parser = argparse.ArgumentParser(description='Run model training.')
parser.add_argument('config', type=str,
                    help='Path to the training configuration.')
parser.add_argument('--work_dir', type=str, required=True,
                    help='The work directory to save logs and checkpoints.')
parser.add_argument('--resume_path', type=str, default=None,
                    help='Path to the checkpoint to resume training.')
parser.add_argument('--weight_path', type=str, default=None,
                    help='Path to the checkpoint to load model weights, '
                            'but not resume other states.')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed. (default: %(default)s)')
parser.add_argument('--launcher', type=str, default='pytorch',
                    choices=['pytorch', 'slurm'],
                    help='Launcher type. (default: %(default)s)')
parser.add_argument('--backend', type=str, default='nccl',
                    help='Backend for distributed launcher. (default: '
                            '%(default)s)')
parser.add_argument('--rank', type=int, default=-1,
                    help='Node rank for distributed running. (default: '
                            '%(default)s)')
parser.add_argument('--local_rank', type=int, default=0,
                    help='Rank of the current node. (default: %(default)s)')

parser.add_argument('--adain_type', type=str, 
                    help='type of adain normalization')
parser.add_argument('--block_size', type=int, default=64, 
                    help='block size if whitening is used')

parser.add_argument('--options', nargs='+', action=DictAction,
                    help='arguments in dict')

args = parser.parse_args()