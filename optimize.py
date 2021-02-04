# python3.7
"""A simple tool to synthesize images with pre-trained models."""

import os
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn 
import time 
from typing import List, Tuple

from models import MODEL_ZOO
from models import build_generator
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
# from utils.visualizer import save_image
from torchvision.utils import save_image

from adaiw import AdaIN, Statistic

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize images with pre-trained models.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/synthesis/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--generate_html', type=bool_parser, default=True,
                        help='Whether to use HTML page to visualize the '
                             'synthesized results. (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--max_seconds', type=float, default=60.0,
                        help='How long to optimize for '
                             '(default: %(default)s)')
    return parser.parse_args()

args = parse_args()

from datasets import BaseDataset
from datasets import LocalIterDataloader

def sample_batch() -> torch.Tensor:
    resolution = 64
    val = dict(root_dir='data/demo.zip',
                data_format='zip',
                resolution=resolution)
    
    dataset = BaseDataset(**val)
    val_loader = LocalIterDataloader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                current_iter=0,
                repeat=1)

    return next(val_loader)['image'].cuda()

def optimize(
    generator: torch.nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    stats: torch.Tensor, 
    lr: float,
    criterion = torch.nn.MSELoss()
) -> Tuple[torch.Tensor, List[float]]:
    start = time.time()
    optimizer = torch.optim.Adam(stats, lr=lr, betas=(0.0, 0.999))

    losses = []
    pred = None

    save_image(target, os.path.join(work_dir, f'target.png'))

    i = 0
    while time.time() - start < args.max_seconds:
        i += 1
        optimizer.zero_grad()
        pred = generator(input.detach())['image']
        loss = criterion(pred, target)
        losses.append(loss.item())
        print(loss.item())
        loss.backward()
        optimizer.step()
        if i % 30 == 0:
            save_image(pred, os.path.join(work_dir, f'result_{i}.png'))
        
    return stats, losses

def postprocess(images: torch.Tensor) -> torch.Tensor:
    images = images.detach()
    images = (images + 1) / 2 
    return images

def create_working_directory() -> str:
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'optimize')
    os.makedirs(work_dir, exist_ok=True)
    return work_dir

def load_generator_from_checkpoint():
    args = parse_args()

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in '
                         f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    model_config.pop('url')

    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    generator.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['models']['generator_smooth'])
    generator = generator.cuda().eval()
    print(f'Finish loading checkpoint.')

    return generator

# Set random seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

generator = load_generator_from_checkpoint()
work_dir = create_working_directory()
targets = sample_batch()

z = torch.randn(args.batch_size, generator.z_space_dim).cuda()

with torch.no_grad():
    images = postprocess(generator(z, trunc_psi=args.trunc_psi,
                                trunc_layers=args.trunc_layers,
                                randomize_noise=args.randomize_noise)['image'])

with AdaIN.optimize_stats(generator) as stats:
    optimize(generator, z, targets, stats, lr=0.05)