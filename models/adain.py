import adaiw

from dataclasses import dataclass
from typing import Type

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import args

_STYLEMOD_WSCALE_GAIN = 1.0

class AdaIN(nn.Module):
    pass

class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], '
                             f'but `{x.shape}` is received!')
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x ** 2, dim=[2, 3], keepdim=True) + self.eps)
        return x / norm

class StyleModLayer(AdaIN):
    """Implements the style modulation layer."""

    def __init__(self,
                w_space_dim,
                out_channels,
                use_wscale=True):
        super().__init__()
        self.normalize = InstanceNormLayer()
        self.w_space_dim = w_space_dim
        self.out_channels = out_channels

        weight_shape = (self.out_channels * 2, self.w_space_dim)
        wscale = _STYLEMOD_WSCALE_GAIN / np.sqrt(self.w_space_dim)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape))
            self.wscale = wscale
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale)
            self.wscale = 1.0

        self.bias = nn.Parameter(torch.zeros(self.out_channels * 2))

    def forward(self, x, w):
        x = self.normalize(x)
        if w.ndim != 2 or w.shape[1] != self.w_space_dim:
            raise ValueError(f'The input tensor should be with shape '
                            f'[batch_size, w_space_dim], where '
                            f'`w_space_dim` equals to {self.w_space_dim}!\n'
                            f'But `{w.shape}` is received!')
        style = F.linear(w, weight=self.weight * self.wscale, bias=self.bias)
        style_split = style.view(-1, 2, self.out_channels, 1, 1)
        x = x * (style_split[:, 0] + 1) + style_split[:, 1]
        return x, style

class BlockwiseAdaIN(adaiw.BlockwiseAdaIN, AdaIN):
    def __init__(self,
                w_space_dim,
                out_channels,
                use_wscale=True):
        print("params", w_space_dim, out_channels)
        super().__init__(w_space_dim, out_channels, block_size=args.block_size, shift_mean=True)
        print(self.block_size)
        print(self.normalizer)

    def forward(self, x, w):
        y = super().forward(x, w)
        return y, self.last_projected_style

class StandardizationAdaIN(adaiw.AdaIN, AdaIN):
    def __init__(self,
                w_space_dim,
                out_channels,
                use_wscale=True):
        super().__init__(w_space_dim, out_channels, shift_mean=True)

    def forward(self, x, w):
        y = super().forward(x, w)
        return y, self.last_projected_style
