"""
Various utilities for neural networks.
"""

from enum import Enum
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.utils.checkpoint

import torch.nn.functional as F
from einops import rearrange, repeat

def equinorm(channels, ksize=5, pad=2):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return EquiGroupNorm(min(32, channels), channels, ksize, pad)


class EquiGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, 
                 kernel_size: int = None, pad: int = 0,
                 eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        if kernel_size is None:
            assert pad == 0
        else:
            assert kernel_size >= 1 

        self.ng = num_groups
        self.nc = num_channels
        self.ks = kernel_size
        self.pad = pad
        self.eps = eps
        self.affine = affine
        if self.affine:
            _sz = (1, self.nc, 1, 1)
            self.weight = nn.Parameter(th.empty(_sz, **factory_kwargs))
            self.bias = nn.Parameter(th.empty(_sz, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.cg = self.nc // self.ng
        if self.ks is not None:
            self.exl = (self.ks - 1) // 2
            self.avgpool = nn.AvgPool3d((self.cg, self.ks, self.ks), (1, 1, 1))

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.size()
        assert C % self.ng == 0 and C == self.nc
        x = rearrange(x, 'n (g c) h w -> n g c h w', g=self.ng)
        if self.ks is None:
            mean = x.mean([-3, -2, -1], keepdim=True)
            var = x.var([-3, -2, -1], unbiased=False, keepdim=True)
        else:
            x = F.pad(x, (self.pad, self.pad, 
                          self.pad, self.pad))
            mean = self.avgpool(x)
            var = self.avgpool(x ** 2) - mean ** 2
            x = x[:, :, :, 
                  self.exl:-self.exl, 
                  self.exl:-self.exl]
        
        x = (x - mean) / (var + self.eps).sqrt()
        x = rearrange(x, 'n g c h w -> n (g c) h w')
        if self.affine:
            return x * self.weight + self.bias
        else:
            return x


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # @th.jit.script
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    chn = 1
    for i in (32, 16, 8, 4, 2):
        if channels % i == 0:
            chn = i
            break
    return GroupNorm32(min(chn, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) *
                   th.arange(start=0, end=half, dtype=th.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state)
    else:
        return func(*args)


if __name__ == '__main__':
    for ker in (None, 3, 7, 9, 17, 33):
        sz = 48
        res = 128 if ker == None else ker + sz - 1
        for bat, grp, chn, sz, res in ((8, 1, 3, sz, res), 
                                       (8, 32, 128, sz, res)):
            print(ker, bat, grp, chn, sz, res)
            norm0 = EquiGroupNorm(grp, chn, ker).cuda()
            norm1 = GroupNorm32(grp, chn).cuda()

            if ker is None:
                img = th.randn(bat, chn, res, res).cuda().float()
                out0 = norm0(img)
                out1 = norm1(img)
                oabs = (out0 - out1).abs()
            else:
                # Test the num consistency between EquiGroupNorm and GroupNorm
                img = th.randn(bat, chn, res, res).cuda().float()
                out0 = norm0(img)
                iufd = img.unfold(2, ker, 1).unfold(3, ker, 1)
                n_h, n_w = iufd.shape[2:4]
                # iufd0 = iufd.permute((0, 2, 3, 1, 4, 5)).reshape((-1, chn, ker, ker))
                iufd0 = rearrange(iufd, 'b c n_h n_w h w -> (b n_h n_w) c h w')
                iufd0 = norm1(iufd0)
                # iufd1 = iufd0[:, None, None].reshape((bat, sz, sz, chn, ker, ker))
                # iufd1 = iufd1.permute((0, 3, 1, 2, 4, 5))
                iufd1 = rearrange(iufd0, '(b n_h n_w) c h w -> b c n_h n_w h w',
                                  n_h=n_h, n_w=n_w)
                # print((iufd1 == iufd).all())
                oabs = (out0 - iufd1[:, :, :, :, ker // 2, ker // 2]).abs()

                # Test boundary consistency
                res_new = sz * 2 - ker
                img_new = th.randn(bat, chn, res_new, res_new).cuda().float()
                img_n = img_new.unfold(2, sz, sz - ker).unfold(3, sz, sz - ker)
                a00 = norm0(img_n[:, :, 0, 0])
                a01 = norm0(img_n[:, :, 0, 1])
                a10 = norm0(img_n[:, :, 1, 0])
                a11 = norm0(img_n[:, :, 1, 1])
                print((a00[:, :, :, -1] - a01[:, :, :, 0]).abs().max())
                print((a00[:, :, -1] - a10[:, :, 0]).abs().max())
                print((a00[:, :, -1, -1] - a11[:, :, 0, 0]).abs().max())
                print((a10[:, :, :, -1] - a11[:, :, :, 0]).abs().max())
                print((a01[:, :, -1] - a11[:, :, 0]).abs().max())
            print('   ', oabs.shape, oabs.max(), oabs.mean())