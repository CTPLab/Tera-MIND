from typing import Tuple
from dataclasses import dataclass

import math
import torch as th
from torch import nn
from config_base import BaseConfig
from .blocks import Upsample, AttentionBlock

from .nn import conv_nd
from einops import rearrange
from utils import M2H


def exists(x):
    return x is not None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(th.ones(1, dim, 1, 1))
        self.b = nn.Parameter(th.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = th.var(x, dim=1, unbiased=False, keepdim=True)
        mean = th.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=3, norm=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)
        ) if exists(emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(emb), 'time (and possibly frame) emb must be passed in'
            condition = self.mlp(emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)


@dataclass
class BeatGANsUNetConfig(BaseConfig):
    image_size: int = 64
    input_size: int = None
    gn_sz: int = 4
    in_channels: int = 3
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    # suggest: 3
    # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
    rna_tpl: tuple = tuple(range(32))
    rna_num: int = 500 # the amount of rnas
    out_channels: int = 3
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # you can also set the number of resblocks specifically for the input blocks
    # default: None = above
    num_input_res_blocks: int = None
    # number of time embed channels and style channels
    embed_channels: int = 512
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    # default: [16]
    # beatgans: [32, 16, 8]
    attention_resolutions: Tuple[int] = (16, )
    # number of time embed channels
    time_embed_channels: int = None
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    # always 2 = 2d conv
    dims: int = 2
    semantic_path: str = ""
    data_num: int = 0
    # don't use this, legacy from BeatGANs
    num_classes: int = None
    use_checkpoint: bool = False
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # what's this?
    num_heads_upsample: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    # never tried
    use_new_attention_order: bool = False
    resnet_two_cond: bool = False
    resnet_cond_channels: int = None
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeattGANs)
    resnet_use_zero_module: bool = True
    semantic_enc: bool = True
    # gradient checkpoint the attention operation
    attn_checkpoint: bool = False
    use_pos: bool = True
    def make_model(self):
        return BeatGANsUNetModel(self)


class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig, 
                 filters_per_layer=64, depth=16, 
                 frame_conditioned=False):
        super().__init__()
        self.conf = conf
        self.z_size = math.ceil(len(self.conf.rna_tpl) / 2)
        self.rna_blocks, self.och = self.init_rblk(conf)

        self.dtype = th.float32

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        emb_dim = time_dim * 2 if frame_conditioned else time_dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(conf.in_channels, dims[0], emb_dim=emb_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1] + self.och[-1] * self.z_size, 
                                             dims[i], emb_dim=emb_dim, norm=True))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], conf.in_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        if frame_conditioned:
            # Encoder for positional embedding of frame
            self.frame_encoder = nn.Sequential(
                 SinusoidalPosEmb(time_dim),
                 nn.Linear(time_dim, time_dim * 4),
                 nn.GELU(),
                 nn.Linear(time_dim * 4, time_dim)
            )

    def init_rblk(self, conf):
        ich = [conf.rna_num, conf.rna_num, 128, 64]
        och = [conf.rna_num, 128, 64, 32]

        blocks = nn.ModuleList([])
        attn = [AttentionBlock(ich[0], 
                          use_checkpoint=conf.use_checkpoint,
                          use_new_attention_order=conf.use_new_attention_order,
                          is_half=False)]
        ker = {1:1, 4:3, 8:5, 16:9}[len(self.conf.rna_tpl)]
        attn += [nn.Conv3d(conf.rna_num, conf.rna_num, 
                           (ker, 3, 3), padding=(0, 1, 1))]
        attn += [Upsample(och[0], False, 3)]
        # Share the rna attn
        blocks.append(nn.Sequential(*attn))
        for rid in range(1, len(ich)):
            blyr = [nn.SiLU(),
                    conv_nd(3, ich[rid], och[rid], 
                            [1, 3, 3], padding=[0, 1, 1]),
                    Upsample(och[rid], False, 3)]
            blocks.append(nn.Sequential(*blyr))
        return blocks, och

    def get_rna(self, rna):
        # rna blocks
        assert rna is not None
        if th.is_tensor(rna):
            rna_t = rna
        else:
            dat, crd, ssz = rna
            rna_t = th.sparse_coo_tensor(crd.long(), dat, ssz)
            rna_t = rna_t.to_dense()
        if rna_t.shape[-1] != len(M2H):
            rna_h = rearrange(rna_t, 'b h w (z g) -> b g z h w', 
                              g=500)
        else:
            rna_h = rearrange(rna_t, 'b h w (z g) -> b g z h w', 
                              g=len(M2H))
        if self.conf.rna_num != rna_h.shape[1]:
            if self.conf.rna_num == len(M2H):
                rna_h = rna_h[:, M2H]
                assert len(self.conf.rna_tpl) == rna_h.shape[2] == 1 
            else:
                rna_h = rna_h[:, :self.conf.rna_num] 
        for rid in range(len(self.rna_blocks)):
            rna_h = self.rna_blocks[rid](rna_h)
        rna_h = rearrange(rna_h, 'b g z h w -> b (z g) h w')
        return rna_h

    def forward(self, x, t, rna, y=None, debug=False, frame_diff=None, **kwargs):
        # rna blocks
        assert rna is not None
        rna = self.get_rna(rna)
        
        time_embedding = self.time_encoder(t)

        if frame_diff is not None:
            frame_embedding = self.frame_encoder(frame_diff)
            embedding = th.cat([time_embedding, frame_embedding], dim=1)
        else:
            embedding = time_embedding

        x = x.type(self.dtype)
        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = th.cat((x, residuals.pop(), rna), dim=1)
            x = layer(x, embedding)

        x = x.type(self.dtype)
        return self.final_conv(x)