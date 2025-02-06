import math
import numpy as np
import torch as th
import torch.nn.functional as F

from torch import nn
from numbers import Number
from einops import rearrange
from abc import abstractmethod
from dataclasses import dataclass
from timm.models.vision_transformer import Mlp
from torch.nn.functional import scaled_dot_product_attention

from utils.choices import Enum
from config_base import BaseConfig
from .blocks import Upsample, Downsample
from .nn import conv_nd, linear, torch_checkpoint, zero_module
approx_gelu = lambda: nn.GELU(approximate="tanh")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, dim=-1, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        assert dim in (1, -1)
        if dim == -1:
            self.weight = nn.Parameter(th.ones(hidden_size))
        elif dim == 1:
            self.weight = nn.Parameter(th.ones(1, hidden_size, 1, 1))
        self.dim=dim
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(th.float32)
        # Here, we assume hidden_states are at dim 1
        variance = hidden_states.pow(2).mean(self.dim, keepdim=True)
        hidden_states = hidden_states * th.rsqrt(variance + self.variance_epsilon)
        if self.dim == 1 and len(hidden_states.shape) == 5:
            return self.weight[..., None] * hidden_states.to(input_dtype)
        return self.weight * hidden_states.to(input_dtype)


class ScaleAt(Enum):
    after_norm = 'afternorm'


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    # condition the resblock with time (and encoder's output)
    use_condition: bool = True
    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_conv: bool = False
    # dimension of conv (always 2 = 2d)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None
    # suggest: False
    has_lateral: bool = False
    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True
    is_pad: bool = True
    k_norm: int = 5
    k_conv: int = 3

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        assert conf.k_norm is None or conf.k_norm >= 1
        self.conf = conf
        p_norm = (conf.k_norm - 1) // 2 if conf.is_pad and conf.k_norm is not None else 0
        p_conv = (conf.k_conv - 1) // 2 if conf.is_pad else 0
        p_conv = p_conv if conf.dims == 2 else (1, p_conv, p_conv)
        k_conv = conf.k_conv
        if conf.dims == 3:
            k_conv = (3, k_conv, k_conv)

        #############################
        # IN LAYERS
        #############################
        layers = [
            LlamaRMSNorm(conf.channels, 1),
            # equinorm(conf.channels, conf.k_norm, p_norm),
            # normalization(conf.channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.channels, conf.out_channels, 
                    k_conv, 
                    padding=p_conv)
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # OUT LAYERS CONDITIONS
        #############################
        if conf.use_condition:
            # condition layers for the out_layers
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(conf.emb_channels, 2 * conf.out_channels),
            )

            # if conf.two_cond:
            #     self.cond_emb_layers = nn.Sequential(
            #         nn.SiLU(),
            #         linear(conf.cond_emb_channels, conf.out_channels),
            #     )
            #############################
            # OUT LAYERS (ignored when there is no condition)
            #############################
            # original version
            conv = conv_nd(conf.dims,
                           conf.out_channels,
                           conf.out_channels,
                           k_conv,
                           padding=p_conv)
            if conf.use_zero_module:
                conv = zero_module(conv)

            # construct the layers
            # - norm
            # - (modulation)
            # - act
            # - dropout
            # - conv
            layers = [
                LlamaRMSNorm(conf.out_channels, 1),
                # equinorm(conf.out_channels, conf.k_norm, p_norm),
                # normalization(conf.out_channels),
                nn.SiLU(),
                nn.Dropout(p=conf.dropout),
                conv,
            ]
            self.out_layers = nn.Sequential(*layers)

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if conf.use_conv:
                kernel_size = k_conv
                padding = p_conv
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(conf.dims,
                                           conf.channels,
                                           conf.out_channels,
                                           kernel_size,
                                           padding=padding)

    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return torch_checkpoint(self._forward, (x, emb, cond, lateral),
                                self.conf.use_checkpoint)

    def _forward(
        self,
        x,
        emb=None,
        cond=None,
        lateral=None,
    ):
        """
        Args:
            lateral: required if "has_lateral" and non-gated, with gated, it can be supplied optionally    
        """
        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if "has_lateral"
            assert lateral is not None
            x = th.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.conf.use_condition:
            # it's possible that the network may not receieve the time emb
            # this happens with autoenc and setting the time_at
            if emb is not None:
                emb_out = self.emb_layers(emb).type(h.dtype)
            else:
                emb_out = None

            # if self.conf.two_cond:
            #     # it's possible that the network is two_cond
            #     # but it doesn't get the second condition
            #     # in which case, we ignore the second condition
            #     # and treat as if the network has one condition
            #     if cond is None:
            #         cond_out = None
            #     else:
            #         cond_out = self.cond_emb_layers(cond).type(h.dtype)

            #     if cond_out is not None:
            #         while len(cond_out.shape) < len(h.shape):
            #             cond_out = cond_out[..., None]
            # else:
            cond_out = None

            # this is the new refactored code
            h = apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                layers=self.out_layers,
                scale_bias=1,
                in_channels=self.conf.out_channels,
            )

        out = self.skip_connection(x)
        out = align_spatial_size(out, h)
        return out + h


def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers: nn.Sequential = None,
    scale_bias: float = 1,
    in_channels: int = 512,
):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    # if two_cond:
    #     # adjusting shapes
    #     while len(cond.shape) < len(h.shape):
    #         cond = cond[..., None]
    #     print('cond shape', cond.shape)
    #     # time first
    #     scale_shifts = [emb, cond]
    # else:
    #     # "cond" is not used with single cond mode
    scale_shifts = [emb]

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        else:
            if each.shape[1] == in_channels * 2:
                a, b = th.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)
    else:
        # a list
        biases = scale_bias

    # default, the scale & shift are applied after the group norm but BEFORE SiLU
    pre_layers, post_layers = layers[0], layers[1:]

    h = pre_layers(h)
    # scale and shift for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = post_layers(h)
    return h


def align_spatial_size(source, target):
    if source is None:
        return source
    cH, cW = target.shape[-2:]
    sH, sW = source.shape[-2:]
    if (cH == sH) and (cW == sW):
        return source
    # Conv has more (or even) layers, the size is always smaller than or equal to source
    assert ((sH - cH) % 2 == 0) and ((sW - cW) % 2 == 0), \
        "Should always have equal padding on two sides, got target ({}x{}) and source ({}x{})".format(
            cH, cW, sH, sW)
    h_st = (sH - cH) // 2
    w_st = (sW - cW) // 2
    return source[..., h_st:h_st+cH, w_st:w_st+cW]

    
def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


def get_conv_mask(hsz, wsz, ker):
    dim, pad = hsz * wsz, (ker - 1) // 2 
    mask = th.zeros(dim, dim).bool()

    idx = th.linspace(0, dim - 1, dim).int()
    h, w = th.meshgrid(idx, idx, indexing='ij')
    pos = th.stack([h, w], -1)
    pos = rearrange(pos, 'h (w_r w_c) two -> h two w_r w_c',
                    w_r=hsz,  w_c=wsz)
    # Hack: Always pad position (0, 0), as (0, 0) always assign 1 for the mask
    pos = F.pad(pos, (pad, pad, pad, pad))
    pos = pos.unfold(2, ker, 1).unfold(3, ker, 1)
    pos = rearrange(pos, 'h two w_r w_c k_r k_c -> h (w_r w_c) two (k_r k_c)')
    pos = th.diagonal(pos, dim1=0, dim2=1)
    # diag dim is moved to the last dim
    pos = rearrange(pos, 'two ker diag -> two (ker diag)')

    mask[pos[0], pos[1]] = 1
    return mask


class AttnBlock(TimestepBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads=1,
        use_checkpoint=False,
        mlp_ratio=4.0,
        enable_flash_attn=True,
        gene_trans=True,
        gene_size=None,
        z_size=1,
        n_h=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.enable_flash_attn = enable_flash_attn
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
            z_size=z_size,
            gene_trans=gene_trans, n_h=n_h)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        adaLN = None if not gene_trans or gene_size is None else \
            nn.Sequential(nn.SiLU(), 
                          nn.Linear(gene_size, 7 * hidden_size, bias=True))
        self.adaLN_modulation = adaLN

        self.gene_trans, self.z_size = gene_trans, z_size
        if not gene_trans:
            self.norm1 = None
            assert n_h is None
            ker = {1:1, 4:3, 8:5, 16:9}[z_size]
            self.down_z = nn.Conv3d(gene_size, gene_size, 
                                    (ker, 3, 3), padding=(0, 1, 1))

    def forward(self, x, emb=None, cond=None, lateral=None):
        return torch_checkpoint(self._forward, (x, emb, cond, lateral), self.use_checkpoint)

    def _forward(self, x, emb=None, cond=None, lateral=None):
        assert len(x.shape) == 5
        H, W = x.shape[-2:]
        _before, _after = 'b c z h w', 'b (z h w) c'

        x = rearrange(x, f'{_before} -> {_after}')
        if cond is not None:
            cond = rearrange(cond, f'{_before} -> {_after}')
            shift_msa, scale_msa, gate_msa, crss_cnd, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(7, dim=-1)
            x = x + gate_msa * self.attn(modulate(self.norm1, x, shift_msa, scale_msa), crss_cnd)
            x = x + gate_mlp * self.mlp(modulate(self.norm2, x, shift_mlp, scale_mlp))
        else:
            assert self.adaLN_modulation is None
            dtype = x.dtype
            if not self.gene_trans:
                x = rearrange(x, 'b n c -> b c n') 
                if self.enable_flash_attn:
                    x = self.attn(x)
                else:
                    x, attn_mat = self.attn(x)
                x = self.norm2(x.to(th.float32)).to(dtype)
                x = self.mlp(x)
                x = rearrange(x, 'b c n -> b n c') 
            else:
                x = self.norm1(x.to(th.float32)).to(dtype)
                x = x + self.attn(x)
                x = self.norm2(x.to(th.float32)).to(dtype)
                x = x + self.mlp(x)

        x = rearrange(x, f'{_after} -> {_before}', h=H, w=W)
        if not self.gene_trans:
            x = self.down_z(x)
        if self.enable_flash_attn:
            return x
        else:
            return x, attn_mat


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash_attn: bool = True,
        z_size=1,
        gene_trans=True,
        n_h=None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias) if gene_trans else None 
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = LlamaRMSNorm(self.head_dim, -1)
        self.k_norm = LlamaRMSNorm(self.head_dim, -1) if gene_trans else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.z_size = z_size
        self.gene_trans = gene_trans
        self.n_h = n_h

    def forward(self, x: th.Tensor, y=None) -> th.Tensor:
        q = self.q(x)
        k = self.k(x if y is None else y) if self.gene_trans else q
        v = self.v(x if y is None else y)
        qkv = rearrange(th.stack((q, k, v)), 
                        'three b zhw (n_head head) -> three b n_head zhw head',
                        three=3, n_head=self.num_heads)
        if self.n_h is not None:
            _sz = int(math.sqrt(qkv.shape[-2] // self.z_size))
            qkv = rearrange(qkv, 
                            'three b n_head (z h w) head -> three b n_head z h w head',
                            h=_sz, w=_sz)
            qkv = rearrange(qkv, 
                            'three b n_head z (n_h h) (n_w w) head -> three b (n_head n_h n_w) (z h w) head',
                            n_h=self.n_h, n_w=self.n_h)

        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k) if self.gene_trans else self.q_norm(k)
        
        if self.enable_flash_attn:
            x = scaled_dot_product_attention(
                q * self.scale,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            dtype = q.dtype
            q = q * (self.scale**2)
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(th.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            # attn = self.attn_drop(attn)
            x = attn @ v

        if self.n_h is not None:
            x = rearrange(x, 
                          'b (n_head n_h n_w) (z h w) head -> b n_head z (n_h h) (n_w w) head',
                          n_h=self.n_h, n_w=self.n_h, 
                          h=_sz//self.n_h, w=_sz//self.n_h)
            x = rearrange(x, 
                          'b n_head z h w head -> b n_head (z h w) head')
        x = rearrange(x, 'b n_head zhw head -> b zhw (n_head head)')
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.enable_flash_attn:
            return x
        else:
            return x, attn

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(th.float32)).to(dtype)
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x


if __name__ == '__main__':

    # def get_conv_mask(hsz, wsz, ker):
    #     dim, pad = hsz * wsz, (ker - 1) // 2 
    #     mask = th.zeros(dim, dim).bool()

    #     idx = th.linspace(0, dim - 1, dim).int()
    #     h, w = th.meshgrid(idx, idx, indexing='ij')
    #     pos = th.stack([h, w], -1)
    #     pos = rearrange(pos, 'h (w_r w_c) two -> h two w_r w_c',
    #                     w_r=hsz,  w_c=wsz)
    #     # Hack: Always pad position (0, 0), as (0, 0) always assign 1 for the mask
    #     pos = F.pad(pos, (pad, pad, pad, pad))
    #     pos = pos.unfold(2, ker, 1).unfold(3, ker, 1)
    #     pos = rearrange(pos, 'h two w_r w_c k_r k_c -> h (w_r w_c) two (k_r k_c)')
    #     pos = th.diagonal(pos, dim1=0, dim2=1)
    #     # diag dim is moved to the last dim
    #     pos = rearrange(pos, 'two ker diag -> two (ker diag)')

    #     mask[pos[0], pos[1]] = 1
    #     return mask, pos
    
    # msk, pos = get_conv_mask(4, 4, 3)
    # print(msk.int())
    # print(msk.shape)

    n_heads, n_feats = 2, 32
    dim, shf = 47, 3
    k_norm, k_attn = 9, 17
    model = EquiAttnBlock(n_feats, 
                          n_heads,
                          use_zero_module=False,
                          use_new_attention_order=False,
                          is_pad=False,
                          k_norm=k_norm, 
                          k_attn=k_attn).cuda()
    # # model = InfQKVLegacy(n_heads, k_attn).cuda()
    # x = th.rand(2, n_heads * n_feats * 3,
    #             dim + shf, dim + shf)
    x = th.rand(2, n_feats, dim + shf, dim + shf)
    x = x.unfold(2, dim, shf).unfold(3, dim, shf)
    x = rearrange(x, 'b c n_h n_w h w -> (b n_h n_w) c h w')
    out = model(x.cuda())
    out = rearrange(out, '(b n_h n_w) c h w -> b n_h n_w c h w',
                    n_h=2, n_w=2)
    print(out.shape)
    print((out[:, 0, 0, :, shf:, :] - out[:, 1, 0, :, :-shf, :]).abs().max())
    print((out[:, 0, 0, :, :, shf:] - out[:, 0, 1, :, :, :-shf]).abs().max())
    print((out[:, 0, 0, :, shf:, shf:] - out[:, 1, 1, :, :-shf, :-shf]).abs().max())
