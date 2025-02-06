import math
import torch as th

from torch import nn, Tensor
from dataclasses import dataclass
from typing import Tuple, NamedTuple
from einops import rearrange, repeat

from utils import M2H
from config_base import BaseConfig
from .nn import conv_nd, linear, timestep_embedding
from .MBAblocks import ResBlockConfig, AttnBlock, LlamaRMSNorm, \
    TimestepEmbedSequential, Downsample, Upsample


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
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf
        assert conf.dims == 3

        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = th.float32

        self.time_emb_channels = conf.time_embed_channels or conf.model_channels
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
            use_pos=conf.use_pos)        

        if conf.num_classes is not None:
            self.label_emb = nn.Embedding(conf.num_classes,
                                          conf.embed_channels)

        self.z_size = math.ceil(len(self.conf.rna_tpl) / 2)
        self.in_chn = conf.in_channels // self.z_size
        self.pxl2vxl = 'b (s z) h w -> b s z h w'
        self.vxl2pxl = 'b s z h w -> b (s z) h w'
        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        self.rna_blocks, roch = self.init_rblk(self.conf)

        kwargs = dict(
            use_condition=True,
            two_cond=conf.resnet_two_cond,
            cond_emb_channels=conf.resnet_cond_channels,
            use_zero_module=conf.resnet_use_zero_module
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

    def init_rblk(self, conf):
        ich = [conf.rna_num, conf.rna_num, 128, 64]
        och = [conf.rna_num, 128, 64, 32]

        blocks = nn.ModuleList([])
        attn = [AttnBlock((conf.gn_sz**2)*len(conf.rna_tpl), 
                          use_checkpoint=conf.use_checkpoint,
                          gene_trans=False,
                          gene_size=conf.rna_num,
                          enable_flash_attn=False,
                          z_size=len(conf.rna_tpl))]
        # Share the rna attn
        blocks.append(nn.Sequential(*attn))
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
        # Assume 4 slices
        alst = []
        for i in range(3):
            rna_ = th.zeros_like(rna_h)
            rna_[:, :, i:i+2] = rna_h[:, :, i:i+2]
            attn = self.rna_blocks[0](rna_)[1]
            alst.append(attn.squeeze())
        attn = self.rna_blocks[0](rna_h)[1] 
        alst.append(attn.squeeze()) 
        return th.stack(alst), rna_h[:, :, 1:-1]

    def to_collage(self, h, half_p, 
                   p1, p2, hei, wid, chn=None):
        if chn is None:
            chn = self.conf.dims
        if chn == 2:
            tl2im = '(b p1 p2) c h w -> b c (p1 h) (p2 w)'
            im2tl = 'b c (p1 h) (p2 w) -> (b p1 p2) c h w'
        elif self.conf.dims == 3:
            tl2im = '(b p1 p2) c z h w -> b c z (p1 h) (p2 w)'
            im2tl = 'b c z (p1 h) (p2 w) -> (b p1 p2) c z h w'
        else:
            raise ValueError(f'Dim {chn} not in (2, 3)')

        h_ori = rearrange(h, tl2im, p1=p1, p2=p2)
        h_ori_crop = h_ori[..., half_p:-half_p, half_p:-half_p]
        h_collage = rearrange(h_ori_crop, im2tl, h=hei, w=wid)
        return h_collage

    def forward(self,
                x,
                t,
                rna=None,
                pos=None,
                y=None,
                imgs=None,
                cond=None,
                noise=None,
                t_cond=None,
                idx = None,
                index = None,
                do_train = False,
                patch_size = 64,
                pos_random = None,
                random = None,
                **kwargs):

        H,W = imgs.shape[-2:]
        patch_num_x = H // patch_size
        patch_num_y = W // patch_size
        p1 = 2 if do_train else patch_num_x + 1
        p2 = 2 if do_train else patch_num_y + 1

        return self.get_rna(rna)


class AutoencReturn(NamedTuple):
    pred: Tensor
    pred2: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels, use_pos=True):
        super().__init__()
        out_chn = time_out_channels // 2 if use_pos else time_out_channels
        self.time_embed = nn.Sequential(
            linear(time_channels, out_chn),
            nn.SiLU(),
            linear(out_chn, out_chn),
        )

        self.pos_embed = None
        if use_pos:
            self.pos_embed = nn.Sequential(
                linear(128, time_out_channels // 2),
                nn.SiLU(),
                linear(time_out_channels // 2, time_out_channels // 2),
            )
        

    def forward(self, time_emb=None, pos_emb=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)

        if pos_emb is not None:
            pos_emb = self.pos_embed(pos_emb)
        else:
            assert self.pos_embed is None

        
        time_out = time_emb if pos_emb is None else th.cat([time_emb, pos_emb], dim=1)
        return EmbedReturn(emb=None, time_emb=time_out)