from typing import Tuple, NamedTuple
from dataclasses import dataclass
import math
import torch as th
from torch import nn, Tensor
from config_base import BaseConfig
from .blocks import ResBlockConfig, AttentionBlock, normalization, \
    TimestepEmbedSequential, Downsample, Upsample

from .nn import conv_nd, linear, timestep_embedding, zero_module
from einops import rearrange, repeat
from utils import M2H

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

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, self.in_chn, ch, (1, 3, 3), 
                        padding=(0, 1, 1)))
        ])

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

        ds, resolution = 1, conf.image_size
        for level, mult in enumerate(conf.input_channel_mult
                                     or conf.channel_mult):
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                            z_size=self.z_size
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                            **kwargs,
                        ).make_model() if conf.
                        resblock_updown else Downsample(ch,
                                                        conf.conv_resample,
                                                        dims=conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
                z_size=self.z_size),
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model())
        
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            rdch = roch[len(roch)-1-level]
            for i in range(conf.num_res_blocks + 1):
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch+ich+rdch,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                            z_size=self.z_size))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(conf.dims, input_ch, self.in_chn, 
                    (1, 3, 3), padding=(0, 1, 1)))

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
        rna_l = []
        for rid in range(len(self.rna_blocks)):
            rna_ = self.rna_blocks[rid](rna_h if rid == 0 else rna_)
            rna_l.append(rna_)
        return rna_l

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
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
            random/pose_random: classifier free
        """
        H,W = imgs.shape[-2:]
        patch_num_x = H // patch_size
        patch_num_y = W // patch_size

        _t_emb, pos_emb = None, None
        if t is not None:
            t_cur = repeat(t,'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))
            _t_emb = timestep_embedding(t_cur, self.conf.model_channels)

        if self.conf.use_pos:
            # print('if use position embedding', self.conf.use_pos)
            pos_x = timestep_embedding(pos[:, 0], 64)
            pos_y = timestep_embedding(pos[:, 1], 64)
            pos_emb = th.cat([pos_x, pos_y], dim=1)
            if pos_random is not None: # pose embedding classifier free
                pos_random = pos_random >= 0.5
                pos_random_old = pos_random.repeat_interleave(x.shape[0] // t.shape[0], dim=0)
                pos_emb = pos_emb * pos_random_old[:,None].to(pos_emb.device)

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                pos_emb=pos_emb)
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None


        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb


        '''====NOTE: Additional embedding===='''
        if not do_train: # rendering, need to revise first
            grid_x = th.linspace(0.5, patch_num_x-0.5, patch_num_x, device=emb.device)
            grid_y = th.linspace(0.5, patch_num_y-0.5, patch_num_y, device=emb.device)
            xx, yy = th.meshgrid(grid_x, grid_y, indexing='ij')
            pos_new = th.stack([xx, yy], dim=-1).flatten(0, 1).repeat(t.shape[0], 1)
        else:
            pos_new = th.stack([index[0]+0.5, index[1]+0.5], dim = -1).unsqueeze(0).repeat(t.shape[0], 1)

        pos_emb_new = None
        if self.conf.use_pos:
            # print('if use new position embedding', self.conf.use_pos)
            pos_x_new = timestep_embedding(pos_new[:, 0], 64)
            pos_y_new = timestep_embedding(pos_new[:, 1], 64)
            pos_emb_new = th.cat([pos_x_new, pos_y_new], dim=1).to(pos_emb.device)

            if pos_random is not None: # pose embedding classifier free
                pos_random_new = pos_random.repeat_interleave(pos_new.shape[0] // t.shape[0], dim=0)
                pos_emb_new = pos_emb_new * pos_random_new[:,None].to(pos_emb_new.device)
        # change of time embedding
        t_cur_new = repeat(t,'h -> (h repeat)',repeat =int(pos_new.shape[0]/t.shape[0]))
        _t_emb_new = timestep_embedding(t_cur_new, self.conf.model_channels)
        # print('time proc', imgs.shape, pos_new.shape, t.shape, t_cur_new.shape, _t_emb_new.shape)
        res_new = self.time_embed.forward(
                time_emb=_t_emb_new,
                pos_emb=pos_emb_new)

        rna = self.get_rna(rna)
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        hs_train = [[] for _ in range(len(self.conf.channel_mult))]
        x = rearrange(x, self.pxl2vxl, z=self.z_size)
        h = x.type(self.dtype)
        # input blocks
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h,
                                         emb=enc_time_emb,
                                         cond=enc_cond_emb)
                hs[i].append(h)
                hs_train[i].append(h.clone())
                k += 1
        assert k == len(self.input_blocks)
        # middle blocks
        h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        h_train = h.clone()
        p1 = 2 if do_train else patch_num_x + 1
        p2 = 2 if do_train else patch_num_y + 1

        # Output blocks: 1st run for collage
        k = 0
        for i in range(len(self.output_num_blocks)):
            hei, wid = h.shape[-2:]
            half_p = int(hei//2)
            if i == 0:
                h = self.to_collage(h, half_p,
                                    p1, p2, hei, wid)
            rna_i = self.to_collage(rna[i], half_p,
                                    p1, p2, hei, wid, 3)

            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                    lateral = self.to_collage(lateral, half_p,
                                              p1, p2, hei, wid)
                except IndexError:
                    lateral = None

                h = th.cat((h, lateral, rna_i), 1)
                assert h.size(0) == res_new.time_emb.size(0) == lateral.size(0)
                h = self.output_blocks[k](h,
                                          emb=res_new.time_emb,
                                          cond=rna_i,
                                          lateral=None)
                k += 1
        
        pred1 = self.out(h)
        pred1 = rearrange(pred1, self.vxl2pxl)

        # Output blocks: 2nd run for origin tiles
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs_train[-i - 1].pop()
                except IndexError:
                    lateral = None

                h_train = th.cat((h_train, lateral, rna[i]), 1)
                h_train = self.output_blocks[k](h_train,
                                                emb=dec_time_emb,
                                                cond=rna[i],
                                                lateral=None)
                k += 1

        pred2 = self.out(h_train)
        pred2 = rearrange(pred2, self.vxl2pxl)

        return AutoencReturn(pred=pred1, pred2=pred2, cond=cond)


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