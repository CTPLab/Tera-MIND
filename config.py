import os, math

from typing import Tuple
from torch import distributed
from dataclasses import dataclass
from multiprocessing import get_context
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import ModelConfig
from model.blocks import ScaleAt
from config_base import BaseConfig
from utils.MBADataset import MBADataset
from diffusion.resample import UniformSampler
from diffusion.diffusion import space_timesteps
from diffusion import SpacedDiffusionBeatGansConfig
from diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from utils.choices import OptimizerType, ModelName, ModelType, Activation, TrainMode, ManipulateMode, ManipulateLossType


@dataclass
class PretrainConfig(BaseConfig):
    name: str
    path: str

@dataclass
class TrainConfig(BaseConfig):
    is_tot: bool = False
    method: str = 'ours'
    dims: int = 2
    # random seed
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celebahq_all
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    cfg: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = 'linear'
    beta_scheduler: str = 'linear'
    semantic_path: str = ''
    data_num: int = 0
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    # initialize 32 to avoid divisble issue 
    # caused by groupnorm
    rna_tpl: Tuple[int] = tuple(range(32))
    rna_num: int = 500
    stain: float = 'DAPI'
    # img_size: int = 64 # Deprecate
    image_size: int = 64
    input_size: int = None # expanded image size for equi
    gn_sz: int = 4
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    full_model_path: str = ""
    patch_size: int = 64
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'adaptivenonzero'
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: bool = None
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: Tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 125_000
    save_every_samples: int = 100_000
    semantic_enc: bool = False
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser('~/cache')
    work_cache_dir: str = os.path.expanduser('~/mycache')
    # to be overridden
    name: str = 'test'
    output_dir: str = f'{work_cache_dir}/gen_images/{name}'
    data_path: str = ""
    use_pos: bool = True

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        # self.data_val_name = self.data_val_name or self.data_name

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        return f'{self.work_cache_dir}/eval_images/{self.data_name}_{self.eval_num_images}'

    @property
    def logdir(self):
        if self.name:
            return f'{self.base_dir}/{self.name}'
        else: return ""

    @property
    def generate_dir(self):
        return self.output_dir

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'beatgans':
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T,
                                              section_counts=section_counts),
                fp16=self.fp16,
                cfg=self.cfg,
                is_tot=self.is_tot,
                image_size=self.image_size,
                input_size=self.input_size or self.image_size,
                gn_sz=self.gn_sz,
                method=self.method,
                rna_tpl=self.rna_tpl
            )
        else:
            raise NotImplementedError()

    @property
    def model_out_channels(self):
        return 3

    def make_T_sampler(self):
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_dataset(self, path=None, **kwargs):
        # 609882 0117 28 
        if self.method == 'ours':
            coef = 2 if self.image_size == 128 else 4
            im_sz = coef * self.image_size
            use_exl = True if self.image_size == 32 else False
        else:
            coef = 2 if self.is_tot else 4
            im_sz = coef * self.image_size
            use_exl = True if self.image_size == 32 else False
        return MBADataset(self.data_path, self.rna_num, 
                          self.image_size//self.gn_sz, 
                          im_sz, self.gn_sz//2, 
                          snum=len(self.rna_tpl), stain=self.stain,
                          transform=True, methd=self.method, use_exl=use_exl)

    def make_loader(self,
                    dataset, 
                    collate_fn,
                    shuffle: bool,
                    num_worker: bool = None,
                    drop_last: bool = True,
                    batch_size: int = None,
                    parallel: bool = False):
        if parallel and distributed.is_initialized():
            # drop last to make sure that there is no added special indexes
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        else:
            sampler = None
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=False,
            drop_last=drop_last,
            multiprocessing_context=get_context('fork'),
        )

    def make_model_conf(self):
        if self.method == 'patch-dm':
            from model.unet_patch_dm import BeatGANsUNetConfig
            self.model_type = ModelType.ddpm
        elif self.method == 'sinf':
            from model.unet_sinf import BeatGANsUNetConfig
            self.model_type = ModelType.ddpm
        elif self.method == 'ours':
            from model.unet_ours import BeatGANsUNetConfig
            self.model_type = ModelType.autoencoder
        elif self.method == 'ours_vis':
            from model.unet_attn import BeatGANsUNetConfig
            self.model_type = ModelType.autoencoder
        rchn = len(self.rna_tpl)
        rchn = math.ceil(rchn / 2)
        self.model_conf = BeatGANsUNetConfig(
            use_pos=self.use_pos,
            attention_resolutions=self.net_attn,
            channel_mult=self.net_ch_mult,
            conv_resample=True,
            semantic_path = self.semantic_path,
            data_num = self.data_num,
            dims=self.dims,
            dropout=self.dropout,
            embed_channels=self.net_beatgans_embed_channels,
            image_size=self.image_size,
            input_size=self.input_size or self.image_size,
            gn_sz=self.gn_sz,
            in_channels=rchn if self.stain != 'all' else rchn * 2,
            model_channels=self.net_ch,
            num_classes=None,
            num_head_channels=-1,
            num_heads_upsample=-1,
            num_heads=self.net_beatgans_attn_head,
            num_res_blocks=self.net_num_res_blocks,
            num_input_res_blocks=self.net_num_input_res_blocks,
            rna_tpl=self.rna_tpl,
            rna_num=self.rna_num,
            out_channels=rchn if self.stain != 'all' else rchn * 2,
            resblock_updown=self.net_resblock_updown,
            semantic_enc = self.semantic_enc,
            use_checkpoint=self.net_beatgans_gradient_checkpoint,
            use_new_attention_order=False,
            resnet_two_cond=self.net_beatgans_resnet_two_cond,
            resnet_use_zero_module=self.
            net_beatgans_resnet_use_zero_module)
        return self.model_conf
