import os
import copy
import torch
import random
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import amp
from pathlib import Path
from einops import rearrange
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid, save_image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from config import TrainConfig
from utils.dist_utils import get_world_size
from utils.MBADataset import sparse_batch_collate
from utils.choices import TrainMode, GenerativeType, OptimizerType


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode == TrainMode.diffusion
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf
        self.model = conf.make_model_conf().make_model()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        self.patch_size = conf.patch_size
        print(f"==Model size is {self.patch_size}==")
        self.sampler = conf.make_diffusion_conf().make_sampler()

        eval_conf = copy.deepcopy(conf)
        self.ddpm_sampler = None
        eval_conf.beatgans_gen_type = GenerativeType.ddim
        self.ddim_sampler = eval_conf._make_diffusion_conf(T=15).make_sampler()

        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu', weights_only=False)
            print('step:', state['global_step'])
            state_dct = {}
            for key, val in state['state_dict'].items():
                if 'ema_model' not in key:
                    state_dct[key] = val
            self.model.load_state_dict(state_dct, strict=True)

    # def on_after_backward(self) -> None:
    #     print("on_after_backward enter")
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)
    #     print("on_after_backward exit")

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        if self.train_data is not None:
            print('train data:', len(self.train_data))

    def train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      collate_fn=sparse_batch_collate,
                                      shuffle=True,
                                      drop_last=drop_last)
        return dataloader

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast('cuda', enabled=False):
            imgs_ori, dat, crd, ssz, idxs = batch
            t = torch.randint(0, 1000, size=(imgs_ori.shape[0],))
            r = torch.sparse_coo_tensor(crd.long(), dat, ssz).to_dense()
            # This branch is deprecated
            if self.conf.is_tot:
                pos, imgs = None, imgs_ori
                if self.conf.method == 'pdm':
                    p_list = np.array([0.2, 0.3, 0.5])
                    s_list = np.array([self.conf.image_size//4,
                                       self.conf.image_size//2,
                                       self.conf.image_size])
                    p_sz = int(np.random.choice(s_list, p=p_list))
                    if p_sz != self.conf.image_size:
                        n_s = self.conf.image_size // p_sz
                        imgs = rearrange(imgs, 'b c (n_h h) (n_w w) -> (b n_h n_w) c h w',
                                         n_h=n_s, n_w=n_s)
                        r = rearrange(r, 'b (n_h h) (n_w w) c -> (b n_h n_w) h w c',
                                      n_h=n_s, n_w=n_s)
                        t = torch.randint(0, 1000, size=(imgs.shape[0],))
                    pos = imgs[:, -2:]
                    imgs = imgs[:, :-2]
                                      
                losses = self.sampler.training_losses_tot(model=self.model,
                                                          x=imgs,
                                                          t=t.to(self.device),
                                                          r=r.to(self.device),
                                                          pos=pos)
            else:
                patch_size = self.patch_size
                halfp = patch_size // 2
                H, W = imgs_ori.shape[2:]
                assert H % patch_size == 0 and W % patch_size == 0

                imgs_ori_pad = F.pad(imgs_ori, (halfp, halfp, halfp, halfp))
                p_x = W // patch_size
                p_y = H // patch_size
                grid_x = torch.linspace(0, p_x, p_x + 1).to(self.device)
                grid_y = torch.linspace(0, p_y, p_y + 1).to(self.device)
                xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
                pos = torch.stack([xx, yy], dim=-1)
                loss_mask = torch.zeros_like(imgs_ori_pad)
                loss_mask[:, :, halfp:-halfp, halfp:-halfp] = 1.0

                losses = self.sampler.training_losses(model=self.model,
                                                    x_start=imgs_ori_pad,
                                                    r_start=(dat, crd, ssz),
                                                    imgs=imgs_ori,
                                                    t=t.to(self.device),
                                                    pos=pos,
                                                    loss_mask=loss_mask,
                                                    idx=idxs,
                                                    patch_size=self.patch_size)
            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.num_samples)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # ema(self.model, self.ema_model, self.conf.ema_decay)

            imgs, dat, crd, ssz, idxs = batch
            rnas = (dat.to(imgs.device), crd.to(imgs.device), ssz)
            self.log_sample(x_start=imgs, r_start=rnas,
                            step=self.global_step, idx=idxs)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            for i in range(len(params)):
                if params[i].size() == (90000, 512):
                    params[i]._grad = params[i]._grad * 0.5


    def gen_sample_tot(self, model, x_start, r_start, stype, pos=None):
        _x_T = torch.randn(self.conf.sample_size,
                           *x_start.shape[1:])
        all_x_T = self.split_tensor(_x_T.cuda())
        loader = DataLoader(all_x_T, batch_size=len(all_x_T))
        Gen, achn = [], None
        for x_T in loader:
            bat = min(len(x_T), x_start.shape[0])
            _xstart = x_start[:bat]

            dat, crd, ssz = r_start
            crd = crd.long()
            _b, _h, _w, _g = ssz
            # gn_dbg = torch.sparse.FloatTensor(crd, dat, ssz).detach()
            # gn_dbg = gn_dbg.to_dense()[:new_b]
            bid = crd[0] < bat
            dat, crd = dat[bid], crd[:, bid]
            ssz = torch.Size([bat, _h, _w, _g])
            _rstart = (dat, crd, ssz)
            # assert (gn_dbg == torch.sparse.FloatTensor(crd, dat, ssz).to_dense()).all()

            if stype == 'DDPM':
                sampler = self.ddpm_sampler
            elif stype == 'DDIM':
                sampler = self.ddim_sampler
            gen = sampler.sample(model=model,
                                 shape=_xstart.shape,
                                 noise=x_T[:bat],
                                 pos=None if pos is None else pos[:bat],
                                 r_start=_rstart)
            
            achn = gen.shape[1] 
            if self.conf.stain == 'all':
                achn //= 2
            chn_lst = list(range(achn))
            if achn > 8:
                random.shuffle(chn_lst)
                chn_lst = chn_lst[:8]
                
            rel = _xstart
            for bid in range(gen.shape[0]):
                for gchn in chn_lst:
                    gen_lst = [gen[[bid], gchn]]
                    rel_lst = [rel[[bid], gchn]]
                    if self.conf.stain == 'all':
                        gen_lst = [-torch.ones_like(gen[[bid], gchn]),
                                    gen[[bid], gchn + achn]] + gen_lst
                        rel_lst = [-torch.ones_like(rel[[bid], gchn]),
                                    rel[[bid], gchn + achn]] + rel_lst
                    gen_tsr = torch.stack(gen_lst, 1)
                    rel_tsr = torch.stack(rel_lst, 1)
                    Gen.append(gen_tsr)
                    Gen.append(rel_tsr)

        gen = torch.cat(Gen)
        gen = self.all_gather(gen)
        if gen.dim() == 5:
            gen = gen.flatten(0, 1)

        if self.global_rank == 0:
            # save samples to the tensorboard
            nrow = 2 * min(achn, 8)
            grid = (make_grid(gen, nrow=nrow) + 1) / 2
            sample_dir = os.path.join(self.conf.logdir, 'sample')
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            path = os.path.join(sample_dir, f'{self.global_step}_{stype}.jpg')
            save_image(grid, path)
            self.logger.experiment.add_image('sample', grid,
                                             self.num_samples)

    def gen_sample(self, model, pnm_w, pnm_h, x_start, r_start, stype):
        _x_T = torch.randn(self.conf.sample_size * pnm_w * pnm_h,
                           x_start.shape[1],
                           self.patch_size,
                           self.patch_size)
        all_x_T = self.split_tensor(_x_T.cuda())
        loader = DataLoader(all_x_T, batch_size=len(all_x_T))
        Gen, achn = [], None
        for x_T in loader:
            bat = len(x_T) // pnm_w // pnm_h
            _xstart = x_start[:bat]
            grid_x = torch.linspace(0, pnm_w, pnm_w+1).to(self.device)
            grid_y = torch.linspace(0, pnm_h, pnm_h+1).to(self.device)
            xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
            pos = torch.stack([xx, yy], dim=-1).\
                flatten(0, 1).repeat(_xstart.shape[0], 1)

            dat, crd, ssz = r_start
            crd = crd.long()
            _b, _h, _w, _g = ssz
            # gn_dbg = torch.sparse.FloatTensor(crd, dat, ssz).detach()
            # gn_dbg = gn_dbg.to_dense()[:new_b]
            bid = crd[0] < bat
            dat, crd = dat[bid], crd[:, bid]
            ssz = torch.Size([bat, _h, _w, _g])
            _rstart = (dat, crd, ssz)
            # assert (gn_dbg == torch.sparse.FloatTensor(crd, dat, ssz).to_dense()).all()

            if stype == 'DDPM':
                sampler = self.ddpm_sampler
            elif stype == 'DDIM':
                sampler = self.ddim_sampler
            gen = sampler.sample(model=model,
                                 shape=_xstart.shape,
                                 noise=x_T.detach(),
                                 r_start=_rstart,
                                 patch_size=self.patch_size,
                                 pos=pos)
            
            achn = gen.shape[1] 
            if self.conf.stain == 'all':
                achn //= 2
            chn_lst = list(range(achn))
            if achn > 8:
                random.shuffle(chn_lst)
                chn_lst = chn_lst[:8]
                
            rel = _xstart
            for bid in range(gen.shape[0]):
                for gchn in chn_lst:
                    gen_lst = [gen[[bid], gchn]]
                    rel_lst = [rel[[bid], gchn]]
                    if self.conf.stain == 'all':
                        gen_lst = [-torch.ones_like(gen[[bid], gchn]),
                                    gen[[bid], gchn + achn]] + gen_lst
                        rel_lst = [-torch.ones_like(rel[[bid], gchn]),
                                    rel[[bid], gchn + achn]] + rel_lst
                    gen_tsr = torch.stack(gen_lst, 1)
                    rel_tsr = torch.stack(rel_lst, 1)
                    Gen.append(gen_tsr)
                    Gen.append(rel_tsr)

        gen = torch.cat(Gen)
        gen = self.all_gather(gen)
        if gen.dim() == 5:
            gen = gen.flatten(0, 1)

        if self.global_rank == 0:
            # save samples to the tensorboard
            nrow = 2 * min(achn, 8)
            grid = (make_grid(gen, nrow=nrow) + 1) / 2
            sample_dir = os.path.join(self.conf.logdir, 'sample')
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            path = os.path.join(sample_dir, f'{self.global_step}_{stype}.jpg')
            save_image(grid, path)
            self.logger.experiment.add_image('sample', grid,
                                             self.num_samples)

    def log_sample(self, x_start, r_start, idx, **kwargs):
        """
        put images to the tensorboard
        """

        if self.conf.sample_every_samples > 0 and \
            (self.global_step == 1 or self.global_step % 2500 == 0):
            self.model.eval()
            with torch.no_grad():
                if self.conf.is_tot:
                    pos = None
                    if self.conf.method == 'pdm':
                        pos = x_start[:, -2:]
                        x_start = x_start[:, :-2]
                    self.gen_sample_tot(self.model, x_start, r_start, 'DDIM', pos)
                else:   
                    H, W = x_start.shape[2:]
                    pnm_w = W // self.patch_size
                    pnm_h = H // self.patch_size
                    self.gen_sample(self.model, pnm_w, pnm_h, x_start, r_start, 'DDIM')
            self.model.train()

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = LambdaLR(optim, 
                             lr_lambda=WarmupLR(self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}, {n}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def train(conf: TrainConfig, gpus, nodes=1):
    print('conf:', conf.name)
    model = LitModel(conf)
    callbacks = [LearningRateMonitor()]
    logdir = Path(conf.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=-1,
                                 every_n_train_steps=10000)
    callbacks.append(checkpoint)

    resume = None
    if conf.full_model_path:
        checkpoint_path = conf.full_model_path
    else:
        checkpoint_path = f'{conf.logdir}/last.ckpt'
    if Path(checkpoint_path).is_file():
        resume = checkpoint_path
    elif conf.continue_from is not None:
        resume = conf.continue_from.path
    print(f'resume from {resume}')

    trainer = pl.Trainer(
        max_steps=conf.total_samples,
        devices=gpus,
        num_nodes=nodes,
        accelerator='cuda',
        precision='16-mixed' if conf.fp16 else 32,
        callbacks=callbacks,
        use_distributed_sampler=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        strategy='ddp')
    print(f'trainer initialized: device={trainer.device_ids} strategy={trainer.strategy}')
    print(f'trainer.world_size={trainer.world_size}')
    print(f'trainer.strategy.cluster_environment.world_size()={trainer.strategy.cluster_environment.world_size()}')
    print(f'trainer.strategy._accelerator.auto_device_count={trainer.strategy._accelerator.auto_device_count()}')
    print(f'trainer.strategy.world_size={trainer.strategy.world_size}')
    trainer.fit(model, ckpt_path=resume)
