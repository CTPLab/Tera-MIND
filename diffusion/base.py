"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import math
import random
import numpy as np
import torch as th
import torch.nn.functional as F

from model import Model
from model.nn import mean_flat
from torch.amp import autocast
from config_base import BaseConfig
from utils.choices import GenerativeType, ModelMeanType, ModelType, ModelVarType, LossType

from typing import Tuple
from dataclasses import dataclass
from einops import rearrange,repeat


@dataclass
class GaussianDiffusionBeatGansConfig(BaseConfig):
    method:str
    image_size: int
    input_size: int
    gn_sz: int
    is_tot: bool
    gen_type: GenerativeType
    betas: Tuple[float]
    model_type: ModelType
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    rescale_timesteps: bool
    fp16: bool
    rna_tpl: Tuple
    train_pred_xstart_detach: bool = True
    cfg: bool = True

    def make_sampler(self):
        return GaussianDiffusionBeatGans(self)


class GaussianDiffusionBeatGans:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, conf: GaussianDiffusionBeatGansConfig):
        self.conf = conf
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))
        assert self.loss_type in [LossType.mse, LossType.l1]
        assert self.model_mean_type == ModelMeanType.eps
        self.tl2im = '(b p1 p2) c h w -> b c (p1 h) (p2 w)'
        self.im2tl = 'b c (p1 h) (p2 w) -> (b p1 p2) c h w'
    
    def sparse_repatch(self, rna, sz):
        dat, crd, ssz = rna
        # r_dbg = th.sparse.FloatTensor(crd, dat, ssz).to_dense()
        # r_dbg = rearrange(r_dbg, 'b (p1 h) (p2 w) g-> (b p1 p2) h w g', 
        #                     h=sz, w=sz)
        _p1, _p2 = ssz[1] // sz, ssz[2] // sz
        crd[0] = crd[0] * _p1 * _p2 + (crd[1] // sz) * _p2 + crd[2] // sz
        crd[1], crd[2] = crd[1] % sz, crd[2] % sz
        ssz = th.Size([ssz[0] * _p1 * _p2, sz, sz, ssz[-1]])
        return (dat, crd, ssz)
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # scale t to be maxed out at 1000 steps
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                           t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def training_losses(self,
                        model:Model,
                        x_start:th.Tensor,
                        r_start,
                        imgs:th.Tensor,
                        t:th.Tensor,
                        pos:th.Tensor,
                        loss_mask:th.Tensor,
                        idx=None,
                        patch_size=64,
                        model_kwargs=None,
                        noise: th.Tensor=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the padded (large) input image.
        :param r_start: the sparse rna tensor.
        :param imgs: the (large) input image.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        
        halfp = patch_size // 2

        t_cur = repeat(t, 'h -> (h repeat)',
                       repeat=int(x_start.shape[0]/t.shape[0]))
        x_t = self.q_sample(x_start, t_cur, noise=noise)
        if loss_mask is not None:
            x_t = x_t * loss_mask
        terms = {'x_t': x_t}

        index_x = random.randrange(pos.shape[0]-1)
        index_y = random.randrange(pos.shape[1]-1)
        index = [index_x, index_y]
        index = th.tensor(index, device = 'cuda')
        assert r_start is not None
        dat, crd, ssz = r_start
        r_size = patch_size // (x_start.shape[2] // ssz[1])

        # rna crop and patchify
        i_crop = (index_x*r_size <= crd[1]) & (crd[1] < (index_x+2)*r_size) & \
            (index_y*r_size <= crd[2]) & (crd[2] < (index_y+2)*r_size)
        dat, crd = dat[i_crop], crd[:, i_crop]
        crd[1] -= index_x*r_size
        crd[2] -= index_y*r_size
        rna = (dat, crd, (ssz[0], 2*r_size, 2*r_size, ssz[-1]))
        rna_pat = self.sparse_repatch(rna, r_size)

        # img crop and patchify
        pos = pos[index_x:index_x+2, index_y:index_y+2].flatten(0,1).repeat(idx.shape[0], 1)
        slc = (slice(None),) + (slice(None),) + \
            (slice(index_x*patch_size, (index_x+2)*patch_size),) + \
            (slice(index_y*patch_size, (index_y+2)*patch_size),)
        slc_pat = []
        for pat in (x_t, noise, loss_mask):
            pat = pat[slc]
            slc_pat.append(rearrange(pat, self.im2tl, h=patch_size, w=patch_size))
        x_t, noise, loss_mask = slc_pat

        if self.conf.cfg:
            s_random = th.tensor(np.random.random(t.shape[0])).to(t.device)
            pos_random = th.tensor(np.random.random(t.shape[0])).to(t.device)
        else:
            s_random = None
            pos_random = None
        with autocast('cuda', enabled=self.conf.fp16):
            # x_t is static wrt. to the diffusion process
            model_forward = model.forward(
                x=x_t.detach(),
                t=self._scale_timesteps(t),
                r=None if r_start is None else rna_pat,
                pos=pos.detach(),
                imgs=imgs.detach(),
                idx=idx.detach(),
                index=index,
                do_train=True,
                patch_size=patch_size,
                random=s_random,
                pos_random=pos_random,
                **model_kwargs)
        output = model_forward.pred2
        output_shift = model_forward.pred

        noise_ori = rearrange(noise, self.tl2im, p1=2, p2=2)
        if loss_mask is None: # current is 4*4
            noise_ori = F.pad(noise_ori, (halfp, halfp, halfp, halfp), 'constant')
        else:
            noise_ori = noise_ori[:, :, halfp:-halfp, halfp:-halfp]
        noise_shift = rearrange(noise_ori, self.im2tl, h=patch_size, w=patch_size)
        assert output.shape == noise.shape
        assert output_shift.shape == noise_shift.shape

        if self.loss_type == LossType.mse:
            # (n, c, h, w) => (n, )
            terms['loss'] = mean_flat((noise_shift - output_shift) ** 2).mean()
            terms['loss'] += mean_flat((noise - output) ** 2 * loss_mask).mean()
        elif self.loss_type == LossType.l1:
            terms['loss'] = mean_flat((noise_shift - output_shift).abs()).mean()
            terms['loss'] += mean_flat((noise - output).abs() * loss_mask).mean()
        return terms
    
    def sample(self,
               model: Model,
               shape=None,
               noise=None,
               pos=None,
               cond=None,
               x_start=None,
               r_start=None,
               imgs=None,
               clip_denoised=True,
               idx = None,
               patch_size = 64,
               model_kwargs=None,
               progress=False):
        """
        Args:
            x_start: given for the autoencoder
        """
        if model_kwargs is None:
            model_kwargs = {'imgs': None}
            if self.conf.model_type.has_autoenc():
                model_kwargs['x_start'] = x_start
                # If idx is given, meaning that we are in the test mode
                # shouldn't assign imgs after shifting
                model_kwargs['imgs'] = imgs if idx is None else None
                model_kwargs['cond'] = cond

        if self.conf.gen_type == GenerativeType.ddpm:
            stype = 'DDPM'
        elif self.conf.gen_type == GenerativeType.ddim:
            stype = 'DDIM' 
        return self.ddm_sample_loop(model,
                                    shape=shape,
                                    noise=noise,
                                    rna=r_start,
                                    pos=pos,
                                    idx=idx,
                                    img=imgs, 
                                    clip_denoised=clip_denoised,
                                    model_kwargs=model_kwargs,
                                    progress=progress,
                                    sample_type=stype)

    def p_mean_variance(self,
                        model: Model,
                        x,
                        t,
                        r,
                        shapes = (1,3,256,256),
                        clip_denoised=True,
                        denoised_fn=None,
                        idx = None,
                        patch_size = 64,
                        pos = None,
                        model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))
        
        B,C,H,W = shapes
        p_x = H // patch_size
        p_y = W // patch_size
        halfp = patch_size // 2
        if model_kwargs['imgs'] is None:
            model_kwargs['imgs'] = th.randn(shapes).to(x.device)
        
        with autocast('cuda', enabled=self.conf.fp16):
            model_forward = model.forward(
                x=x,
                t=self._scale_timesteps(t),
                r=r,
                idx=idx,
                patch_size=patch_size,
                pos=pos,
                **model_kwargs)
        output = model_forward.pred
        if output.size(0) % (p_x * p_y) == 0 and output.size(0) // (p_x * p_y) == t.shape[0]: # should be 17 * 17 originally
            noise_ori = rearrange(output, self.tl2im, p1=p_x, p2=p_y)
            noise_ori = F.pad(noise_ori, (halfp, halfp, halfp, halfp), 'constant', -1)
        elif output.size(0) % ((p_x+1) * (p_y+1)) == 0 and output.size(0) // ((p_x+1) * (p_y+1)) == t.shape[0]: #should be 16 * 16 
            noise_ori = rearrange(output, self.tl2im, p1=p_x+1, p2=p_y+1)
            noise_ori = noise_ori[:, :, halfp:-halfp, halfp:-halfp]
        output = rearrange(noise_ori, self.im2tl, h=patch_size, w=patch_size)
            
        # if self.conf.cfg:
        #     w=0.5
        #     e_t_cond, e_t_uncond = model_output.chunk(2)
        #     model_output = (1 + w) * e_t_cond - w * e_t_uncond
        #     x = x.chunk(2)[0]
        #     t = t.chunk(2)[0]
        #     t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))

        if self.model_var_type in [
                ModelVarType.fixed_large, ModelVarType.fixed_small
        ]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t_cur, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t_cur,
                                                      x.shape)

        pred_xstart = self._predict_xstart_from_eps(
            x_t=x, t=t_cur, eps=output)
        pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t_cur)[0]

        assert (model_mean.shape == model_log_variance.shape ==
                pred_xstart.shape == x.shape)
        return {"mean": model_mean,
                "variance": model_variance,
                "log_variance": model_log_variance,
                "pred_xstart": pred_xstart,
                'model_forward': model_forward,
                "noise": output}

    def ddm_sample(
        self,
        model: Model,
        x,
        t,
        r,
        noise_patch=None,
        shapes = (1,3,256,256),
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        patch_size = 64,
        model_kwargs=None,
        eta=0.0,
        sample_type='DDIM',
        pos=None):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            r,
            shapes = shapes,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            patch_size = patch_size,
            pos=pos,
            model_kwargs=model_kwargs,
        )
        if self.conf.cfg:
            x = x.chunk(2)[0]
            t = t.chunk(2)[0]
        t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))
         # no noise when t == 0
        nonzero_mask = ((t_cur != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        if sample_type == 'DDPM':
            noise = th.randn_like(x) if noise_patch is None else noise_patch
            sample = out["mean"] + nonzero_mask * \
                  th.exp(0.5 * out["log_variance"]) * noise
        elif sample_type == 'DDIM':
            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t_cur, out["pred_xstart"])

            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t_cur, x.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t_cur,
                                                x.shape)
            sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                    th.sqrt(1 - alpha_bar / alpha_bar_prev))
            # Equation 12.
            mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                        th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
            sample = mean_pred
            if eta != 0:
                noise = th.randn_like(x) if noise_patch is None else noise_patch
                sample += nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddm_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        rna=None,
        pos=None,
        idx=None,
        img=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        sample_type='DDIM'):

        final = None
        for sample in self.ddm_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            rna=rna,
            pos=pos,
            idx=idx,
            img_patch=img,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            sample_type=sample_type):
            final = sample
        return final

    def ddm_sample_loop_progressive(
        self,
        model: Model,
        shapes=None,
        noise=None,
        rna=None,
        pos=None,
        idx=None,
        img_patch=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        sample_type='DDIM'):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        
        if img_patch is not None:
            assert idx is not None
        img = th.randn(shapes, device=device)
        b,c,H,W = shapes
        patch_size = noise.shape[2]
        halfp = patch_size // 2
        p_x = H // patch_size
        p_y = W // patch_size
        indices = list(range(self.num_timesteps))[::-1] if idx is None else [idx,]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        if self.conf.cfg: # classifier free-guidance
            pos_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            seman_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            model_kwargs["random"] = seman_random
            model_kwargs["pos_random"] = pos_random
            model_kwargs['pos'] = th.cat([model_kwargs['pos']]*2, dim = 0)
            model_kwargs['cond'] = th.cat([model_kwargs['cond']]*2, dim = 0)
            shapes = (2*b,c,H,W)

        rna_msk = None
        if th.is_tensor(rna):
            rna_new = rna
        elif isinstance(rna, (tuple, list)) and len(rna) == 2:
            rna_new, rna_msk = rna
        else:
            r_sz = patch_size // ((img.shape[2] + patch_size) // rna[2][1])
            rna_new = self.sparse_repatch(rna, r_sz)
            print(f'{sample_type}: {len(indices)}')
        for i in indices:
            
            t = th.tensor([i] * b, device=device) # change_code_note

            if self.conf.cfg:
                img = th.cat([img]*2, dim = 0)
                t = th.cat([t]*2)

            if img_patch is None:
                img_new = F.pad(img, (halfp, halfp, halfp, halfp), 'constant')
                img_new = rearrange(img_new, self.im2tl, h=patch_size, w=patch_size)
            else:
                img_new = img_patch
            
            with th.no_grad():
                out = self.ddm_sample(
                    model,
                    img_new,
                    t,
                    rna_new,
                    noise_patch= None if img_patch is None else noise,
                    shapes = shapes,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    patch_size = patch_size,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    sample_type=sample_type,
                    pos=pos)
            img_new = rearrange(out['sample'], self.tl2im, p1=p_x+1, p2=p_y+1)
            img = img_new[:, :, halfp:-halfp, halfp:-halfp]
            if rna_msk is not None:
                img = img * rna_msk + rna_msk - 1
            yield img

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2,
        )
    elif schedule_name == "const0.01":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.01] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.015] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0065":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0065] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0055":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0055] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0045":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0045] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0035":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0035] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0025":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0025] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0015] * num_diffusion_timesteps,
                        dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)