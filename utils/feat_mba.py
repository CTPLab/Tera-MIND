
import os
import clip
import zarr
import torch
import sparse
import pickle
import shutil
import random
import urllib
import argparse
import itertools
import contextlib
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from torch import nn
from tqdm import tqdm
from pathlib import Path
from PIL import ImageFile
from einops import rearrange
from cellpose import models as cmodels

from utils.metrics import calc_cellpose
ImageFile.LOAD_TRUNCATED_IMAGES = True


inception_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"

"""
Download the pretrined inception weights if it does not exists
ARGS:
    fpath - output folder path
"""
def check_download_inception(fpath="./"):
    inception_path = os.path.join(fpath, "inception-2015-12-05.pt")
    if not os.path.exists(inception_path):
        # download the file
        with urllib.request.urlopen(inception_url) as response, open(inception_path, 'wb') as f:
            shutil.copyfileobj(response, f)
    return inception_path

@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    # On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run. See
    #   https://github.com/GaParmar/clean-fid/issues/5
    #   https://github.com/pytorch/pytorch/issues/64062
    if torch.__version__.startswith('1.9.'):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith('1.9.'):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


class InceptionV3W(nn.Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

    path: locally saved inception weights
    """
    def __init__(self, path, download=True, resize_inside=False):
        super(InceptionV3W, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        if download:
            check_download_inception(fpath=path)
        path = os.path.join(path, "inception-2015-12-05.pt")
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers
        self.resize_inside = resize_inside

    """
    Get the inception features without resizing
    x: Image with values in range [0,255]
    """
    def forward(self, x):
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            if self.resize_inside:
                features = self.base(x, return_features=True).view((bs, 2048))
            else:
                # make sure it is resized already
                assert (x.shape[2] == 299) and (x.shape[3] == 299)
                # apply normalization
                # https://github.com/mseitzer/pytorch-fid/issues/3
                # https://github.com/mseitzer/pytorch-fid/issues/112
                x = x / 127.5 - 1
                features = self.layers.forward(x, ).view((bs, 2048))
            return features


class InceptionV3S(nn.Module):
  def __init__(self, path, download=True, resize_inside=False,
               use_fp16=False) -> None:
    super().__init__()
    self.use_fp16 = use_fp16
    if download:
        check_download_inception(fpath=path)
    path = os.path.join(path, "inception-2015-12-05.pt")
    self.base = torch.jit.load(path).eval()
    layers = self.base.layers
    self.layers = nn.Sequential(
      layers.conv,
      layers.conv_1,
      layers.conv_2,
      layers.pool0,
      layers.conv_3,
      layers.conv_4,
      layers.pool1,
      layers.mixed,
      layers.mixed_1,
      layers.mixed_2,
      layers.mixed_3,
      layers.mixed_4,
      layers.mixed_5,
      layers.mixed_6.conv)

  def forward(self, x: torch.Tensor) -> torch.FloatTensor:
    _, _, h, w = x.shape
    assert h == 299 and w == 299
    if self.use_fp16:
      x = x.to(torch.float16)
    x = x / 127.5 - 1
    features: torch.FloatTensor = self.layers.forward(x)
    # we have a 17x17 feature map. taking the first 7 channels (7*17*17=2023)
    # gives us a comparable size to the 2048 pool_3 feature vector.
    features = features[:,:7,:,:].flatten(start_dim=1)
    return features.float()


"""
returns a functions that takes an image in range [0,255]
and outputs a feature embedding vector
"""
def feature_extractor(name="torchscript_inception", device=torch.device("cuda"), resize_inside=False, use_dataparallel=True):
    if "torchscript_inception" in name:
        Inception = InceptionV3W if name == "torchscript_inception" else InceptionV3S
        model = Inception("Data/", download=True, resize_inside=resize_inside).to(device)
        model.eval()
        if use_dataparallel:
            model = torch.nn.DataParallel(model)
        def model_fn(x): return model(x)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


"""
Build a feature extractor for each of the modes
"""
def build_feature_extractor(mode, device=torch.device("cuda"), use_dataparallel=True):
    if mode == "legacy_pytorch":
        feat_model = feature_extractor(name="pytorch_inception", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    elif mode == "clean_s":
        feat_model = feature_extractor(name="torchscript_inception_s", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    elif mode == "clean":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    return feat_model


class CLIP_fx():
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.float().eval()
        self.name = "clip_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, img_t):
        img_x = F.normalize(img_t, 
                            (0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        with torch.no_grad():
            z = self.model.encode_image(img_x)
        return z


def post_dim(x, batch_dim=False):
    assert len(x.shape) in (3, 4)
    if len(x.shape) == 3 and batch_dim:
        x = x[None]
    if len(x.shape) == 3:
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 2:
            x = torch.cat((torch.zeros_like(x[0])[None], x), 0)
    else:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat((torch.zeros_like(x[:, 0])[:, None], x), 1)
    return x


def fn_resize(x, sz, mode='clean', batch_dim=False):
    assert 'float32' in str(x.dtype)
    if mode == 'clip':
        x = x.byte()
    # Bicubic is 3
    x = F.resize(x, sz, 3, antialias=True).clamp(0, 255)
    if mode == 'clip':
        # simulate toTensor
        x = x.float().div(255)
    x = post_dim(x, batch_dim)
    return x


class ResizeDataset_gn(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, step=4):
        self.pths = files
        self.step = step
    def __len__(self):
        return len(self.pths)
    
    def _getgene(self, pth, sz=128):
        gn = sparse.load_npz(str(pth))
        h, w = gn.shape[:2]
        gn = gn[h//2-sz:h//2+sz,
                w//2-sz:w//2+sz].sum((0,1))
        return torch.from_numpy(gn.todense().astype(float))
    
    def __getitem__(self, i):
        gn_lst = [self._getgene(p) for p in self.pths[i]]
        out = rearrange(torch.stack(gn_lst),
                        'c (z g) -> z (c g)', g=500)
        return out.sum((-1))


class ResizeDataset_MBA(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, stain, size=(299, 299), 
                 step=4, nslc=50, is_cellps=False):
        self.pths = files
        self.mode = mode
        self.step = step
        self.slc = {'DAPI': [slice(None, nslc), 1], 
                    'PolyT': [slice(nslc, None), 1],
                    'all': [slice(None, None), 2]}[stain]
        self.is_cellps = is_cellps
    
    def __len__(self):
        return len(self.pths)
    
    def _getgene(self, pth, sz=128):
        pth = str(pth).replace('img', 'gene').replace('.zip', '.npz')
        gn = sparse.load_npz(pth)
        h, w = gn.shape[:2]
        gn = gn[h//2-sz:h//2+sz,
                w//2-sz:w//2+sz].sum((0,1))
        return torch.from_numpy(gn.todense().astype(float))
    
    def _getimg(self, pth, sz=128):
        im = zarr.load(str(pth))
        h, w = im.shape[1:]
        im = im[self.slc[0], 
                h//2-sz:h//2+sz,
                w//2-sz:w//2+sz]
        return torch.from_numpy(im).float()
    
    def __getitem__(self, i):
        if self.is_cellps:
            im_lst = [self._getimg(p) for p in self.pths[i]]
            out = rearrange(torch.stack(im_lst),
                            '(p1 p2) (s z) h w -> z s (p1 h) (p2 w)',
                            p1=self.step, p2=self.step, s=self.slc[1])
            gn_lst = [self._getgene(p) for p in self.pths[i]]
            gn = rearrange(torch.stack(gn_lst),
                           'c (z g) -> z (c g)', 
                           g=500).sum((-1))
            pnm = Path(self.pths[i][0]).stem
            pnm = pnm.split('_')
            pnm = np.array([int(_p) for _p in pnm])
            return (gn != 0)[:, None, None, None] * out, torch.from_numpy(pnm)
        else:
            im = self._getimg(self.pths[i])
            im = rearrange(im, '(s z) h w -> z s h w', 
                           s=self.slc[1])
            gn = self._getgene(self.pths[i])
            gn = rearrange(gn, '(z g) -> z g', 
                           g=500).sum((-1))
            if 'clean' in self.mode:
                sz = (299, 299)
            elif self.mode == 'clip':
                sz = (224, 224)
            out = [fn_resize(im[:, [0]], sz, self.mode),
                   fn_resize(im[:, [1]], sz, self.mode),
                   fn_resize(im, sz, self.mode)]
        return torch.stack(out), gn != 0


EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy', 'JPEG', 'JPG', 'PNG'}


def get_batch_features(batch, model, device):
    with torch.inference_mode():
        feat = model(batch.to(device))
    return feat.detach()


def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None,
                       description="", fdir=None, verbose=True,
                       stain='all', custom_image_tranform=None):
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset_MBA(l_files, mode=mode, stain=stain)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    
    mu, scm, tot, cnt = 0, 0, 0, 0
    for batch, gmsk in pbar:
        _b, _r = batch.shape[:2]
        batch = rearrange(batch, 
                          'b resz z s h w -> (resz z b) s h w')
        feat = get_batch_features(batch, model, device)
        feat = rearrange(feat, '(resz z b) f -> resz z b f', 
                         resz=_r, b=_b)
        gmsk = rearrange(gmsk, 'b z -> z b')[None].to(feat)
        feat = gmsk[..., None] * feat
        mu += feat.sum(-2)
        scm += feat.mT @ feat
        tot += gmsk.sum(-1)
        cnt += 1
        # if cnt == 100:
        #     break
    # mu /= tot[..., None]
    # scm /= tot[..., None, None]
    # sigma = scm - mu[:, :, :, None] @ mu[:, :, None]
    print(mu.shape, scm.shape, tot.shape, tot)
    return mu, scm, tot


def make_custom_stats(name, path, num=None, mode="clean", model_name="inception_v3",
                      stain='all', num_workers=0, batch_size=2, device=torch.device("cuda"),
                      verbose=True, shuffle=False, seed=None, custom_fn_resize=None,
                      buffer=1000000):
    stats_folder = "stats_mba"
    os.makedirs(stats_folder, exist_ok=True)
    outf = os.path.join(
        stats_folder, f"{name}_{mode}_{stain}.pt")
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)
    if model_name == "inception_v3":
        feat_model = build_feature_extractor(mode, device)
        custom_fn_resize = None
        custom_image_tranform = None
    elif model_name == "clip_vit_b_32":
        clip_fx = CLIP_fx("ViT-B/32")
        feat_model = clip_fx
        custom_fn_resize = None
        custom_image_tranform = None
    else:
        raise ValueError(
            f"The entered model name - {model_name} was not recognized.")

    # get all relevant files in the dataset
    if verbose:
        print(f"Found {len(path)} images")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(path)
        path = path[:num]
    feats = get_files_features(path, feat_model, num_workers=num_workers,
                               batch_size=batch_size, device=device, mode=mode,
                               custom_fn_resize=custom_fn_resize,
                               custom_image_tranform=custom_image_tranform,
                               verbose=verbose, stain=stain)
    # if feats.shape[0] > buffer:
    #     mu, scm, buf, tot = 0, 0, buffer // 5,  feats.shape[0]
    #     bat = math.ceil(tot / buf)
    #     for b in range(bat):
    #         f = feats[b * buf: (b + 1) * buf].double()
    #         mu += f.sum(0)
    #         scm += f.T @ f
    #     mu /= tot
    #     scm /= tot
    #     sigma = scm - mu[:, None] @ mu[None]
    # else:
    #     feats = feats.double()
    #     mu = feats.mean(0)
    #     scm = (feats.T @ feats) / feats.shape[0]
    #     sigma = feats.T.cov(correction=0)
    #     _err = (scm - mu[:, None] @ mu[None]) - sigma
    #     print(f'{_err.abs().max()}', feats.shape, scm.shape, sigma.shape)
    torch.save(feats, outf)
    # del mu, sigma, scm


def is_valid(pnm, gnm):
    for p in pnm:
        if Path(p).stem not in gnm:
            return False
    return True

def dt_region(dir, hst, wst, hnm, wnm, 
              size=256, is_gt=True, ext='.npz'):
    pad = size // 2
    dt_lst = []
    for ph in range(hnm):
        hsz = ph * size
        for pw in range(wnm):
            wsz = pw * size
            pnm = [hst + hsz, hst + size + hsz, 
                   wst + wsz, wst + size + wsz]
            if is_gt:
                pnm += [hst - pad + hsz, hst + size + pad + hsz,
                        wst - pad + wsz, wst + size + pad + wsz]
            pnm = [str(p) for p in pnm]
            pnm = '_'.join(pnm)
            pth = dir / f'{pnm}{ext}'
            dt_lst.append(str(pth))
    return dt_lst


def dt_sublst(dir, gnm,
              hst=256, wst=256, hnm=16, wnm=16,
              size=256, step=1, is_gt=True, ext='.npz'):
    # for 1024 x 1024, hst=wst=512, 
    # hnm=412, wnm=284, step = 4
    dt_lst= []
    # Here, we assume no boundary issue
    for pw in range(0, wnm, step):
        wsz = pw * size
        for ph in range(0, hnm, step):
            hsz = ph * size
            dt_reg = dt_region(dir, hst+hsz, wst+wsz,
                               step, step, 
                               is_gt=is_gt, ext=ext)
            if is_valid(dt_reg, gnm):
                dt_lst.append(dt_reg)
    return dt_lst


def debug_stats(mse):
    gn_dir = Path(f'Data/MERFISH_50/gene_{mse}')
    gn_pth = pd.read_csv(f'utils/{mse}_exl.csv')['pth'].to_list()
    gn_pth = [Path(p).stem for p in gn_pth]
    gn_lst = dt_sublst(gn_dir, gn_pth,
                       11264, 21504, 200, 248, 
                       step=args.step, ext='.npz')
    random.seed(int(mse))
    random.shuffle(gn_lst)
    print(len(gn_lst))
    data = ResizeDataset_gn(gn_lst)
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=2, shuffle=True,
        drop_last=False, num_workers=2)
    acc = torch.zeros((50))
    for gid, gn in enumerate(dataloader):
        acc += (gn != 0).sum(0)
        if (gid + 1) % 10 == 0:
            print(gid, acc, acc/(2*(gid+1)))

def prep_stats(mse, args):
    im_dir = Path(f'Data/MERFISH_50/img_{mse}')
    gn_pth = pd.read_csv(f'utils/{mse}_exl.csv')['pth'].to_list()
    gn_pth = [Path(p).stem for p in gn_pth]
    # For compatible data loading for both FID-like scores and cellpose derived ones,
    # we load 1024 x 1024 region as one batch of data  
    # Reduce 1/4 of WSI search window to avoid black regions
    # 200 * 248 = 49600 \approx 50000
    # 11264, 21504 are the left, top pos of the centered window
    gn_lst = dt_sublst(im_dir, gn_pth,
                       11264, 21504, 200, 248, 
                       step=args.step, ext='.zip')
    print(len(gn_lst))
    assert args.stain == 'all'
    if args.is_cellpose:
        data = ResizeDataset_MBA(gn_lst, mode=None, 
                                 is_cellps=True, stain=args.stain)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=True,
            drop_last=False, num_workers=2)
        model = cmodels.Cellpose(gpu=True, model_type='cyto')
        stat_dir = Path(f'stats_mba/{args.mouse}')
        stat_dir.mkdir(parents=True, exist_ok=True)
        reg_lst = []
        for iid, (img, roi) in enumerate(dataloader):
            if iid % 100 == 0:
                print(iid, img.shape)
            # if iid == 99:
            #     break
            img = img.squeeze().numpy()
            roi = roi[0].numpy().tolist()
            roi = '_'.join(map(str, roi))
            calc_cellpose(img, model, stat_dir, roi, args.debug)
            reg_lst.append(roi)
        with open(f'utils/{mse}_cell.pickle', 'wb') as f:
            pickle.dump(reg_lst, f)
        print(f'{mse} is done.')
    else:
        gn_lst = list(itertools.chain(*gn_lst))
        model = 'inception_v3' if 'clean' in args.mode else 'clip_vit_b_32'
        make_custom_stats(mse, gn_lst, stain=args.stain,
                          mode=args.mode, model_name=model,
                          num_workers=2)
        print(f'{mse}_{args.mode}_{args.stain} is done.')

if __name__ == '__main__':
    # model = InceptionV3S('/tmp').cuda()
    # a = torch.rand(5, 3, 299, 299).cuda()
    # out = model(a)
    # print(out.shape)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--stain', type=str, default='all',
                        choices=['DAPI', 'PolyT', 'all'])
    parser.add_argument('--step', type=int,
                        default=4,
                        help='The region size 4 x 256 = 1024')
    parser.add_argument('--is_cellpose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gn_debug', action='store_true')
    parser.add_argument('--mode', type=str, default='clean',
                        choices=['clip', 'clean', 'clean_s'])
    args = parser.parse_args()

    mse = args.mouse
    if args.gn_debug:
        debug_stats(mse)
    else:
        prep_stats(mse, args)

