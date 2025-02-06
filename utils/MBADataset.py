import zarr
import torch
import random
import sparse
import pickle
import itertools
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from utils import MOUSE
from random import shuffle
from einops import rearrange
from torch.utils.data import Dataset


class MBADataset(Dataset):
    def __init__(self, mous,
                 gdim, gblk,
                 sdim, pdim,
                 snum=None,
                 stain='DAPI',
                 repeat=10,
                 transform=False,
                 debug=False,
                 methd='ours',
                 use_exl=False):

        assert mous in MOUSE 
        assert snum in (None, 1, 4, 8, 16)
        assert stain in ('DAPI', 'PolyT', 'all')  
        self.gmax = MOUSE[mous][0] + 1
        # padding along slice
        # make sure (50 + 2 * spad) / (snum / 2) - 1 is an integer
        # for snum in (8, 16) we lost 2 boundary slices  
        self.spad = {None:None, 1:0, 4:1, 8:1, 16:3}[snum] 
        
        # The valid slices for training 
        self.gdim = gdim  # gdim-plex of gene
        self.gblk = gblk  # Size of small gene array to be summed
        self.sdim = sdim  # Spatial dim of gene: same as the (output) img dim
        self.pdim = pdim # gene pad dim
        # Subset of stain, gene section
        self.snum, self.stain = snum, stain
        self.transform = transform
        self.debug = debug
        self.methd = methd
        
        _exl = '_exl' if use_exl else ''
        if mous == '609882':
            self.gn_pth = pd.read_csv(f'utils/609889{_exl}.csv')['pth'].to_list()
        elif mous == '609889':
            self.gn_pth = pd.read_csv(f'utils/609882{_exl}.csv')['pth'].to_list()
        elif mous == '638850':
            gn_p0 = pd.read_csv(f'utils/609882{_exl}.csv')['pth'].to_list()
            gn_p1 = pd.read_csv(f'utils/609889{_exl}.csv')['pth'].to_list()
            self.gn_pth = gn_p0 + gn_p1
        if repeat > 1:
            _pth = list(itertools.repeat(self.gn_pth, repeat))
            self.gn_pth = list(itertools.chain(*_pth))
        shuffle(self.gn_pth)

        if self.debug:
            # If take genes for all slices
            # then canont fit in the memory for debugging
            assert snum == 1
        print(f'Training data: {len(self.gn_pth)}; Slices: {self.snum}')

    def _getgene(self, pth, gnm, debug):
        gene = sparse.load_npz(pth)
        top = random.randint(0, gene.shape[0]-self.sdim)
        left = random.randint(0, gene.shape[1]-self.sdim)
        
        # The h and w channels should be 1st and 2nd dim
        # Otherwise the crop maybe be slow
        gn_crop = gene[top:top+self.sdim,
                       left:left+self.sdim]
        gn = gn_crop[:, None, :, None]
        gn = gn.reshape((self.sdim//self.gblk, self.gblk,
                         self.sdim//self.gblk, self.gblk, -1))
        gn = gn.sum((1, 3))
        if gnm is not None:
            # The step shoud be after gn=gn.sum((1, 3))
            # Otherwise it is very slow
            if self.snum > 1:
                gn.coords[2] += (self.spad * 500)
                shp = list(gn.shape)
                shp[-1] += self.spad * 1000
                gn = sparse.COO(gn.coords, gn.data, 
                                shape=tuple(shp))
            gn = gn[:, :, gnm*500:(gnm+self.snum)*500]
            if debug:
                gn_crop = gn_crop[:, :, 
                                  gnm*500:(gnm+self.snum)*500]
        if debug:
            self._gene_test(gn_crop, gn)
            return gn, top, left, gn.copy()
        return gn, top, left, None

    def _getimg(self, pth, top, left, snm):
        pth = pth.replace('gene', 'img').replace('.npz', '.zip')
        im = zarr.load(pth)
        im = im[:,top:top+self.sdim,
                left:left+self.sdim]
        if snm is not None:
            im = rearrange(im, '(s z) h w -> s z h w', s=2)
            if self.stain == 'DAPI':
                im = im[[0]]
            elif self.stain == 'PolyT':
                im = im[[1]]
            shf = self.snum // 4 if self.snum > 1 else 0
            if self.snum > 1:
                pd = np.zeros((im.shape[0], self.spad, 
                               self.sdim , self.sdim))
                im = np.concatenate((pd, im, pd), 1)
            im = im[:, snm+shf:snm+self.snum-shf]
            im = rearrange(im, 's z h w -> (s z) h w')
        return torch.from_numpy(im).float()

    def _to_sparse(self, gn, pad=0):
        # naive init sparse tensor, incompatible to spconv
        dat = torch.FloatTensor(np.array(gn.data).astype(float))
        crd = torch.LongTensor(np.array(gn.coords))
        bat = torch.LongTensor(np.zeros(len(gn.data)))
        h, w, c = gn.shape
        if pad > 0:
            crd[:2] += pad
            h = h + pad * 2
            w = w + pad * 2
        # gn = torch.sparse.FloatTensor(i, v, s)
        return dat, crd, bat, (h, w, c)

    def __getitem__(self, idx):
        snm = self.snum
        if self.snum is not None:
            snm = random.randint(0, self.gmax+2*self.spad-snm)
        gn, top, left, gn_dbg = self._getgene(self.gn_pth[idx],
                                              snm, self.debug)
        im = self._getimg(self.gn_pth[idx], top, left, snm)
        if self.transform:
            im, gn = self._trans(im, gn, gn_dbg, self.debug)

        img = im / 127.5 - 1
        is_pad = (not self.debug) and self.methd in ('ours', 'patch-dm')
        pad = self.pdim if is_pad else 0
        dat, crd, bat, ssz = self._to_sparse(gn, pad)

        if self.debug:
            gnd = gn.todense().astype(float)
            return img, dat, crd, bat, ssz, torch.from_numpy(gnd)
        else:
            return img, dat, crd, bat, ssz

    def _trans(self, im, gn, gn_dbg, debug, p=0.5):
        # gn: [h, w, gdim], im: [chn, h, w]
        rot = random.randint(0, 3)
        if rot > 0:
            for _ in range(rot):
                gn = gn.transpose((1, 0, 2))
                # reverse the H axis
                gn.coords[0] = gn.shape[0] - 1 - gn.coords[0]
            im = torch.rot90(im, rot, [1, 2])
        flp = torch.rand(1) < p
        if flp:
            gn.coords[1] = gn.shape[1] - 1 - gn.coords[1]
            im = F.hflip(im)

        if debug:
            self._trans_test_sp(gn_dbg, gn, rot, flp)
        return im, gn

    def _trans_test_sp(self, gn, gn_trans, rot, flp):
        if rot == 0 and not flp:
            return

        _gn = gn.transpose((2, 0, 1))
        _gn = torch.from_numpy(_gn.todense().astype(float))
        if rot > 0:
            _gn = torch.rot90(_gn, rot, [1, 2])
        if flp:
            _gn = F.hflip(_gn)
        _gn = _gn.permute((1, 2, 0))
        _gn_ts = torch.from_numpy(gn_trans.todense().astype(float))
        assert (_gn == _gn_ts).all()
        return

    def _gene_test(self, gene_raw, gene):
        # Here rna is the spatial resolved raw mrna count data
        # gene is the processed n x n x plex expression table
        _sz, _bk = self.sdim // self.gblk, self.gblk
        for i in range(_sz):
            for j in range(_sz):
                row = slice(i * _bk,
                            (i + 1) * _bk)
                col = slice(j * _bk,
                            (j + 1) * _bk)
                gn = gene_raw[row, col, :].sum((0, 1)).todense()
                assert (gn == gene[i, j].todense()).all()
        # print('gene debug done')

    def __len__(self):
        return len(self.gn_pth)


def prep_pkl(root, pth):
    pkl = pth.stem + '.pickle'
    with open(str(root / pkl), 'wb') as f:
        coo = sparse.load_npz(pth)
        print('max value', coo.max())
        pickle.dump(coo.astype(np.uint16), f, protocol=5)


def sparse_batch_collate(batch):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    gn = None
    if len(batch[0]) == 5:
        im, dat, crd, bat, sz = zip(*batch)
    elif len(batch[0]) == 6:
        im, dat, crd, bat, sz, gn = zip(*batch)
    im = torch.stack(im)

    dat = torch.cat(dat)
    crd = torch.cat(crd, -1)
    bat = torch.cat([b + i for i, b in enumerate(bat)])
    crd = torch.cat((bat[None], crd))
    ssz = torch.Size([len(im), *sz[0]])
    lab = torch.zeros(len(im), dtype=torch.long)

    if gn is None:
        return im, dat, crd, ssz, lab
    else:
        gn = torch.stack(gn)
        return im, dat, crd, ssz, lab, gn


if __name__ == '__main__':
    import argparse
    import multiprocessing
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='StyleGAN2 data class')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--debug',
                        action='store_true')
    args = parser.parse_args()

    data = MBADataset(args.mouse,
                      500, 16, 256, 0, 4 if not args.debug else 1,
                      stain='all', transform=True,
                      debug=args.debug, methd='ours')
    dload = DataLoader(data, batch_size=args.batch, shuffle=True,
                       collate_fn=sparse_batch_collate,
                       **{'drop_last': True,
                          'num_workers': 2,
                          'pin_memory': False,
                          'prefetch_factor': 2})

    for i, bat in enumerate(dload):
        if args.debug:
            im, dat, crd, ssz, idx, gn = bat
        else:
            im, dat, crd, ssz, idx = bat
        print(i, im.shape, ssz, idx.shape)

        if args.debug:
            gn_inp = torch.sparse.FloatTensor(crd, dat, ssz)
            assert (gn == gn_inp.to_dense()).all()

        if i % 99 == 0:
            print(i, im.shape, dat.shape, crd.shape, ssz)

    # with multiprocessing.Pool(processes=8) as pool:
    #     root_npz = Path(f'Data/MERFISH_3D/gene_{args.mouse}')
    #     gn_pth = list(root_npz.glob('*.npz'))

    #     root_pkl = Path(f'Data/MERFISH_3D/gene_{args.mouse}_pkl')
    #     root_pkl.mkdir(parents=True, exist_ok=True)
    #     prep_args = list()
    #     for g in gn_pth:
    #         prep_args.append((root_pkl, g))
    #     pool.starmap(prep_pkl, prep_args)
