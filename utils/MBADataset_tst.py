import zarr
import torch
import sparse
import numpy as np

from pathlib import Path
from einops import rearrange
from torch.utils.data import Dataset


# https://en.wikipedia.org/wiki/Linear_congruential_generator
# https://stackoverflow.com/questions/8569113/why-1103515245-is-used-in-rand
def lcg(x, a=1103515245, c=12345, m=2**31):
    return (a * x + c) % m

class MBADataset_tst(Dataset):
    def __init__(self, gpth, 
                 gdim, gblk,
                 snum, idir=None, 
                 stain='DAPI', 
                 hst=256, wst=256,
                 hnm=286, wnm=414,
                 hei=36 * 8, wid=52 * 8,
                 chn=100, pad=32,
                 size=256):

        self.gdim = gdim  # gdim-plex of gene
        self.gblk = gblk  # Size of small gene array to be summed
        self.snum = snum
        self.spad = {None:None, 1:0, 4:1, 8:1, 16:3}[snum] 
        self.idir = idir
        self.stain = stain
        self.hst, self.wst = hst//size, wst//size
        self.hed, self.wed = self.hst+hnm, self.wst+wnm
        self.hei, self.wid = hei, wid
        self.chn, self.pad = chn, pad
        self.size = size
        self.gsz = (size + 2 * pad) // gblk
        self.psz = pad // gblk

        self.gn_pth = gpth
        self.set_step(0)
        print(len(self.gn_pth))

    def set_step(self, step:int):
        # Update the timestamp at the beginning of each epoch
        self.step = step

    def _prep_pad(self, pth, row, col, 
                  shape, tstep):
        s_img = lcg(row * self.wid + col)

        # By setting manual seed, we ensure the consistent padding
        # from the fixed neighbor noise tile
        if pth is None:
            assert tstep == 0
            torch.manual_seed(s_img)
            img = torch.randn(shape)
        else:
            assert tstep > 0
            img = torch.from_numpy(zarr.load(pth))
            img = rearrange(img, 'c h w -> h w c')
        return img

    def _getgene(self, pth):
        gene = sparse.load_npz(pth)
        gh, gw = gene.shape[:2]
        gn = gene[:, None, :, None]
        gn = gn.reshape((gh//self.gblk, self.gblk,
                         gw//self.gblk, self.gblk, -1))
        gn = gn.sum((1, 3))
        if self.snum not in (1, None):
            gn.coords[2] += (self.spad * 500)
            shp = list(gn.shape)
            shp[-1] += self.spad * 1000
            gn = sparse.COO(gn.coords, gn.data, 
                            shape=tuple(shp))
        return gn

    def _pad_gn(self, gn, roi, roio):
        dat, crd, ssz = gn.data, gn.coords, gn.shape
        idx = []
        for i in range(2):
            # Shift the coordinates with and without overlap
            crd[i] += (self.psz - (roi[i * 2] - roio[i * 2]) // self.gblk)
            # Crop the array so that the dim becomes (2048 // 16, 2048 // 16)
            idx.append((crd[i] >= 0) & (crd[i] < self.gsz))
        idx = idx[0] & idx[1]
        return dat[idx], crd[:, idx], (self.gsz, self.gsz, ssz[-1])

    def _pad_im(self, roi, timestep):
        # initial image and noise, may speed up the padding efficiency
        # when setting the channel to last dim
        shp = (self.gsz * self.gblk, self.gsz * self.gblk, self.chn)
        out = -torch.ones(shp)

        # Core impl of padding denoised image and noise
        # using neighboring tiles to ensure seamless generation
        _row = roi[0] // self.size
        _col = roi[2] // self.size
        olst = [None, self.pad, -self.pad, None]
        plst = [-self.pad, None, None, self.pad]
        for rid, r in enumerate((-1, 0, 1)):
            for cid, c in enumerate((-1, 0, 1)):
                r_val = self.hst <= _row + r < self.hed
                c_val = self.wst <= _col + c < self.wed
                if r_val and c_val:
                    r_roi = f'{(_row + r) * self.size}_{(_row + r + 1) * self.size}'
                    c_roi = f'{(_col + c) * self.size}_{(_col + c + 1) * self.size}.zip'
                    # Get the (_row + r, _col + c) neighor tile
                    pth_new = None if timestep == 0 else Path(f'{self.idir}_{timestep}') / f'{r_roi}_{c_roi}'
                    # pth_new = None
                    _pad = self._prep_pad(pth_new, _row + r, _col + c,  
                                          (self.size, self.size, self.chn),
                                          timestep)
                    o_slc = (slice(olst[rid], olst[rid + 1]),
                             slice(olst[cid], olst[cid + 1]), slice(None,))
                    p_slc = (slice(plst[rid], plst[rid + 1]),
                             slice(plst[cid], plst[cid + 1]), slice(None,))
                    # Assign padding from the (_row + r, _col + c) neighbor tile
                    out[o_slc] = _pad[p_slc]

        return out, timestep

    def _to_torch(self, dat, crd, ssz):
        dat = torch.FloatTensor(np.array(dat).astype(float))
        crd = torch.LongTensor(np.array(crd))
        bat = torch.LongTensor(np.zeros(len(dat)))
        return dat, crd, bat, ssz

    def __getitem__(self, idx):
        pth = self.gn_pth[idx]
        gn = self._getgene(pth)

        pnm = Path(pth).stem
        pnm = pnm.split('_')
        pnm = np.array([int(_p) for _p in pnm])
        roi, roio = pnm[:4], pnm[4:]

        dat, crd, ssz = self._pad_gn(gn, roi, roio)
        dat, crd, bat, ssz = self._to_torch(dat, crd, ssz)

        out, tstp = self._pad_im(roi, self.step)
        return out, torch.from_numpy(roi), \
            dat, crd, bat, ssz, tstp
    
    def __len__(self):
        return len(self.gn_pth)


def sparse_batch_collate_tst(batch):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    out, roi, dat, crd, bat, sz, step = zip(*batch)
    out = torch.stack(out)
    roi = torch.stack(roi)

    dat = torch.cat(dat)
    crd = torch.cat(crd, -1)
    bat = torch.cat([b + i for i, b in enumerate(bat)])
    crd = torch.cat((bat[None], crd))
    ssz = torch.Size([len(out), *sz[0]])
    
    step = torch.from_numpy(np.asarray(step))

    return out, roi, dat, crd, ssz, step


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='StyleGAN2 data class')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--debug',
                        action='store_true')
    args = parser.parse_args()

    data = MBADataset_tst(f'Data/MERFISH_3D/gene_{args.mouse}',
                          500, 16, chn=120)
    dload = DataLoader(data, batch_size=args.batch, shuffle=False,
                       collate_fn=sparse_batch_collate_tst,
                       **{'drop_last': False,
                          'num_workers': 4,
                          'pin_memory': True,
                          'prefetch_factor':2})
    for e in range(4):
        data.set_step(e)
        for i, bat in enumerate(dload):
            out, roi, dat, crd, ssz, stp = bat
            print(stp)
            assert (stp == e).all()
            # out_pat = rearrange(out, 'b (p1 h) (p2 w) c -> (b p1 p2) c h w',
            #                     h=64, w=64)
            print(i, out.shape, roi.shape, ssz)
            # if i == 20:
            #     break
        print(data.step)