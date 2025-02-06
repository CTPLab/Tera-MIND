import torch
import sparse
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from einops import reduce, rearrange, repeat

from utils import MROI

# https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
# https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, dat, bat):
        # Here, we assume dat shape (n, g)
        # where g is the amount of genes
        self.n += bat.sum(0)
        self.delta = (dat - self.mean) * bat
        # Here, we avoid batch 0
        self.n_0 = self.n.clone()
        self.n_0[self.n == 0] = 1
        self.mean += (self.delta / self.n_0).sum(0)
        self.M2 += (self.delta * (dat - self.mean)).sum(0)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return self.variance.sqrt()


def gen_heatmap(pth, df, is_full, id, 
                vmin=0, vmax=10):
    r"""
    Output the heatmap plot for gene expressions

    Args:
        pth: Path to the save folder
        df: The dataframe for creating the heatmap
        is_full: Output the full list of gene expression if True else
                 partial list of high expressed genes
        id: The id of the gene list, 
            which is meant for the sperated full gene list
    """

    font = {'weight': 'bold',
            'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(5, 25 if is_full else 35))
    ht = sns.heatmap(df,
                     yticklabels=True, xticklabels=True,
                     vmin=vmin, vmax=vmax)
    ht.set_yticklabels(ht.get_yticklabels())
    for lab in ht.get_yticklabels():
        # Here it sets all italic.
        lab.set_style('italic')
        lab.set_size(8 if is_full else 16)
        lab.set_weight('bold')
    plt.savefig(str(pth / f'heat_{id}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def gn_sublst(gdir, 
              hst=256, wst=256,
              hnm=16, wnm=16,
              size=256):
    pad = size // 2
    gn_lst = []
    # Here, we assume no boundary issue
    for pw in range(wnm):
        wsz = pw * size
        for ph in range(hnm):
            hsz = ph * size
            # Here, we ignore outer most row and col
            # of MBE to avoid corner cases 
            pnm = [hst + hsz, hst + size + hsz, 
                   wst + wsz, wst + size + wsz,
                   hst - pad + hsz, hst + size + pad + hsz,
                   wst - pad + wsz, wst + size + pad + wsz]
            pnm = [str(p) for p in pnm]
            pnm = '_'.join(pnm)
            pth = gdir / f'{pnm}.npz'
            assert pth.is_file()
            gn_lst.append(pth)
    return gn_lst


def gn_proc(gpth, spth, mask, slc, st=128, size=256, gblk=16,
            debug=False):
    slc_gn = slice(slc[0]*500, (slc[-1]+1)*500)
    gn_lst = [] 
    OV_sum, OV_slc = OnlineVariance(), OnlineVariance()
    for pid, pth in enumerate(gpth):
        if pid % 500 == 0:
            print(pid)
        h, _, w, _ = pth.stem.split('_')[:4]
        h, w = int(h)//32, int(w)//32
        msk = mask[h:h+4,w:w+4,slc] != 0
        # if pid == 0 and np.any(msk.sum((0, 1)) == 0):
        #     raise ValueError('the first batch cannot have 0 batch, choose dif seed!')
        msk_sum = repeat(msk, 'h w c -> (h w c) rep', rep=500)
        msk_slc = repeat(msk, 'h w c -> (h w) (c rep)', rep=500)
        
        # if pid == 15000:
        #     break
        gene = sparse.load_npz(str(pth))
        gn_crop = gene[st:st+size,st:st+size]
        gn = gn_crop[:, None, :, None]
        gn = gn.reshape((size//gblk, gblk,
                         size//gblk, gblk, -1))
        gn = gn.sum((1, 3))
        gn = gn[:, :, slc_gn].todense().astype('float')
        gn = np.log2(gn + 1)
        gn_sum = reduce(gn, 
                        '(n_h h) (n_w w) (z g) -> (n_h n_w z) g', 
                        'sum', n_h=4, n_w=4, g=500)
        gn_slc = reduce(gn, 
                        '(n_h h) (n_w w) c -> (n_h n_w) c', 
                        'sum', n_h=4, n_w=4)
        OV_sum.include(torch.from_numpy(gn_sum).float(),
                       torch.from_numpy(msk_sum))
        OV_slc.include(torch.from_numpy(gn_slc).float(),
                       torch.from_numpy(msk_slc))
        if debug:
            # gn_sum0 = rearrange(gn, 
            #                     '(n_h h) (n_w w) (z g) -> (n_h n_w z) (h w) g',
            #                     n_h=4, n_w=4, g=500)
            # gn_sum0 = gn_sum0.sum(1)
            gn_slc0 = rearrange(gn, 
                                '(n_h h) (n_w w) c -> (n_h n_w) (h w) c',
                                n_h=4, n_w=4)
            gn_slc0 = gn_slc0.sum(1)
            # assert (gn_sum0 == gn_sum).all()
            assert (gn_slc0 == gn_slc).all()
            gn_lst.append(gn_slc)
    if debug:
        gn_np = np.concatenate(gn_lst, 0)
        gn_ts = torch.from_numpy(gn_np)
        avg_err = (gn_ts.mean(0) - OV_slc.mean).abs().max()
        std_err = (gn_ts.std(0) - OV_slc.std).abs().max()
        print(gn_ts.shape, avg_err, std_err)
    
    torch.save({'mean': torch.cat((OV_sum.mean, OV_slc.mean)), 
                'std': torch.cat((OV_sum.std, OV_slc.std))}, 
               str(spth/'all.pt'))
    if '638850' in str(gpth[0]):
        return OV_sum.mean[:229], OV_sum.std[:229]
    else:
        return OV_sum.mean, OV_sum.std


def proc_gene(mouse, task, save_pth, 
              is_roi=False, debug=False):
    r"""
    Calc eigval, eigvec of the scm and draw the heatmap/boxplot of gene expressions 

    Args:
        task: Name of the task, e.g., gene-cell for calc the heatmap
            of high-expressed genes or all the genes
        gene_num: The amount of genes
        save_pth: Path to the output file 
    """

    gene_all = pd.read_csv(f'utils/{mouse[:-1]}_gnm.csv')['gene'].to_list()
    if mouse == '638850':
        gene_all = gene_all[:229]
    print(len(gene_all))
    gn_dir = Path(f'Data/MERFISH_50/gene_{mouse}')
    gn_dct = {}
    mask_lst = list(Path(f'Data/MERFISH_50/mask_{mouse}_large').glob('*.png'))
    mask_lst.sort(key=lambda m:int(m.stem.split('_')[0]))
    mask_npy = [np.array(Image.open(m)) for m in mask_lst]
    mask = np.stack(mask_npy, -1)
    print(mask_lst, 'mask done')
    if is_roi:
        og_dct = {0: 'Isocortex (left)', 
                  1: 'Isocortex (right)',
                  2: 'Hippocampal formation (left)', 
                  3: 'Hippocampal formation (right)'}
        gn_num = len(gene_all) if task == 'all' else 16
        slst, size, pos = MROI[mouse][:3]
        for pid, crd in enumerate(pos):
            gn_pth = gn_sublst(
                gn_dir, 
                crd[0]*32, crd[1]*32,
                size//8, size//8)
            gn_dct[pid] = gn_proc(gn_pth, save_pth, mask, slst, debug=debug)
    else:
        og_dct = {0: 'all'}
        gn_num = len(gene_all) if task == 'all' else 16
        gn_pth = gn_sublst(
            gn_dir, 256, 256, 286, 414
        )
        gn_dct[0] = gn_proc(gn_pth, save_pth, mask, list(range(50)), debug=False)

    if task == 'all':
        if mouse != '638850':
            splt_lst = ((0, 125), (125, 250), (250, 375), (375, 500))
        else:
            splt_lst = ((0, 76), (76, 152), (152, 229))
        gene_lst = [[gene_all[i] for i in range(*sl)] for sl in splt_lst]
    else:
        gene_lst = []
        for s, og in og_dct.items():
            sb_mean, sb_std = gn_dct[s]
            assert (sb_mean >= 0).all()
            gn_idx = torch.argsort(sb_mean, descending=True)
            gn_lst = [gene_all[i] for i in gn_idx[:gn_num]
                      if gene_all[i] not in gene_lst]
            gene_lst += gn_lst
            print(s, og, len(gn_lst))
        gene_lst = [gene_lst,]
    print(gene_lst[0], len(gene_lst[0]))

    for gn_id, gn_lst in enumerate(gene_lst):
        df = pd.DataFrame(gn_lst, columns=['Gene expression',])
        for s, og in og_dct.items():
            sb_mean, sb_std = gn_dct[s]
            sb_mean = sb_mean.numpy()
            df[og] = [sb_mean[gene_all.index(g)] for g in gn_lst]
        df = df.set_index('Gene expression')
        print(df.head(10))
        gen_heatmap(save_pth, df, task=='all', gn_id,
                    vmax=12 if is_roi else 2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--gene_lst', type=str,
                        default='top',
                        choices=['top', 'all'],
                        help='Gene list')
    parser.add_argument('--is_roi', action='store_true',
                        help='if is roi, create the gene heatmap for predefined roi')
    parser.add_argument('--debug', action='store_true',
                        help='Check if online std/mean is equal to naive ones')
    args = parser.parse_args()
    out_dir = Path(f'MBA/heat_msk/{args.mouse}_{args.gene_lst}_{args.is_roi}')
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_gene(args.mouse, args.gene_lst,
              out_dir, args.is_roi, args.debug)