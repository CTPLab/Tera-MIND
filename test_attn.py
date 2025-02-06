import os
import zarr
import torch
import pyvips
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path
from matplotlib import cm
from einops import rearrange
from torch.amp import autocast
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from config import TrainConfig
from utils import MROI, MALL, CM
from config_parm import prep_config_parm
from utils.choices import GenerativeType
from utils.dist_utils import get_rank, get_world_size, barrier
from utils.MBADataset_tst import MBADataset_tst, sparse_batch_collate_tst


def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prep_dloader(dataset: Dataset, batch_size: int, num_workers: int, collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2)


def gn_sublst(gdir, size=256,
              hst=256, wst=256,
              hnm=286, wnm=414):
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
            gn_lst.append(gdir / f'{pnm}.npz')
    return gn_lst


def draw_heatmap(arr, pth, w, h, dpi, cmap,
                 vmin=0, vmax=1, cbar=False):
    if pth.is_file():
        return
    # Make sure the heatmap fit the entire plot
    fig = plt.figure(figsize=(w/dpi, h/dpi), 
                     dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    sns.heatmap(arr, vmin=vmin, vmax=vmax, 
                cbar=cbar, cmap=cmap)
    plt.savefig(pth, bbox_inches='tight', 
                pad_inches=0, dpi=dpi)
    plt.close(fig)


def draw_attplot(df, pth, w, h, dpi, colors, gn_lst, fontsize=24):
    if pth.is_file():
        return
    # Make sure the heatmap fit the entire plot
    font = {'weight': 'bold',
            'size': fontsize}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(h, w), 
                     dpi=dpi*2)
    x_, y_ = 'Gene expr. level', 'Attention level'
    jp = sns.JointGrid(data=df, x=x_, y=y_, 
                       hue='Gene', palette=colors, 
                       space=0, ratio=3)
    jp.ax_joint.set_yticks([0, 2, 4, 6, 8])
    jp.ax_joint.set_ylim(0, 8)
    if 'Slc17a7' in gn_lst:
        jp.ax_joint.set_xticks([0, 2, 4])
        jp.ax_joint.set_xlim(0, 4)
    else:
        jp.ax_joint.set_xticks([0, 1, 2])
        jp.ax_joint.set_xlim(0, 2)

    for gid, gn in enumerate(gn_lst):
        df_grp = df[df['Gene']==gn]
        sns.regplot(data=df_grp, x=x_, y=y_, 
                    scatter=True, line_kws={'color': colors[gid], 'lw':4}, 
                    ax=jp.ax_joint,
                    scatter_kws={'color': colors[gid], 's':2, 'alpha':0.05})
    jp.set_axis_labels(x_, y_, 
                       fontsize=fontsize, fontweight='bold')
    sns.stripplot(data=df, x=x_, ax=jp.ax_marg_x, hue='Gene', dodge=True, legend=False, 
                  palette=colors, s=2, alpha=0.05)
    sns.kdeplot(data=df, y=y_, ax=jp.ax_marg_y, fill=True, legend=False, cut=0, linewidths=4)
    # lp.set_ylabel(lp.get_ylabel(), fontdict={'weight': 'bold'})
    # lp.legend_.set_title(None)
    # plt.title(i)
    # plt.tight_layout()
    plt.savefig(str(pth),bbox_inches='tight')
    plt.close(fig)


def cm_platte(st=(0, 0, 0), ed=(1, 1, 1), step=100):
    # cls = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    cm = LinearSegmentedColormap.from_list(
        "Custom", [st, ed], N=step)
    return cm


def gen_img(gen_dir, rel_dir, att_dir, 
             chn, gn_lst, gn_grp, mse, 
             hst=20480, wst=40960,
             hnm=286, wnm=286, 
             size=256, dpi=200, alpha=200):
    gen = zarr.load(gen_dir / f'all_{chn}.zip').astype('float')
    h = gen.shape[1]//2
    w = gen.shape[2]//2
    gen_s = F.resize(torch.FloatTensor(gen), (h//2, w//2))
    gen_s = gen_s.clamp(0).numpy()
    gen = F.resize(torch.FloatTensor(gen), (h, w))
    gen = gen.clamp(0).numpy()
    
    att_dir = att_dir / f'{gn_grp}'
    att_dir.mkdir(parents=True, exist_ok=True)
    # Prepare DAPI image for cmp
    rel, rel_s, rel_pth = None, None, att_dir / f'real/{chn}.jpg' 
    if rel_dir is not None:
        # Get the DAPI
        real = Image.open(str(rel_dir / f'{chn}.jpg'))
        rel = np.array(real.resize((w, h)))
        rel_s = np.array(real.resize((w//2, h//2)))
        if not rel_pth.is_file(): 
            rel_pth.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rel).save(str(rel_pth))

    # Draw color bar
    cm_lst = []
    if gn_grp == 'GLUT':
        vextr = {'att_updn':(2, 4), 'att_all':2, 'expr':(1, 3)}
    elif gn_grp in ('DOPA', 'BLOD'):    
        vextr = {'att_updn':(2, 4), 'att_all':2, 'expr':(1, 1)}
    for c in range(len(gn_lst) + 1):
        vmax = vextr['expr'][c] if c < len(gn_lst) else vextr['att_all']
        cm_ = cm_platte(ed=CM[gn_grp][c]) if c < len(gn_lst) else cm.inferno
        cm_lst.append(cm_)
        cbar_pth = att_dir / f'{gn_grp}_{c}.jpg'
        draw_heatmap(np.zeros_like(gen[0]), cbar_pth,
                     w, h, dpi, cm_lst[-1], 
                     vmax=vmax, cbar=True)
    draw_heatmap(np.zeros_like(gen[0]), att_dir / f'{gn_grp}_updn.jpg',
                 w, h, dpi, cm.coolwarm, 
                 vmax=vextr['att_updn'][1]-vextr['att_updn'][0], cbar=True)
   
    for oi in (0, 2, 3):
        wei = 229 if mse == '638850' else 500
        sfix = {0:'att_updn', 2:'att_all', 3:'expr'}[oi]
        gen_sum = gen[3*len(gn_lst):].sum(0)
        gen_msk, gen_msk_s = True, True
        for gi in range(len(gn_lst)):
            gen_msk = gen_msk & (gen[3*len(gn_lst)+gi] != 0)
            gen_msk_s = gen_msk_s & (gen_s[3*len(gn_lst)+gi] != 0)  

        out_dir = att_dir / f'{sfix}'
        out_dir.mkdir(parents=True, exist_ok=True)
        reg_dir = att_dir / 'reg'
        reg_dir.mkdir(parents=True, exist_ok=True)

        if oi == 2:
            gen_out = gen[2*len(gn_lst):3*len(gn_lst)].sum(0)
            gfld = np.log2(gen_out*wei+1)
            gfld = gaussian_filter(gfld * gen_msk, sigma=2)
            draw_heatmap(gfld, out_dir / f'{chn}.jpg',
                         w, h, dpi, cm.inferno, 
                         vmax=vextr[sfix], cbar=False)
            
            gen_out = gen_s[2*len(gn_lst):3*len(gn_lst)].sum(0)
            at0 = np.log2(gen_out[gen_msk_s][:]*wei+1)
            at_all = np.concatenate((at0, at0))
            gn0 = gen_s[3*len(gn_lst)]
            gn0 = np.log2(gn0[gen_msk_s][:]+1)
            gn1 = gen_s[3*len(gn_lst)+1]
            gn1 = np.log2(gn1[gen_msk_s][:]+1)
            gn_all = np.concatenate((gn0, gn1))
            gn_nm = [gn_lst[0]]*len(gn0) + [gn_lst[1]]*len(gn1)
            df = pd.DataFrame({'Gene': gn_nm, 'Gene expr. level': gn_all, 'Attention level': at_all})
            assert np.all(gn_all >=0) and np.all(at_all >= 0)
            draw_attplot(df, reg_dir / f'{chn}.png',
                         w, h, dpi, CM[gn_grp][:2], gn_lst)
            
        elif oi == 3:
            for gi, gn in enumerate(gn_lst):
                (out_dir / gn).mkdir(parents=True, exist_ok=True)
                gen_out = np.log2(gen[3*len(gn_lst)+gi]+1)
                gen_out = gaussian_filter(gen_out, sigma=2)
                draw_heatmap(gen_out, out_dir / f'{gn}/{chn}_{gn}.jpg',
                             w, h, dpi, cm_platte(ed=CM[gn_grp][gi]), 
                             vmax=vextr[sfix][gi], cbar=False)
                
            gen_lst = []
            for gi, gn in enumerate(gn_lst):
                gen_out = Image.open(str(out_dir / f'{gn}/{chn}_{gn}.jpg'))
                gen_lst.append(np.array(gen_out.convert('RGB')))
            (out_dir / 'all').mkdir(parents=True, exist_ok=True)
            out_pth = str(out_dir / f'all/{chn}.jpg')
            msk = pyvips.Image.new_from_array(gen_lst[-1])
            msk = msk.bandjoin(alpha)
            img = pyvips.Image.new_from_array(gen_lst[0])
            img.composite(msk, 'over').write_to_file(str(out_pth))
            # workaround
            img = np.array(Image.open(out_pth))
            coef = gen_lst[-1].sum(-1) != 0
            coef = gaussian_filter(coef.astype('float'), sigma=2).clip(0, 1)
            coef = coef[:,:,None]
            img = coef*img +(1-coef)*gen_lst[0]
            Image.fromarray(img.astype('uint8')).save(out_pth)

        elif oi == 0:
            gen0 = gen_s[:len(gn_lst)].sum(0)
            gfld0 = np.log2(gen0*wei+1)
            gfld0 = gaussian_filter(gfld0 * gen_msk_s, sigma=2)
            gen1 = gen_s[len(gn_lst):2*len(gn_lst)].sum(0)
            gfld1 = np.log2(gen1*wei+1)
            gfld1 = gaussian_filter(gfld1 * gen_msk_s, sigma=2)

            gout0 = np.rot90(gfld0.T, axes=(0,1))
            gout1 = np.rot90(gfld1.T, axes=(0,1))
            y, x = gout0.shape
            y, x = range(y), range(x)
            Y, X = np.meshgrid(y, x)
            Y, X = Y.T, X.T
            # Make sure the heatmap fit the entire plot
            fig = plt.figure(figsize=(w/dpi, h/dpi), 
                             dpi=dpi, frameon=False)
            ax = fig.add_subplot(projection='3d')
            ax.set_axis_off()
            fig.add_axes(ax)

            vmin, vmax = vextr['att_updn']
            ax.plot_surface(Y, X, gout0+vmin,
                            rstride=1, cstride=1,  
                            cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
            
            
            ax.plot_surface(Y, X, -gout1-vmin,
                            rstride=1, cstride=1,  
                            cmap=cm.coolwarm.reversed(), vmin=-vmax, vmax=-vmin)
            ## elev = 90, azim = -90
            # ax.view_init(90, -90)
            plt.savefig(out_dir / f'{chn}.png', 
                        bbox_inches='tight', transparent=True,
                        pad_inches=0, dpi=dpi)
            plt.close(fig)

            if rel_s is not None:
                bkg_out = Image.fromarray(rel_s)
                bkg_out = np.asarray(bkg_out).astype('float') / 255.
                bkg_out = np.rot90(bkg_out.transpose(1, 0, 2), axes=(0, 1))
                # bkg_out = np.stack((bkg_out, bkg_out, bkg_out), -1)
                print(bkg_out.shape, gout0.shape)
                fig = plt.figure(figsize=(w/dpi, h/dpi), 
                                    dpi=dpi, frameon=False)
                ax = fig.add_subplot(projection='3d')
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.plot_surface(Y, X, np.zeros_like(gout0), 
                                rstride=1, cstride=1,  
                                facecolors=bkg_out)
                plt.savefig(out_dir / f'{chn}_real.png', 
                            bbox_inches='tight', transparent=True,
                            pad_inches=0, dpi=dpi)
                plt.close(fig)

    print(f'{chn}-th slice is done')


class Tester:
    def __init__(
        self,
        conf: TrainConfig,
        args,
        test_data,
        gpu_id: int,
        epochs: int,
        epolst,
        slst, glst
    ) -> None:
        self.conf = conf
        self.out_dir = args.out_dir
        self.test_data = test_data
        self.gpu_id = gpu_id
        model = conf.make_model_conf().make_model()
        # print(model)
        if args.ckpt_pth is not None:
            state = torch.load(args.ckpt_pth, map_location='cpu', weights_only=False)
            state_dct = {}
            for key, val in state['state_dict'].items():
                if 'ema_model' not in key:
                    state_dct[key.replace('model.', '')] = val
            model.load_state_dict(state_dct, strict=False)
            print('checkpoint is loaded.')
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.epochs = epochs

        # self.trn_sampler = conf.make_diffusion_conf().make_sampler()
        # eval_conf = copy.deepcopy(conf)
        self.conf.beatgans_gen_type = GenerativeType.ddim
        self.sampler = self.conf._make_diffusion_conf(T=epochs).make_sampler()
        betas = np.array(self.sampler._betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        print('betas shape', self.betas.shape, self.betas)
        self.tot_rna, self.tot_slc = 500, 50
        self.patch_test = args.patch_test
        self.epolst = epolst or list(range(epochs))
        self.args = args
        self.z_size = len(conf.rna_tpl)
        assert self.z_size == 4
        self.n_stn = 2 if conf.stain == 'all' else 1
        self.slst, self.glst = slst, glst


    def _run_batch(self, batch, epoch, cur_pth):
        out, roi, dat, crd, ssz, stp = batch
        # Make sure that denoised image loaded from
        # the correct timestamp
        assert (stp == epoch).all()
        out = out.to(self.gpu_id)
        rna = torch.sparse_coo_tensor(crd.long(), dat, ssz)
        rna = rna.to_dense().to(self.gpu_id)
        if self.z_size == 1:
            out = rearrange(out, 'b h w (s z) -> (z b) h w s', 
                            z=self.tot_slc)
            rna = rearrange(rna, 'b h w (z g) -> (z b) h w g',
                            g=self.tot_rna)
        elif self.z_size in (4, 8, 16):
            out = rearrange(out, 'b h w (s z) -> b h w s z', 
                            z=self.tot_slc)
            out = rearrange(out, 'b h w s (n_z z) -> (n_z b) h w (s z)', 
                            z=self.z_size//2)
            rna = rearrange(rna, 'b h w (z g) -> b h w z g',
                            g=self.tot_rna)

            rna = rna.unfold(3, self.z_size, self.z_size//2)
            rna = rearrange(rna, 'b h w n_s g s -> (n_s b) h w (s g)')
        shp = out.shape
        # b h w c (without padding)
        shp = (shp[0], shp[3] // 2,
               shp[1] - self.conf.patch_size,
               shp[2] - self.conf.patch_size)
        # print(out.shape, ssz, shp)
         
        out = rearrange(out,  'b (p1 h) (p2 w) c-> (b p1 p2) c h w',
                        h=self.conf.patch_size, w=self.conf.patch_size)
        p1 = rna.shape[1] // self.conf.gn_sz
        p2 = rna.shape[2] // self.conf.gn_sz
        rna = rearrange(rna,  'b (p1 h) (p2 w) c-> (b p1 p2) h w c',
                        h=self.conf.gn_sz, w=self.conf.gn_sz)
        # normalization based on checking the readout
        pad = self.conf.gn_sz // 2
        # autocast will cause minor err to symmetric attn
        with autocast('cuda', enabled=self.conf.fp16), torch.no_grad(): 
            attn, rna = self.model.forward(
                x=out, t=None, rna=rna,
                imgs=torch.zeros(shp).to(self.gpu_id),
                patch_size=self.conf.patch_size)
            
            # Get 0 <-> 1, 1 <-> 2
            rna0 = rearrange(rna[:, self.glst, 0], 
                             'b g h w -> b g (h w)')
            att0 = attn[:2, :, self.glst]
            att0 = rearrange(att0[..., self.glst], 'd b c g -> b (d c) g')
            
            # Get 1 <-> 2, 2 <-> 3
            rna1 = rearrange(rna[:, self.glst, 1], 
                             'b g h w -> b g (h w)')
            att1 = attn[1:3, :, self.glst]
            att1 = rearrange(att1[..., self.glst], 'd b c g -> b (d c) g')
            out = torch.cat([att0@rna0, att1@rna1], -1)

            # Get ensemble
            rna2 = rearrange(rna[:, self.glst], 
                            'b g z h w -> b g (z h w)')
            att2 = (attn[3, :, self.glst])[..., self.glst]
            # append the summed rna as baseline
            out = torch.cat([out, att2@rna2, rna2], 1)
            out = rearrange(out, '(n_z b p1 p2) g (z h w) -> b (n_z z) g (p1 h) (p2 w)',
                            b=roi.shape[0], p1=p1, p2=p2, 
                            h=self.conf.gn_sz, w=self.conf.gn_sz)
            out = out[:,:,:,pad:-pad,pad:-pad]
        roi = roi.numpy().tolist()
        out = out.half().cpu().numpy()
        for rid, pos in enumerate(roi):
            zarr.save_array(cur_pth / f'{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}.zip',
                            out[rid])

    def _run_epoch(self, epoch, cur_pth):
        for batch in self.test_data:
            self._run_batch(batch, epoch, cur_pth)

    def test(self):
        for epoch in range(1):
            print((f'[GPU{self.gpu_id}_{get_rank()}_{get_world_size()}] | '
                   f'Epoch {epoch} | '
                   f'Batchsize: {self.conf.batch_size} | '
                   f'Steps: {len(self.test_data)}'))

            if self.gpu_id > 0:
                barrier()
            self.test_data.sampler.set_epoch(epoch)
            self.test_data.dataset.set_step(epoch)
            cur_pth = Path(str(self.out_dir) + f'_{self.args.path}')
            cur_pth.mkdir(parents=True, exist_ok=True)
            if self.gpu_id == 0:
                barrier()

            with torch.inference_mode():
                self._run_epoch(epoch, cur_pth)
            # make sure all gpus complete the epoch
            barrier()


def main(rank: int, world_size: int, total_epochs: int, conf, args):
    ddp_setup(rank, world_size, args.port)
    _chn = 48 if len(conf.rna_tpl) in (8, 16) else 50
    if conf.stain == 'all':
        _chn *= 2
    # gn_pth = list(Path(gdir).glob('*.npz'))
    if args.region != -1:
        slst, size, pos, gnm = MROI[args.mouse]
        gn_lst = pd.read_csv(f'utils/{args.mouse[:-1]}_gnm.csv')['gene'].to_list()
        glst = [gn_lst.index(g) for g in gnm[args.region]]
        args.hst = pos[args.region][0] * 32
        args.wst = pos[args.region][1] * 32
        args.hnm = size // 8
        args.wnm = size // 8
    else:
        slst, gnm = list(range(50)), MALL[args.path]
        gn_lst = pd.read_csv(f'utils/{args.mouse[:-1]}_gnm.csv')['gene'].to_list()
        glst = [gn_lst.index(g) for g in gnm]
        args.hst, args.wst = 256, 256
        args.hnm, args.wnm = 286, 414
    gn_pth = gn_sublst(Path(f'Data/MERFISH_50/gene_{args.mouse}'), 
                       hst=args.hst, wst=args.wst,
                       hnm=args.hnm, wnm=args.wnm)
    _blk = conf.patch_size // conf.gn_sz
    dataset = MBADataset_tst(gn_pth, 500, _blk, len(conf.rna_tpl),
                             pad=conf.patch_size // 2,
                             idir=args.out_dir, stain=conf.stain, chn=_chn,
                             hst=args.hst, wst=args.wst, hnm=args.hnm, wnm=args.wnm)
    test_data = prep_dloader(dataset, conf.batch_size, conf.num_workers,
                             sparse_batch_collate_tst)
    epolst = None if args.cur_epoch is None else \
        list(range(args.cur_epoch, total_epochs))
    tester = Tester(conf, args, test_data, rank, total_epochs, epolst, 
                    slst, glst)
    tester.test()
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    import multiprocessing
    parser = argparse.ArgumentParser(
        description='simple distributed training job')
    parser.add_argument('--data_path', '-d', type=str, default="./dataset",
                        help='dataset path')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--path', type=str,
                        default='GLUT',
                        choices=['GLUT', 'DOPA', 'BLOD'],
                        help='Folder to different mouses')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Input batch size on each device (default: 32)')
    parser.add_argument('--port', default=23498, type=int,
                        help='master port')
    parser.add_argument('--patch_size', '-ps', type=int, default=64,
                        help='model base patch size')
    parser.add_argument('--out_dir', '-g', type=str, default="./output_images",
                        help='dir of intermediate timestamp result')
    parser.add_argument('--ckpt_pth', type=Path,
                        default="checkpoints/exp/last.ckpt",
                        help='generate path')
    parser.add_argument('--patch_test', default=4, type=int,
                        help='The amount of test patches')
    parser.add_argument('--region', default=0, type=int, choices=(-1,0,1,2,3),
                        help='The roi for profiling')
    parser.add_argument('--hst', type=int, default=256,
                        help='The starting location of height')
    parser.add_argument('--wst', type=int, default=256,
                        help='The starting location of width')
    parser.add_argument('--hnm', type=int, default=286,
                        help='The number of rows')
    parser.add_argument('--wnm', type=int, default=414,
                        help='The number of cols')
    parser.add_argument('--tot_epoch', type=int, default=15,
                        help='The total number of epochs')
    parser.add_argument('--cur_epoch', type=int, default=None,
                        help='The current epoch, needed when restarting the infer from the cur epoch.')
    parser.add_argument('--calc_attn', action='store_true',
                        help='calculate attention')
    parser.add_argument('--is_vis', action='store_true',
                        help='Output visualization')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    args = parser.parse_args()

    cfg_parm = args.ckpt_pth.parent.name
    trn_mouse, size, nrna, stain, srna = cfg_parm.split('_')[:-1]
    world_size = torch.cuda.device_count()
    conf = prep_config_parm(args.data_path, args.batch_size, 
                            int(size), world_size, 
                            stain, args.mouse,
                            int(nrna), int(srna),
                            is_test=True)
    conf.method = 'ours_vis'
    print(args.mouse, args.path)

    if args.calc_attn:
        T = args.tot_epoch
        assert args.cur_epoch is None or args.cur_epoch < T
        mp.spawn(main,
                 args=(world_size, T, conf, args),
                 nprocs=world_size)
    
    if args.is_vis:
        if args.region != -1:
            slst, size, pos, gnm = MROI[args.mouse]
            args.hst = pos[args.region][0] * 32
            args.wst = pos[args.region][1] * 32
            args.hnm = size // 8
            args.wnm = size // 8
            gnm = gnm[args.region]
        else:
            slst, gnm = list(range(50)), MALL[args.path]
            gn_lst = pd.read_csv(f'utils/{args.mouse[:-1]}_gnm.csv')['gene'].to_list()
            args.hst, args.wst = 256, 256
            args.hnm, args.wnm = 286, 414
        gen_dir = Path(args.out_dir) / f'{args.mouse[-1]}_vis/attn_{args.path}' 
        rel_dir = Path(args.out_dir) / f'gen_{args.mouse}_rhalf'
        att_dir = Path(args.out_dir) / f'attn_{args.mouse}'
        att_dir.mkdir(parents=True, exist_ok=True)  
        with multiprocessing.Pool(processes=args.core) as pool:
            gargs = []
            for sl in slst:
                gargs.append((gen_dir, None, att_dir,
                              sl, gnm, args.path, args.mouse))
            pool.starmap(gen_img, gargs)