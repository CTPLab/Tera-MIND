
import os
import zarr
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils import MOUSE
from config import TrainConfig
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


def gen_img(gdir, rdir, chn, stain,
            hst=20480, wst=40960,
            hnm=286, wnm=286, 
            size=256, patch_size=64):
    if hnm > 32 or wnm > 32:
        return

    gen_out, rel_out, rel_pad = [], [], size // 2
    shf = rel_pad
    # Here, we assume no boundary issue
    for ph in range(hnm):
        hsz = ph * size
        gen_col, rel_col = [], []
        for pw in range(wnm):
            wsz = pw * size
            g_pos = [hst + hsz, hst + size + hsz,
                     wst + wsz, wst + size + wsz]
            g_nam = '_'.join([str(g) for g in g_pos])
            gen_col.append(zarr.load(gdir / f'{g_nam}.zip'))
            if rdir is not None:
                r_pos = g_pos + [hst + hsz - rel_pad,
                                 hst + size + hsz + rel_pad,
                                 wst + wsz - rel_pad,
                                 wst + size + wsz + rel_pad]
                r_nam = '_'.join([str(r) for r in r_pos])
                rel = zarr.load(rdir / f'{r_nam}.zip')
                rel_col.append(rel[chn, shf:shf + size, shf:shf + size])
        gen_out.append(np.concatenate(gen_col, 2))
        if rdir is not None:
            rel_out.append(np.concatenate(rel_col, 2))
    gen_out = (np.concatenate(gen_out, 1) + 1) * 127.5
    gen_out = np.clip(gen_out, 0, 255).astype(np.uint8)
    if rdir is not None:
        rel_out = np.concatenate(rel_out, 1).astype(np.uint8)
    odir, onm = gdir.parent, gdir.stem
    print('outputed image', gen_out.shape, odir / f'{onm}_.jpg')

    chn_len = len(chn) if stain != 'all' else len(chn) // 2
    for c in range(chn_len):
        if stain != 'all':
            Image.fromarray(gen_out[c]).save(odir / f'{c}_{stain}_gen.jpg')
            if rdir is not None:
                Image.fromarray(rel_out[c]).save(odir / f'{c}_{stain}_rel.jpg')
        else:
            Image.fromarray(gen_out[c]).save(odir / f'{c}_DAPI_gen.jpg')
            Image.fromarray(gen_out[c+50]).save(odir / f'{c}_PolyT_gen.jpg')
            if rdir is not None:
                Image.fromarray(rel_out[c]).save(odir / f'{c}_DAPI_rel.jpg')
                Image.fromarray(rel_out[c+50]).save(odir / f'{c}_PolyT_rel.jpg')


class Tester:
    def __init__(
        self,
        conf: TrainConfig,
        args,
        test_data,
        gpu_id: int,
        epochs: int,
        epolst
    ) -> None:
        self.conf = conf
        self.out_dir = args.out_dir
        self.test_data = test_data
        self.gpu_id = gpu_id
        model = conf.make_model_conf().make_model()

        if args.ckpt_pth is not None:
            state = torch.load(args.ckpt_pth, map_location='cpu', weights_only=False)
            state_dct = {}
            for key, val in state['state_dict'].items():
                if 'ema_model' not in key:
                    state_dct[key.replace('model.', '')] = val
            model.load_state_dict(state_dct, strict=True)
            print('checkpoint is loaded.')
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.epochs = epochs

        self.conf.beatgans_gen_type = GenerativeType.ddim
        self.sampler = self.conf._make_diffusion_conf(T=epochs).make_sampler()
        betas = np.array(self.sampler._betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        print('betas shape', self.betas.shape, self.betas)
        msid = Path(conf.data_path).name.split('_')[-1]
        assert msid in MOUSE
        self.gmax = MOUSE[msid][0]
        self.rna_slc = [i for i in range(self.gmax + 1)]
        self.total_rna, self.total_slc = 500, len(self.rna_slc)
        self.rna_idx = -np.ones([self.gmax + 1])
        self.rna_idx[self.rna_slc] = list(range(len(self.rna_slc)))
        self.rna_slc = torch.Tensor(self.rna_slc).long()
        self.rna_idx = torch.Tensor(self.rna_idx).long()
        self.patch_test = args.patch_test
        self.epolst = epolst or list(range(epochs))
        self.args = args
        self.z_size = len(conf.rna_tpl)
        self.n_stn = 2 if conf.stain == 'all' else 1

    def _run_batch(self, batch, epoch, cur_pth):
        out, roi, dat, crd, ssz, stp = batch
        # Make sure that denoised image loaded from
        # the correct timestamp
        assert (stp == epoch).all()
        out = out.to(self.gpu_id)
        rna = torch.sparse_coo_tensor(crd.long(), dat, ssz)
        rna = rna.to_dense().to(self.gpu_id)
        pad = self.conf.patch_size // 2
        if self.z_size == 1:
            out = rearrange(out, 'b h w (s z) -> (z b) h w s', 
                            z=self.total_slc)
            rna = rearrange(rna, 'b h w (z g) -> (z b) h w g',
                            g=500)
        elif self.z_size in (4, 8, 16):
            out = rearrange(out, 'b h w (s z) -> b h w s z', 
                            z=self.total_slc)
            out = rearrange(out, 'b h w s (n_z z) -> (n_z b) h w (s z)', 
                            z=self.z_size//2)
            rna = rearrange(rna, 'b h w (z g) -> b h w z g',
                            g=500)
            
            rna = rna.unfold(3, self.z_size, self.z_size//2)
            rna = rearrange(rna, 'b h w n_s g s -> (n_s b) h w (s g)')
        shp = out.shape
        # b h w c (without padding)
        shp = (shp[0], shp[3],
               shp[1] - self.conf.patch_size,
               shp[2] - self.conf.patch_size)
        # print(out.shape, ssz, shp)
         
        out = rearrange(out,  'b (p1 h) (p2 w) c-> (b p1 p2) c h w',
                        h=self.conf.patch_size, w=self.conf.patch_size)
        rna = rearrange(rna,  'b (p1 h) (p2 w) c-> (b p1 p2) h w c',
                        h=self.conf.gn_sz, w=self.conf.gn_sz)
        out = self.sampler.sample(model=self.model,
                                  shape=shp,
                                  imgs=out,
                                  noise=out,
                                  r_start=rna,
                                  patch_size=self.conf.patch_size,
                                  # Need to reverse the timestamp
                                  idx=self.epochs - epoch - 1,
                                  model_kwargs=None)
        
        if self.z_size in (1, 4, 8, 16):
            out = rearrange(out, '(n_z b) (s z) h w -> b (s n_z z) h w', 
                            b=roi.shape[0], s=self.n_stn)
        out = out.half().cpu().numpy()
        roi = roi.numpy().tolist()
        for rid, pos in enumerate(roi):
            zarr.save_array(cur_pth / f'{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}.zip',
                            out[rid])

    def _run_epoch(self, epoch, cur_pth):
        for batch in self.test_data:
            self._run_batch(batch, epoch, cur_pth)

    def test(self):
        for epoch in self.epolst:
            print((f'[GPU{self.gpu_id}_{get_rank()}_{get_world_size()}] | '
                   f'Epoch {epoch} | '
                   f'Batchsize: {self.conf.batch_size} | '
                   f'Steps: {len(self.test_data)}'))

            if self.gpu_id > 0:
                barrier()
            if epoch > 1:
                prv_pth = Path(str(self.out_dir) + f'_{epoch - 1}')
                if prv_pth.is_dir():
                    shutil.rmtree(prv_pth)
            self.test_data.sampler.set_epoch(epoch)
            self.test_data.dataset.set_step(epoch)
            cur_pth = Path(str(self.out_dir) + f'_{epoch + 1}')
            cur_pth.mkdir(parents=True, exist_ok=True)
            if self.gpu_id == 0:
                barrier()

            with torch.inference_mode():
                self._run_epoch(epoch, cur_pth)
            # make sure all gpus complete the epoch
            barrier()

            if epoch == self.epochs - 1 and self.gpu_id == 0:
                img_lst = self.rna_slc.numpy().tolist()
                if self.z_size in (8, 16):
                    img_lst = img_lst[1:-1] 
                if self.conf.stain == 'all':
                    img_lst = img_lst + [i + self.gmax + 1 for i in img_lst]
                gen_img(cur_pth,
                        None if not self.args.has_img else Path(f'Data/MERFISH_50/img_{self.args.mouse}'),
                        img_lst,
                        self.conf.stain,
                        hst=self.args.hst, wst=self.args.wst,
                        hnm=self.args.hnm, wnm=self.args.wnm)

        if self.gpu_id == 0:
            prv_pth = Path(str(self.out_dir) + f'_{self.epochs - 1}')
            if prv_pth.is_dir():
                shutil.rmtree(prv_pth)


def main(rank: int, world_size: int, total_epochs: int, conf, args):
    ddp_setup(rank, world_size, args.port)
    _chn = 48 if len(conf.rna_tpl) in (8, 16) else 50
    if conf.stain == 'all':
        _chn *= 2
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
    tester = Tester(conf, args, test_data, rank, total_epochs, epolst)
    tester.test()
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='simple distributed training job')
    parser.add_argument('--data_path', '-d', type=str, default="./dataset",
                        help='dataset path')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
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
    parser.add_argument('--has_img', action='store_true',
                        help='Whether has the gt training images')
    args = parser.parse_args()

    cfg_parm = args.ckpt_pth.parent.name
    trn_mouse, size, nrna, stain, srna = cfg_parm.split('_')[:-1]
    world_size = torch.cuda.device_count()
    conf = prep_config_parm(args.data_path, args.batch_size, 
                            int(size), world_size, 
                            stain, args.mouse,
                            int(nrna), int(srna),
                            is_test=True)
    print(conf)

    T = args.tot_epoch
    assert args.cur_epoch is None or args.cur_epoch < T
    mp.spawn(main,
             args=(world_size, T, conf, args),
             nprocs=world_size)
