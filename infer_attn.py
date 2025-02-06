import zarr
import argparse
import numpy as np
import multiprocessing as mp

from pathlib import Path


def gen_col(gdir, odir,
            hst=20480, wst=40960, 
            hnm=16, pw=0,
            size=256):
    wsz = pw * size
    gen_col = []
    for h in range(hnm):
        hsz = h * size
        g_pos = [hst+hsz, hst+size+hsz,
                 wst+wsz, wst+size+wsz]
        g_nam = '_'.join([str(g) for g in g_pos])
        g_npy = zarr.load(gdir / f'{g_nam}.zip')
        gen_col.append(g_npy)
    gen_col = np.concatenate(gen_col, -2)

    for sl in range(gen_col.shape[0]):
        (odir / f'{sl}').mkdir(parents=True, exist_ok=True)
        zarr.save_array(str(odir / f'{sl}/{pw}.zip'), gen_col[sl])
    print(f'Col {pw} is done')


def gen_mba(odir, slc, wnm=16):
    if not (odir / f'{slc}').is_dir():
        print(str(odir / f'{slc}'), 'does not exist.')
        return 
    gcol = [zarr.load(str(odir / f'{slc}/{w}.zip'))
            for w in range(wnm)]
    gmba = np.concatenate(gcol, -1)
    zarr.save_array(str(odir / f'all_{slc}.zip'),
                    gmba)
    print(gmba.shape, f'all_{slc}.zip is done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gdir', type=Path,
                        help='Path to generated/gt tiles')
    parser.add_argument('--odir', type=Path,
                        help='Path to output results')
    parser.add_argument('--hst', type=int, default=20480,
                        help='The starting location of height')
    parser.add_argument('--wst', type=int, default=40960,
                        help='The starting location of width')
    parser.add_argument('--hnm', type=int, default=16,
                        help='The number of rows')
    parser.add_argument('--wnm', type=int, default=16,
                        help='The number of cols')
    parser.add_argument('--gen_col', action='store_true',
                        help='generate row of mbe')
    parser.add_argument('--gen_mba', action='store_true',
                        help='generate the whole mbe')
    parser.add_argument('--core', type=int, default=8,
                        help='The number of cols')
    args = parser.parse_args()

    # Here, we assume no boundary issue
    assert 256 <= args.hst and 256 <= args.wst 
    assert 0 < args.hnm <= 286 and 0 < args.wnm <= 414 

    if args.gen_col:
        args.odir.mkdir(parents=True, exist_ok=True)
        with mp.Pool(processes=args.core) as pool:
            prep_args = list()
            for pw in range(args.wnm):
                prep_args.append((
                    args.gdir, args.odir,
                    args.hst, args.wst,
                    args.hnm, pw))
            pool.starmap(gen_col, prep_args)

    if args.gen_mba:
        with mp.Pool(processes=args.core) as pool:
            prep_args = list()
            for sl in range(50):
                prep_args.append(
                    (args.odir, sl, args.wnm))
            pool.starmap(gen_mba, prep_args)