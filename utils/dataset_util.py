import argparse
import pandas as pd

from utils import HBR
from pathlib import Path


def prep_dlst(dir, dir_exl=None,       
              hst=20480, wst=40960, size=256,
              hnm=16, wnm=16, to_csv=False):
    gn_lst, rel_pad =[], size // 2
    mouse = dir.name.split('_')[-1]
    # Here, we assume no boundary issue
    for pw in range(wnm):
        wsz = pw * size
        for ph in range(hnm):
            hsz = ph * size
            g_pos = [hst + hsz, hst + size + hsz,
                     wst + wsz, wst + size + wsz]
            r_pos = g_pos + [max(hst + hsz - rel_pad, 0),
                             min(hst + size + hsz + rel_pad, 73728),
                             max(wst + wsz - rel_pad, 0),
                             min(wst + size + wsz + rel_pad, 106496)]
            r_nam = '_'.join([str(r) for r in r_pos])
            is_exl = dir_exl is not None and (dir_exl / f'{r_nam}.npz').is_file()
            if not is_exl:
                assert (dir / f'{r_nam}.npz').is_file()
                gn_lst.append(str(dir / f'{r_nam}.npz'))
    assert len(gn_lst) == len(set(gn_lst))
    print(f'The amout of training data for {mouse}: {len(gn_lst)}')
    suf = '' if dir_exl is None else '_exl'
    if to_csv:
        pd.DataFrame({'pth': gn_lst}).\
            to_csv(f'utils/{mouse}{suf}.csv', 
                   index=False)
        
def cmp_gene(ms_gn, hm_gn):
    hm_gn = [m.lower() for m in hm_gn]
    out = {i:m for i, m in enumerate(ms_gn) if m.lower() in hm_gn}
    print(len(out))
    print(out)
    # print(ms_set)
    # print(hm_set)


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=Path, default=Path('Data/'),
                        help='Path to generated tiles')
    parser.add_argument('--dir_exl', type=Path, default=None,
                        help='Path to real tiles')
    parser.add_argument('--hst', type=int, default=0,
                        help='The starting location of height')
    parser.add_argument('--wst', type=int, default=0,
                        help='The starting location of width')
    parser.add_argument('--hnm', type=int, default=288,
                        help='The number of rows (73728 // 256) = 288')
    parser.add_argument('--wnm', type=int, default=416,
                        help='The number of cols (106496 // 256) = 416')
    parser.add_argument('--to_csv', action='store_true',
                        help='Whether save to csv')
    parser.add_argument('--prep_dlst', action='store_true',
                        help='prep data list to avoid scanning the folder on the cluster')
    parser.add_argument('--cmp_gene', action='store_true',
                        help='Compare the gene list between MBE and HBR')
    args = parser.parse_args()

    if args.prep_dlst:
        prep_dlst(args.dir, args.dir_exl, args.hst, args.wst,
                  hnm=args.hnm, wnm=args.wnm, to_csv=args.to_csv)
    
    if args.cmp_gene:
        ms_csv = Path('utils/60988_gnm.csv')
        ms_gene = pd.read_csv(str(ms_csv))['gene']
        # ms_gene = ms_gene.to_list()[:229]
        cmp_gene(ms_gene, HBR)