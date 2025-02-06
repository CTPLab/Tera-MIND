import cv2
import copy
import torch
import random
import pyvips
import argparse
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from einops import reduce
from PIL import Image, ImageEnhance

from utils import MOUSE, MALL

def add_bbx(input, bbx_clr=(255, 255, 255), 
            bbx_len=4, bdry=(0, 1, 2, 3)):
    assert len(input.shape) in (3, 4)
    for i in range(3):
        for j in bdry:
            slc = [[None, None], [None, None]]
            # if j == 0, then slc = [[None, bbx_len], [None, None]]
            slc[j // 2][(j + 1) % 2] = (-1) ** j * bbx_len
            # single image, assume the color chn is the last dim
            if len(input.shape) == 3:
                assert input.shape[2] == 3
                slicing = (slice(*slc[0]),) + (slice(*slc[1]),) + (i,)
            # multiple image as a batch, assume the color chn is the 2nd dim
            else:
                assert input.shape[1] == 3
                slicing = (slice(None),) + (i,) + \
                    (slice(*slc[0]),) + (slice(*slc[1]),)
            input[slicing] = bbx_clr[i]
        # input[:bbx_len, :, i] = bbx_clr[i]
        # input[-bbx_len:, :, i] = bbx_clr[i]
        # input[:, :bbx_len, i] = bbx_clr[i]
        # input[:, -bbx_len:, i] = bbx_clr[i]
    return input


def add_contrast(input, axis, weight=1.5):
    assert len(input.shape) in (3, 4)
    assert input.shape[axis] in (1, 3, 6)
    assert 0 <= axis < len(input.shape)

    if input.shape[axis] == 6:
        for ch in range(6):
            slicing = (slice(None),) * axis + (ch,)
            add_wei = 0
            if ch == 2:  # Actin
                add_wei = 0.5
            elif ch == 4:  # Mitochondria
                add_wei = 1.5
            input[slicing] *= (weight + add_wei)
    input[input < 0] = 0
    input[input > 1] = 1
    return input


def sort_img(idir, is_vis=False):
    # TODO: The func may need refactored depending on the registration
    r"""
    Sort the image order based on their converted integer name such that 
        the order of the tiff images matches the QuuPath folder name 1, 2, ... 

    Args:
        idir: Dir to the images
    """

    def nam2int(i, is_vis=is_vis):
        if is_vis:
            return int((i.stem.split('_')[1]).replace('HK', '60'))
        return int(i.stem.replace('HK', '00'))
    return sorted(idir, key=nam2int)


def gen_roi(img, wei, msk, seed, 
            num=2, sz=256, top_n=16, blen=8):
    assert img.shape[:2] == wei.shape
    # Sort the region based on pixel intensity
    # Random select {num} roi from top {n} brigh regions
    h, w = wei.shape
    h_n, w_n = h // sz, w // sz
    out = reduce(wei[:h_n*sz,:w_n*sz], 
                 '(p1 h) (p2 w) -> (p1 p2)', 'sum', 
                 p1=h_n, p2=w_n)
    thr = reduce(msk[:h_n*sz,:w_n*sz], 
                 '(p1 h) (p2 w) -> (p1 p2)', 'sum', 
                 p1=h_n, p2=w_n)
    idx = np.argsort(-out)
    # Make sure that pos is not far from mask
    pos = [(i//w_n, i%w_n) for i in idx if thr[i] >= sz and (i//w_n + i%w_n)%2==0]
    pos = pos[:top_n]
    random.seed(seed)
    random.shuffle(pos)
    # # Sort the selected roi based on col number
    pos = list(sorted(pos[:num], key=lambda x: x[1]))
    # pos = pos[:num]
    roi_img, roi_wei, roi_msk = [], [], []
    for p in pos:
        # print(p)
        row = slice(p[0]*sz, (p[0]+1)*sz)
        col = slice(p[1]*sz, (p[1]+1)*sz)
        roi_i = copy.deepcopy(img[row, col])
        # zoom in roi with thin bbox len
        roi_img.append(add_bbx(roi_i, bbx_len=1))
        roi_wei.append(wei[row, col])
        roi_msk.append(msk[row, col])
        img[row, col]=add_bbx(img[row, col], 
                              bbx_len=blen)
    # roi = np.concatenate(roi, 1)
    return img, roi_img, roi_wei, roi_msk


def onto_overlay(top, bottom, img, out_pth, 
                 onto='main', add_onto='all', alpha=100, bright=None):
    h, w = img.shape[:2]
    if not (top is None or bottom is None):
        assert top.shape == bottom.shape
        h_s, w_s = top.shape[:2]
        h_new = int((w / w_s) * h_s)
        top = Image.fromarray(top).resize((w, h_new))
        bottom = Image.fromarray(bottom).resize((w, h_new))
        wsi = np.concatenate((top, img, bottom))
    else:
        wsi, h_new = img, 0
    print(out_pth, wsi.shape, bright)
    if bright is not None:
        # commonly, we set bright = 2
        wsi = Image.fromarray(wsi)
        enhancer = ImageEnhance.Brightness(wsi)
        wsi = np.array(enhancer.enhance(bright))

    if add_onto is None:
        Image.fromarray(wsi).save(str(out_pth))
        return 
    
    wsi_mk = pyvips.Image.new_from_array(wsi)
    if add_onto == 'quarter':
        msk = np.zeros_like(onto)
        msk[:h//2, :w//2] = onto[:h//2, :w//2]
    elif add_onto in ('main', 'bhalf'):
        msk = np.zeros_like(onto)
        msk[h//2:] = onto[h//2:]
    elif add_onto == 'half':
        msk = np.zeros_like(onto)
        msk[:, :w//2] = onto[:, :w//2]
    elif add_onto == 'rhalf':
        msk = np.zeros_like(onto)
        msk[:, w//2:] = onto[:, w//2:]
    elif add_onto == 'thalf':
        msk = np.zeros_like(onto)
        msk[:h//2] = onto[:h//2]
    elif add_onto == '3quarter':
        msk = onto
        msk[:h//2, w//2:] = 0
    elif add_onto == 'all':
        msk = onto
    if h_new != 0:
        pad = np.zeros((h_new, w, 3))
        msk = np.concatenate((pad, msk, pad))
    wei = (msk.sum(-1) != 0)[:, :, None]
    msk = pyvips.Image.new_from_array(msk)
    msk = msk.bandjoin(alpha)
    wsi_mk.composite(msk, 'over').write_to_file(str(out_pth))
    # workaround
    wsi_mk = np.array(Image.open(out_pth))
    if add_onto == 'main':
        wsi[h//2:] = 0
        stn_pth = str(out_pth).replace('.jpg', '_stn.jpg')
        Image.fromarray(wsi.astype('uint8')).save(stn_pth)
        Image.fromarray((wei*wsi_mk).astype('uint8')).save(str(out_pth)) 
    else:
        wsi = (1-wei)*wsi + wei*wsi_mk
        Image.fromarray(wsi.astype('uint8')).save(str(out_pth))
    return


def gen_zoom(mse, ref, dapi, polyt, onto, out_pth,
             add_onto=None, alpha=100, is_wsi=False,
             roi_nm=2, roi_sz=128,
             cll_nm=4, cll_sz=16):
    slc = int(out_pth.stem)
    dapi = np.array(Image.open(str(dapi)))
    polyt = np.array(Image.open(str(polyt)))
    onto = np.array(Image.open(str(onto)))
    msk = onto.sum(-1) != 0
    h, w = msk.shape
    assert dapi.shape == polyt.shape == (h, w)
    # PolyT green and DAPI blue
    img = np.stack([np.zeros_like(dapi),
                    polyt, dapi], -1)
    out = copy.deepcopy(img)
    if is_wsi:
        onto_overlay(None, None, out, out_pth,
                     onto, add_onto, alpha)
        return

    # Imposing same bbox when using same reference image
    ref = np.array(Image.open(str(ref))) 
    wei = ref.astype('float') * msk
    print(out_pth, dapi.shape, img.shape)
    # get top left, top right, 
    # bottom left, bottom righ region
    top, bottom = [], []
    for r in range(2):
        row = slice(r*h//2, (r+1)*h//2)
        for c in range(2):
            col = slice(c*w//2, (c+1)*w//2)
            seed = mse * (r + 1) + c
            top_n = 3 if slc in (46, 47, 48, 49) and mse == 638850 else 6 
            img_o, roi_i, roi_w, roi_m = gen_roi(
                img[row, col], wei[row, col], msk[row, col], 
                seed, roi_nm, roi_sz, top_n, blen=16)
            out[row, col] = img_o
            # print(r,c, img[row, col].shape, len(roi_i), len(roi_w), len(roi_m))
            for im, wi, mk in zip(roi_i, roi_w, roi_m):
                roi_o, cll_i = gen_roi(im, wi, mk, 0, 
                                       cll_nm, cll_sz, top_n=16, blen=1)[:2]
                cll_i = np.concatenate(cll_i, 1)
                h_c, w_c = cll_i.shape[:2]
                fac = roi_sz / (cll_nm * cll_sz)
                assert fac == int(fac) and w_c == cll_nm * cll_sz
                cll_i = Image.fromarray(cll_i).resize((int(fac * w_c), int(fac * h_c)))

                if r == 0:
                    roi_top = np.concatenate((cll_i, roi_o))
                    top.append(roi_top)
                else:
                    roi_bottom = np.concatenate((roi_o, cll_i))
                    bottom.append(roi_bottom)

    top = np.concatenate(top, 1)
    bottom = np.concatenate(bottom, 1)
    onto_overlay(top, bottom, out, out_pth,
                 onto, add_onto, alpha)

def calc_df(pth, rows_avg):
    df = pd.read_csv(pth, index_col=0)
    stt_avg = df.loc[['clean_s', ]].astype('float').to_numpy()
    if 'DAPI' in str(pth):
        rows = [i for i in df.index if i in ['nsize_ot', 'nsize_gt', 'narea_ot', 'narea_gt']]
        stt = df.loc[rows].to_numpy()
        print(pth.stem, stt[:, -1])
    else:
        rows = [i for i in df.index if i not in rows_avg]
        stt = df.loc[rows].to_numpy()
        # stt = df.loc[rows].to_numpy()
        print(pth.stem, stt_avg[:, :-1].mean(-1), stt_avg[:, :-1].std(-1), stt[:, -1])

def csv2tab(all_pth, cols=['nsize_ot', 'nsize_gt', 'narea_ot', 'narea_gt']):
    df_all = pd.read_csv(all_pth, index_col=0).transpose()
    df_all = df_all[['clean', 'clip', 'clean_s', 'psnr', 'ssim']]
    dapi_pth = all_pth.replace('all', 'DAPI')
    df_dapi = pd.read_csv(dapi_pth, index_col=0).transpose()
    df_all[cols] = df_dapi[cols]
    for col in ['psnr', 'ssim'] + cols:
        stat = df_all[col].str.split(' ', expand=True)
        df_all[col] = stat[0].astype('float')
        df_all[f'{col}_std'] = stat[1].astype('float')
        if 'ssim' in col:
            df_all[col] *= 10
            df_all[f'{col}_std'] *= 10
        elif 'narea' in col:
            df_all[col] *= 0.01
            df_all[f'{col}_std'] *= 0.01
    print(all_pth, df_all.transpose())
    df_out = df_all.transpose().iloc[:, :-1]
    df_out = df_out.iloc[:, ::-1]
    df_out.columns = 49 - df_out.columns.astype(int)
    out = all_pth.replace('all', 'order')
    df_out.to_csv(out)

                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility function for proc MERFISH 3D data.')
    parser.add_argument('--dir',
                        type=Path,
                        default=Path('Data/visual'),
                        help='Dir to NanoString dataset.')
    parser.add_argument('--mouse', type=str,
                        default='638850',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--shf', type=int, default=8,
                        help='The boundary pixel needed to be excluded (256 / 32 = 8)')
    parser.add_argument('--crop_msk',
                        action='store_true',
                        help='Crop label mask')
    parser.add_argument('--merg_msk',
                        action='store_true',
                        help='merge mask and gt')
    parser.add_argument('--alpha', type=int, default=160,
                        help='transparent')
    parser.add_argument('--gen_zoom',
                        action='store_true',
                        help='Whether to generate zoomed figure')
    parser.add_argument('--core', type=int, default=8,
                        help='The number of cols')
    parser.add_argument('--gen_im', type=str,
                        default='gt',
                        choices=['gt', 'gen'])
    parser.add_argument('--add_onto', type=str,
                        default='half',
                        choices=['quarter', 'main', 'half', 'thalf', 'bhalf', 'rhalf', '3quarter', 'all'])
    parser.add_argument('--is_msk',
                        action='store_true',
                        help='Whether msk or contour')
    parser.add_argument('--is_wsi',
                        action='store_true',
                        help='Whether only wsi or with zoom in details')
    parser.add_argument('--calc_df',
                        action='store_true',
                        help='merge mask and gt')
    parser.add_argument('--calc_gene',
                        action='store_true',
                        help='merge mask and gt')
    parser.add_argument('--stain', type=str,
                        default='all',
                        choices=['DAPI', 'PolyT', 'all'])
    parser.add_argument('--gen_attn',
                        action='store_true',
                        help='whether generate attn brain')
    parser.add_argument('--pathway',
                        default='DOPA',
                        choices=['DOPA', 'GLUT'])
    parser.add_argument('--is_attn',
                        action='store_true',
                        help='Whether msk or contour')
    parser.add_argument('--gene',
                        default=None,
                        choices=['Nr4a2', 'Th', 'Slc17a6', 'Slc17a7'])
    parser.add_argument('--to_tab',
                        action='store_true',
                        help='Convert the df format for keynote table')
    args = parser.parse_args()

    if args.gen_zoom:
        with mp.Pool(processes=args.core) as pool:
            mark = 'mask' if args.is_msk else 'cnt' 
            res = 'fig_wsi' if args.is_wsi else 'zoom'
            prep_args = list() 
            # reference
            rdir = args.dir / f'gt_{args.mouse}'
            idir = args.dir / f'{args.gen_im}_{args.mouse}'
            mdir = args.dir / f'{mark}_{args.mouse}_crop'
            odir = args.dir / f'results/{res}/{args.gen_im}_{args.mouse}'
            odir.mkdir(parents=True, exist_ok=True)
            add_onto = args.add_onto if args.gen_im == 'gt' or args.is_wsi else None
            for i in range(50):
                parg = (int(args.mouse),
                        rdir / f'all_{2*i}.jpg',
                        idir / f'all_{2*i}.jpg',
                        idir / f'all_{2*i+1}.jpg',
                        mdir / f'{i}.jpg',
                        odir / f'{i}.jpg', 
                        add_onto, args.alpha, args.is_wsi)
                prep_args.append(parg)
            pool.starmap(gen_zoom, prep_args)
    
    if args.gen_attn:
        if args.gene is None:
            bright, expr, inm = None, 'expr/all', '.jpg'
        else:
            bright, expr, inm = 2, f'expr/{args.gene}', f'_{args.gene}.jpg'
        if not (args.is_attn or args.gene is None):
            args.add_onto = None

        with mp.Pool(processes=args.core) as pool:
            res = 'att_all' if args.is_attn else expr
            prep_args = list() 
            idir = args.dir / f'results/fig_attn/attn_{args.mouse}/{args.pathway}/{res}'
            mdir = args.dir / f'mask_{args.mouse}_crop'
            odir = args.dir / f'results/fig_attn/res_{args.mouse}/{args.pathway}_{args.add_onto}/{res}'
            odir.mkdir(parents=True, exist_ok=True)
            for i in range(50):
                im_nm = f'{i}.jpg' if args.is_attn else f'{i}{inm}'
                im = np.array(Image.open(str(idir / im_nm)))
                h_i, w_i = im.shape[:2]
                msk = np.array(Image.open(str(mdir / f'{i}.jpg')))
                img = np.zeros_like(msk)
                img[:h_i, :w_i] = im
                parg = (None, None,
                        img,
                        odir / f'{i}.jpg', 
                        msk,
                        args.add_onto, args.alpha, bright)
                prep_args.append(parg)
            pool.starmap(onto_overlay, prep_args)

    if args.calc_gene:
        gn_lst = MALL[args.pathway]
        # Figure 3
        gn_slc = [48, 46, 20, 18] if args.pathway == 'DOPA' else [46, 34, 24, 14]
        gene_all = pd.read_csv(f'utils/{args.mouse[:-1]}_gnm.csv')['gene'].to_list()
        if args.mouse == '638850':
            gene_all = gene_all[:229]
        gn_idx = [gene_all.index(g) for g in gn_lst]
        print(gn_idx, gn_lst)
        pth = args.dir / f'{args.mouse}_all_False/all.pt'
        stt = torch.load(str(pth))

        for slc in gn_slc:
            for gi in gn_idx:
                mean = stt['mean'][(slc+1)*500+gi]
                std = stt['std'][(slc+1)*500+gi]
                if gene_all[gi] == 'Slc17a7':
                    mean = torch.log10(mean + 1)
                    std = torch.log10(std + 1)
                print(slc, gi, mean, std)

    if args.calc_df:
        for pth in args.dir.glob('*.csv'):
            if args.stain in pth.stem and args.mouse in pth.stem:
                calc_df(pth, ['clean', 'clean_s', 'clip'])

    if args.to_tab:
        for pth in args.dir.glob('all_*.csv'):
            csv2tab(str(pth))
    
    shf = args.shf    
    if args.crop_msk:
        msk_pth = (args.dir / args.mouse / 'msk').glob('*.jpg')
        out_dir = (args.dir / args.mouse / 'crop_msk')
        out_dir.mkdir(parents=True, exist_ok=True)
        msk_pth = sort_img(list(msk_pth), is_vis=True)
        if args.mouse == '609882':
            msk_pth = msk_pth[::-1]
        wsi_lst = list()
        for didx, ddir in enumerate(msk_pth):
            if didx not in MOUSE[args.mouse][1]:
                print(didx, ddir)
                wsi_lst.append(ddir)
            else:
                print('exclude', didx, ddir)
        print(len(wsi_lst))

        for lid, wsi in enumerate(wsi_lst):
            img = cv2.imread(str(wsi))
            cv2.imwrite(str(out_dir / f'{lid}.jpg'), 
                        img[shf:-shf, shf:-shf])
            

    if args.merg_msk:
        msk_dir = (args.dir / f'mask_{args.mouse}_crop')
        gt_dir = Path('/home/jwu/Downloads/attn_638850/Slc17a6_total')
        out_dir = (args.dir / f'mask_{args.mouse}_Slc17a6')
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(50):
            # dapi = gt_dir / f'all_{i*2}.jpg'
            # dapi = cv2.imread(str(dapi))[:, :, 0]
            # polyt = gt_dir / f'all_{i*2+1}.jpg'
            # polyt = cv2.imread(str(polyt))[:, :, 0]
            # img = np.stack([np.zeros_like(dapi), 
            #                 np.zeros_like(dapi), 
            #                 np.zeros_like(dapi)], 
            #                -1).astype('uint8')
            img = cv2.imread(str(gt_dir / f'{i}.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = pyvips.Image.new_from_array(img)
            msk = cv2.imread(str(msk_dir / f'{i}.jpg')).sum(-1)
            msk[msk!=0] = 255
            msk = np.stack([msk, msk, msk], -1).astype('uint8') 
            msk = pyvips.Image.new_from_array(msk)
            # img.write_to_file(str(out_dir / f'{i}_gt.jpg'))
            # msk = pyvips.Image.new_from_file(str(msk_dir / f'{i}.jpg'))
            msk = msk.bandjoin(args.alpha)
            out_pth = out_dir / f'{i}_{args.alpha}.png'
            img.composite(msk, 'over').write_to_file(str(out_pth))
            print(f'{i}-th image is done.')

        
    # for tid, tdir in enumerate(sort_img(tif_dir)):
    #     tidx, wnm = tid + 1, tdir.name[-4:]
    #     odir = tdir / 'out'
    #     aff_arr = aff_dir / str(tidx) / AFF_NM
    #     roi_ply = aff_dir / str(tidx) / ROI_NM
    #     if aff_arr.is_file():
    #         aff_out = aff2arr(aff_arr)
    #         roi_out = roi2arr(str(roi_ply), ann_dct)
    #         print(tidx, wnm, len(roi_out), roi_out.keys())
    #         msk_ome = odir / f'aff_{wnm}_msk.ome.tiff'
    #         if not msk_ome.is_file():
    #             mome = poly2ome(roi_out, aff_out,
    #                             out_wid, out_hei, 'arr_clr')
    #             tif2ome(mome[:1], str(msk_ome))
    #             print(f'{msk_ome} is done.')

    #         if debug:
    #             msk_jpg = odir / f'aff_{wnm}_msk.jpg'
    #             if not msk_jpg.is_file():
    #                 mjpg = poly2ome(roi_out, aff_out,
    #                                 out_wid, out_hei, 'clr')
    #                 # need to remove the last alpha channel
    #                 mjpg = mjpg.reduce(scale, scale)[:-1]
    #                 mjpg.write_to_file(str(msk_jpg))
    #             else:
    #                 mjpg = pyvips.Image.new_from_file(str(msk_jpg))
    #             mjpg = mjpg.bandjoin(100)

    #             aff_jpg = odir / f'aff_{wnm}_DAPI.jpg'
    #             if aff_jpg.is_file():
    #                 affj = pyvips.Image.new_from_file(str(aff_jpg))
    #                 aff_fuse = odir / f'aff_{wnm}_DAPI_msk.jpg'
    #                 if not aff_fuse.is_file():
    #                     affj.composite(mjpg, 'over').write_to_file(
    #                         str(aff_fuse))
    #                     print(f'{aff_fuse} is done.')
    #             else:
    #                 print(f'{aff_jpg} does not exist, need run proc_img(...) first')