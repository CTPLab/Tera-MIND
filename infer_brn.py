import zarr
import pyvips
import argparse
import numpy as np
import multiprocessing as mp

from pathlib import Path
from einops import rearrange


def get_ome(im, pth, check_alpha=True, is_small=False):
    r"""
    Convert the pyvips image to *.tif such that the output 
        is comptabile to QuPath 

    Args:
        im: the pyvips image
        pth: Path to the json file
    """

    # openslide will add an alpha ... drop it
    if im.hasalpha() and check_alpha:
        im = im[:-1]

    im = pyvips.Image.arrayjoin(im.bandsplit(), across=1)    
    # set minimal OME metadata
    # before we can modify an image (set metadata in this case), we must take a
    # private copy
    im = im.copy()
    # im.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    im.set_type(pyvips.GValue.gstr_type, "image-description",
                f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    SizeC="{im.bands}"
                    SizeT="1"
                    SizeX="{im.width}"
                    SizeY="{im.height}"
                    SizeZ="1"
                    Type="uint8">
            </Pixels>
        </Image>
    </OME>""")
    if is_small:
        im.tiffsave(pth, tile=True, compression='jpeg')
    else:
        im.tiffsave(pth, tile=True, compression='jpeg',
                    pyramid=True, bigtiff=True,
                    tile_height=256, tile_width=256)


def gen_col(gdir, odir,
            hst=20480, wst=40960, 
            hnm=16, pw=0,
            is_gen=False, 
            size=256, slc=50):
    shf = size // 2
    wsz = pw * size
    gen_col = []
    for h in range(hnm):
        hsz = h * size
        g_pos = [hst+hsz, hst+size+hsz,
                 wst+wsz, wst+size+wsz]
        if not is_gen:
            g_pos += [
                hst+hsz-shf, hst+size+hsz+shf,
                wst+wsz-shf, wst+size+wsz+shf]
        g_nam = '_'.join([str(g) for g in g_pos])
        g_npy = zarr.load(gdir / f'{g_nam}.zip')
        if not is_gen:
            g_npy = g_npy[:, shf:shf+size, shf:shf+size]
        g_npy = rearrange(g_npy, '(c s) h w -> (s c) h w', s=slc)
        if is_gen:
            g_npy = ((g_npy + 1) * 127.5).astype(np.uint8)
        gen_col.append(g_npy[..., None])

    for sl in range(g_npy.shape[0]):
        (odir / f'{sl}').mkdir(parents=True, exist_ok=True)
        gen_slc = [pyvips.Image.new_from_array(gen_col[h][sl]) for h in range(hnm)]
        gen_slc = pyvips.Image.arrayjoin(gen_slc,
                                         across=1)
        get_ome(gen_slc, str(odir / f'{sl}/{pw}.tif'), is_small=True)
    print(f'Col {pw} is done')


def gen_mba(odir, slc, page,
            wnm=16):
    if not (odir / f'{slc}').is_dir():
        print(str(odir / f'{slc}'), 'does not exist.')
        return 
    gcol = [pyvips.Image.new_from_file(str(odir / f'{slc}/{w}.tif'),
                                       access='sequential')
            for w in range(wnm)]
    gen_wsi = pyvips.Image.arrayjoin(gcol, across=wnm)
    get_ome(gen_wsi, str(odir / f'all_{slc}.tif'))
    wsi = pyvips.Image.new_from_file(str(odir / f'all_{slc}.tif'),
                                     page=page)
    wsi.write_to_file(str(odir / f'all_{slc}.jpg'))
    print(gen_wsi.bands, gen_wsi.height, gen_wsi.width, 
          f'all_{slc}.tif is done')


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
    parser.add_argument('--page', type=int, default=5,
                        help='The page of pyramid')
    parser.add_argument('--is_gen', action='store_true',
                        help='generate row of mbe')
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
                    args.hnm, pw, args.is_gen))
            pool.starmap(gen_col, prep_args)

    if args.gen_mba:
        with mp.Pool(processes=args.core) as pool:
            prep_args = list()
            for sl in range(100):
                prep_args.append(
                    (args.odir, sl, args.page, args.wnm))
            pool.starmap(gen_mba, prep_args)