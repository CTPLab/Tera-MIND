import torch
import argparse
from pathlib import Path
from experiment import train
from config_parm import prep_config_parm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='batch size (all gpus)')
    parser.add_argument('--patch_size', '-ps', type=int, default=64,
                        help='model base patch size')
    parser.add_argument('--data_path', '-d', type=Path, default="./dataset",
                        help='dataset path')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    parser.add_argument('--stain', type=str,
                        default='all',
                        choices=['DAPI', 'PolyT', 'all'],
                        help='Folder to different mouses')
    parser.add_argument('--rna_slc', type=int, 
                        choices=(1, 4, 8, 16),
                        help='use random single slice')
    parser.add_argument('--to_hbr', action='store_true',
                        help='use gene list for human brain generalization')
    parser.add_argument('--method', type=str,
                        default='ours',
                        choices=['ours', 'patch-dm', 'sinf'],
                        help='models')

    args = parser.parse_args()
    gpus = [i for i in range(torch.cuda.device_count())]
    nrna = 500 if args.mouse in ('609882', '609889') else 229
    if args.to_hbr:
        nrna = 81
        assert args.mouse == '638850' and args.rna_slc == 1
    conf = prep_config_parm(args.data_path, args.batch_size, 
                            args.patch_size, len(gpus), 
                            args.stain, args.mouse,
                            nrna, args.rna_slc, args.method)
    
    train(conf, gpus=gpus)