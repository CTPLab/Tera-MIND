from utils import MOUSE
from config import TrainConfig
from utils.choices import GenerativeType, ModelName

def prep_config_parm(pth, bat, 
                     size, gpus,
                     stain, mouse, 
                     nrna, srna=4, 
                     method='ours',
                     is_test=False):
    conf = TrainConfig()
    conf.is_tot = False if method in ('ours', 'patch-dm') else True
    conf.method = method
    conf.beatgans_gen_type = GenerativeType.ddpm
    conf.beta_scheduler = 'linear'
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_beatgans_resnet_two_cond = True
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.lr = 1e-4
    conf.net_ch = 64
    conf.warmup = 0
    conf.batch_size = bat
    conf.sample_size = gpus * 4 if srna==1 else gpus
    conf.patch_size = size
    conf.image_size = size
    conf.data_path = mouse
    conf.semantic_enc = False
    conf.semantic_path = ''
    conf.cfg = False
    
    assert mouse in MOUSE
    conf.rna_tpl = tuple(i for i in range(srna))
    print(srna, conf.rna_tpl)
    # see 60988.csv and 63885.csv
    conf.rna_num = nrna
    conf.stain = stain
    conf.use_pos = False 
    conf.fp16 = True
    conf.net_attn = (16, )
    conf.net_beatgans_resnet_use_zero_module = True
    conf.num_workers = 2
    conf.net_resblock_updown = True
    conf.accum_batches = 64 // bat
    conf.dropout = 0.1
    conf.gn_sz = conf.image_size // 16
    conf.lr = 2e-5

    if conf.image_size in (32, 64, 128):
        conf.net_ch_mult = (1, 2, 4, 8) 
        conf.net_enc_channel_mult = (1, 2, 4, 8, 8) 
        conf.input_size = conf.image_size
    else:
        raise NotImplementedError("Patch size not in [32, 64, 128]")
    
    conf.name = f'{mouse}_{conf.image_size}_{conf.rna_num}_{conf.stain}_{srna}_{method}'
    conf.dims = 3
    return conf