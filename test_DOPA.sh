python -m test_attn --batch_size 1  --patch_size 64 \
    --data_path Data/MERFISH_50 --mouse 638850 --port 18850 \
    --ckpt_pth checkpoints/638850_64_229_all_4_ours/last.ckpt \
    --out_dir MBA/0_vis/timestep --region -1 --path DOPA --calc_attn
