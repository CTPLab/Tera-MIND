python -m test_brn --port 38850 --batch_size 1  --patch_size 64 \
    --data_path Data/MERFISH_50/  --mouse 638850 \
    --ckpt_pth checkpoints/638850_64_229_all_4_ours/last.ckpt \
    --out_dir MBA/0_final/timestep --hst 256 --wst 256 --hnm 286 --wnm 414