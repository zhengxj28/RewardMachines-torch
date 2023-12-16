cd src
nohup python -u run.py --algorithm="qrm" --load_rm_mode "formulas" --world="water" --use_wandb --use_cuda >../logs/water/qrm_seed0.log 2>&1 &