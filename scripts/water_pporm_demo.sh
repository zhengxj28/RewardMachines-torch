cd src
nohup python -u run.py --algorithm="pporm" --load_rm_mode "formulas" --world="water" --use_wandb --use_cuda >../logs/water/pporm_demo.log 2>&1 &