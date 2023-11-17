cd src
nohup python -u run.py --algorithm="pporm" --world="water" --seed=0 --use_wandb --use_cuda>../logs/water/pporm_seed0.log 2>&1 &