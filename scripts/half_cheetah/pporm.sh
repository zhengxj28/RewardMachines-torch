cd src
nohup python -u run.py --algorithm="pporm" --load_rm_mode "files" --world="half_cheetah" --use_wandb --use_cuda >../logs/half_cheetah/pporm.log 2>&1 &