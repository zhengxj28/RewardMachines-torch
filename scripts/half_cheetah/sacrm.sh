cd src
nohup python -u run.py --algorithm="sacrm" --load_rm_mode "files" --world="half_cheetah" --use_wandb --use_cuda >../logs/half_cheetah/ppo.log 2>&1 &