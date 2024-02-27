cd src
nohup python -u run.py --project "Stochastic Reward Machines" --algorithm="qrm" --load_rm_mode "srm" --world="craft" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 >../logs/craft/qrm.log 2>&1 &
