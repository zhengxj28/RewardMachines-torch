cd src
nohup python -u run.py --project "Stochastic Reward Machines" --algorithm="qsrm" --load_rm_mode "srm" --world="water" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 >../logs/water/qsrm.log 2>&1 &
