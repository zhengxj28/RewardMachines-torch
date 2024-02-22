cd src
python -u run.py --project "Stochastic Reward Machines" --algorithm="qsrm" --load_rm_mode "srm" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 >../logs/office/qsrm.log 2>&1;
