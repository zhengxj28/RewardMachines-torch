cd src
nohup python -u run.py --project "Lifelong Learning" --algorithm="lifelong_qrm" --load_rm_mode "formulas" --world="water" --use_wandb --use_cuda --seeds 0 >../logs/water/lifelong_qrm.log 2>&1 &