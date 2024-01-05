cd src
nohup python -u run.py --algorithm="ltlenc_dqn" --load_rm_mode "formulas" --world="water" --use_wandb --use_cuda >../logs/water/ltlenc_dqn_demo.log 2>&1 &