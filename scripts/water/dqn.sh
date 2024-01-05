cd src
nohup python -u run.py --algorithm="dqn" --load_rm_mode "formulas" --world="water" --use_wandb --use_cuda >../logs/water/dqn_demo.log 2>&1 &