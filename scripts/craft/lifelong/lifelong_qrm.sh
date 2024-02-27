cd src
nohup python -u run.py --project "Lifelong Learning" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="craft" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 >../logs/craft/qrm.log 2>&1 &
