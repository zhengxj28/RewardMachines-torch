cd src
nohup python -u run.py --project "Lifelong Learning"  --wandb_name "qrmrs" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --use_wandb --use_cuda --use_rs --seeds 0 1 2 3 >../logs/office/qrmrs.log 2>&1 &
