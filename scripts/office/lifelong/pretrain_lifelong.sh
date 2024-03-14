cd src
nohup python -u run.py --project "Lifelong Learning" --wandb_name "pretrain" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 --save_model_name "phase0" --map 5 >../logs/craft/pretrain.log 2>&1 &
