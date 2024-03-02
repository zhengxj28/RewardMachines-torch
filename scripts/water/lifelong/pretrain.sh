cd src
nohup python -u run.py --project "Value Composition" --wandb_name "pretrain" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="water" --label_noise 0 --use_wandb --use_cuda --seeds 0 --save_model_name "source" --map 1 >../logs/water/pretrain.log 2>&1 &
