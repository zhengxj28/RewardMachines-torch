cd src
python -u run.py --project "Lifelong Learning" --algorithm "qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 >../logs/office/qrm.log 2>&1;
