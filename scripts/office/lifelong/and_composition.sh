cd src
nohup python -u run.py --project "Value Composition" --wandb_name "and_composition" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 --load_model_name "source" --map 2 --value_com "average" >../logs/office/and1.log 2>&1 &
sleep 60
nohup python -u run.py --project "Value Composition" --wandb_name "and_composition" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 --load_model_name "source" --map 2 --value_com "max" >../logs/office/and2.log 2>&1 &
sleep 60
nohup python -u run.py --project "Value Composition" --wandb_name "and_composition" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 --load_model_name "source" --map 2 --value_com "left" >../logs/office/and3.log 2>&1 &
sleep 60
nohup python -u run.py --project "Value Composition" --wandb_name "and_composition" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 --load_model_name "source" --map 2 --value_com "right" >../logs/office/and4.log 2>&1 &
sleep 60
nohup python -u run.py --project "Value Composition" --wandb_name "and_composition" --algorithm "lifelong_qrm" --load_rm_mode "formulas" --world="office" --label_noise 0 --use_wandb --use_cuda --seeds 0 1 2 3 --load_model_name "source" --map 2 --value_com "none" >../logs/office/and5.log 2>&1 &
