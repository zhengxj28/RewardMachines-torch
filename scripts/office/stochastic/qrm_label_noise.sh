cd src
for noise in 0.001 0.005 0.01 0.05;
do
nohup python -u run.py --project "Stochastic Reward Machines" --algorithm="qrm" --load_rm_mode "files" --world="office" --label_noise $noise --use_wandb --use_cuda >../logs/office/qrm_$noise.log 2>&1;
done &