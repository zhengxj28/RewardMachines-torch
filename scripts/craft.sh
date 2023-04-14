cd src
for i in `seq 0 9`;
do
	nohup python -u run.py --algorithm="qrm" --world="craft" --seed=$i --use_wandb >../logs/craft/qrm_seed$i.log 2>&1 &&
	nohup python -u run.py --algorithm="qrm-rs" --world="craft" --seed=$i --use_wandb >../logs/craft/qrmrs_seed$i.log 2>&1
done