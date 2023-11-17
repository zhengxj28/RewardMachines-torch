cd src
for i in `seq 0 9`;
do
	nohup python -u run.py --algorithm="qrm" --world="office" --seed=$i --use_wandb >../logs/office/qrm_seed$i.log 2>&1 &&
	nohup python -u run.py --algorithm="qrm-rs" --world="office" --seed=$i --use_wandb >../logs/office/qrmrs_seed$i.log 2>&1
done
