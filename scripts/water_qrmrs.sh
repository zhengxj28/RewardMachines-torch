cd src
for i in `seq 0 1`;
do
	nohup python -u run.py --algorithm="qrm-rs" --world="water" --seed=$i --use_wandb >../logs/water/qrmrs_seed$i.log 2>&1 &
done