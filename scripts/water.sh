cd src
for i in `seq 0 9`;
do
	nohup python -u run.py --algorithm="qrm" --world="water" --seed=$i --use_wandb >../logs/water/qrm_seed$i.log 2>&1 &&
done