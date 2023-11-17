cd src
for i in `seq 0 9`;
do
	nohup python -u run.py --algorithm="pporm" --world="office" --seed=$i --use_wandb >../logs/office/pporm_seed$i.log 2>&1 &&
	nohup python -u run.py --algorithm="pporm-rs" --world="office" --seed=$i --use_wandb >../logs/office/ppormrs_seed$i.log 2>&1
done
