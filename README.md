# RewardMachines-torch

Q-learning for Reward Machines (QRM) and QRM with reward shaping (QRM-rs) algorithms implemented by pytorch. These algorithms are first proposed by Icarte et. al (2018).

### Running Example

Here is an running example in the OfficeWorld domain. Using bash shell to run experiments with random seed 0-9

```
bash scripts/office.sh
```

or using command to run experiment with random seed 0

```
python run.py --algorithm="qrm" --world="office" --seed=0 --use_wandb
```

Although using cuda `--use_cuda` is avaliable, we do not use it because the domains are relatively simple and running with cpu is faster than using cuda.


### Installation instructions

The code has the following requirements:

* python 3.8
* numpy 1.24.2
* gym 0.26.2
* torch 2.0.0 or 1.13.1 (either cpu or cuda version is available)


### Original paper

```
@inproceedings{icarte2018using,
  title={Using reward machines for high-level task specification and decomposition in reinforcement learning},
  author={Icarte, Rodrigo Toro and Klassen, Toryn and Valenzano, Richard and McIlraith, Sheila},
  booktitle={International Conference on Machine Learning},
  pages={2107--2116},
  year={2018},
  organization={PMLR}
}
```
