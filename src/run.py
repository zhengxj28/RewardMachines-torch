import argparse, os, sys, time
import random
import wandb
import torch
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))  # location of current file
sys.path.append(os.path.join(file_path, ".."))  # abs project path

from src.algos.qrm import QRMAlgo
from src.algos.pporm import run_pporm_experiments
from get_params import *
from src.tester.saver import Saver

# The pickle library is asking me to have access to Ball and BallAgent from the main...
# Do not delete this line
from src.worlds.water_world import Ball, BallAgent


def run_experiment(args, tester, curriculum):
    alg_name = args.algorithm
    show_print = False
    use_cuda = args.use_cuda

    # # Baseline 1 (standard DQN with Michael Littman's approach)
    # if alg_name == "dqn":
    #     run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print)

    # # Baseline 2 (Hierarchical RL)
    # if alg_name == "hrl":
    #     run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm = False)

    # # Baseline 3 (Hierarchical RL with DFA constraints)
    # if alg_name == "hrl-rm":
    #     run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm = True)

    # QRM
    if alg_name in ["qrm", "qrm-rs"]:
        algo = QRMAlgo(tester, curriculum, show_print, use_cuda)
        # run_qrm_experiments(tester, curriculum, show_print, use_cuda)

    # PPORM
    # if alg_name in ["pporm", "pporm-rs"]:
    #     run_pporm_experiments(tester, curriculum, show_print, use_cuda)
    algo.run_experiments()


def get_wandb_config(alg_name, args, learning_params):
    if args.set_params:
        learning_params.lr = args.lr
        learning_params.buffer_size = args.buffer_size
        learning_params.batch_size = args.batch_size
        learning_params.entropy_loss_coef = args.e_coef
        learning_params.n_updates = args.n_updates

    if alg_name in ["qrm", "qrm-rs"]:
        config = {"seed": s,
                  "lr": learning_params.lr,
                  "batch_size": learning_params.batch_size,
                  "target_network_update_freq": learning_params.target_network_update_freq
                  }
    if alg_name in ["pporm", "pporm-rs"]:
        config = {"seed": s,
                  "lr": learning_params.lr,
                  "buffer_size": learning_params.buffer_size,
                  "batch_size": learning_params.batch_size,
                  "n_updates": learning_params.n_updates,
                  "policy_loss_coef": learning_params.policy_loss_coef,
                  "value_loss_coef" : learning_params.value_loss_coef,
                  "entropy_loss_coef": learning_params.entropy_loss_coef,
                  }
    return config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # EXAMPLE: python run.py --algorithm "qrm" --world "office" --seeds 0 1 2 3 4 --use_wandb

    # Getting params
    algorithms = ["qrm", "qrm-rs", "pporm", "pporm-rs"]
    worlds = ["office", "craft", "water"]

    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='qrm', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0],
                        help='A list of random seeds to run experiments.')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use wandb or not.')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Whether to use cuda or not.')
    parser.add_argument('--notes', default='', type=str,
                        help='Notes on the algorithm, shown in wandb.')


    # add learning params in command lines for tests
    parser.add_argument('--set_params', action='store_true', help='Whether to set learning parameters in command lines.')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--buffer_size', default=1000, type=int, help='buffer size')
    parser.add_argument('--batch_size', default=1000, type=int, help='(mini) batch size')
    parser.add_argument('--n_updates', default=10, type=int, help='updated times for ppo')
    parser.add_argument('--e_coef', default=0, type=float, help='entropy loss coef')


    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError(
        "Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not (0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")

    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    use_wandb = args.use_wandb

    filename = "office.txt" if world == "office" else "%s_%d.txt" % (world, map_id)
    project_path = os.path.dirname(__file__)
    experiment = os.path.join(project_path, "..", "experiments", world, "tests", filename)
    world += "world"

    use_rs = alg_name.endswith("-rs")

    for s in args.seeds:
        print("*" * 10, "seed:", s, "*" * 10)

        # get default params for each env, defined in get_params.py
        if world == 'officeworld':
            tester, curriculum = get_params_office_world(alg_name, experiment, use_rs, use_wandb)
        if world == 'craftworld':
            tester, curriculum = get_params_craft_world(alg_name, experiment, use_rs, use_wandb)
        if world == 'waterworld':
            tester, curriculum = get_params_water_world(alg_name, experiment, use_rs, use_wandb)

        learning_params = tester.learning_params
        # saver = Saver(alg_name, tester, curriculum)
        print("alg_name:", alg_name)
        if use_wandb:
            wandb_config = get_wandb_config(alg_name, args, learning_params)
            for key, value in wandb_config.items():
                print("%s:" % key, value)
            wandb.init(
                project="RewardMachines-torch",
                notes=args.notes,
                group=world,
                name=alg_name,
                config=wandb_config
            )
        time_init = time.time()
        setup_seed(s)
        run_experiment(args, tester, curriculum)
        print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")
        # saver.save_results(filename="seed%d.npy" % s)
        if use_wandb: wandb.finish()
