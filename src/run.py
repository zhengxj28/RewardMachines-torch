import argparse, os, sys, time
import random
import wandb

file_path = os.path.dirname(os.path.abspath(__file__))  # location of current file
sys.path.append(os.path.join(file_path, ".."))  # abs project path

from src.algos.qrm import run_qrm_experiments
from get_params import *
from src.tester.saver import Saver

# The pickle library is asking me to have access to Ball and BallAgent from the main...
from src.worlds.water_world import Ball, BallAgent


def run_experiment(args, tester, curriculum):
    alg_name = args.algorithm
    show_print = args.verbosity is not None
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
        run_qrm_experiments(alg_name, tester, curriculum, show_print, use_cuda)


if __name__ == "__main__":
    # EXAMPLE: python run.py --algorithm "qrm" --world "office" --seeds 0 1 2 3 4 --use_wandb

    # Getting params
    algorithms = ["qrm", "qrm-rs"]
    worlds = ["office", "craft", "water"]

    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='qrm', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    # parser.add_argument('--num_times', default=1, type=int,
    #                     help='This parameter indicated which map to use. It must be a number greater or equal to 1')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0],
                        help='A list of random seeds to run experiments.')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use wandb or not.')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Whether to use cuda or not.')
    parser.add_argument("--verbosity", help="increase output verbosity")

    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError(
        "Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not (0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")

    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    use_wandb = args.use_wandb

    filename = "office.txt" if world=="office" else "%s_%d.txt" % (world, map_id)
    project_path = os.path.dirname(__file__)
    experiment = os.path.join(project_path, "..", "experiments", world, "tests", filename)
    world += "world"

    use_rs = alg_name.endswith("-rs")

    for s in args.seeds:
        print("*"*10, "seed:", s, "*"*10)

        if world == 'officeworld':
            tester, curriculum = get_params_office_world(experiment, use_rs, use_wandb)
        if world == 'craftworld':
            tester, curriculum = get_params_craft_world(experiment, use_rs, use_wandb)
        if world == 'waterworld':
            tester, curriculum = get_params_water_world(experiment, use_rs, use_wandb)

        learning_params = tester.learning_params
        saver = Saver(alg_name, tester, curriculum)
        if use_wandb:
            wandb.init(
                project="RewardMachines-torch",
                group=world,
                name=alg_name,
                config={
                    "seed": s,
                    "epsilon": learning_params.epsilon,
                    "gamma": learning_params.gamma,
                    "lr": learning_params.lr,
                    "batch_size": learning_params.batch_size,
                    "target_network_update_freq": learning_params.target_network_update_freq
                }
            )
        time_init = time.time()
        random.seed(s)
        run_experiment(args, tester, curriculum)
        print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")
        saver.save_results(filename="seed%d.npy"%s)
        if use_wandb: wandb.finish()
