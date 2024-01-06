import argparse, os, sys, time
import random
import wandb
import torch
import numpy as np
import yaml
import copy

file_path = os.path.dirname(os.path.abspath(__file__))  # location of current file
sys.path.append(os.path.join(file_path, ".."))  # abs project path

from src.tester.saver import Saver

# The pickle library is asking me to have access to Ball and BallAgent from the main...
# Do not delete this line
from src.worlds.water_world import Ball, BallAgent


def run_experiment(args, tester, curriculum):
    alg_name = args.algorithm
    show_print = False
    use_cuda = args.use_cuda

    if alg_name == "qrm":
        from src.algos.qrm import QRMAlgo
        algo = QRMAlgo(tester, curriculum, show_print, use_cuda)
    elif alg_name == "dqn":
        from src.algos.dqn import DQNAlgo
        algo = DQNAlgo(tester, curriculum, show_print, use_cuda)
    elif alg_name == "ltlenc_dqn":
        from src.algos.ltlenc_dqn import LTLEncDQNAlgo
        algo = LTLEncDQNAlgo(tester, curriculum, show_print, use_cuda)
    elif alg_name == "ppo":
        from src.algos.ppo import PPOAlgo
        algo = PPOAlgo(tester, curriculum, show_print, use_cuda)
    elif alg_name == "pporm":
        from src.algos.pporm import PPORMAlgo
        algo = PPORMAlgo(tester, curriculum, show_print, use_cuda)
    else:
        raise NotImplementedError("Algorithm:" + alg_name)

    algo.run_experiments()


def get_tester_curriculum(world, experiment, args):
    from src.tester.tester import Tester
    from src.tester.params import Params
    from src.common.curriculum import MultiTaskCurriculumLearner
    project_path = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(project_path, "params", world, args.algorithm + ".yaml")
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)

    tester = Tester(params, experiment, args)

    # Setting the curriculum learner
    curriculum = MultiTaskCurriculumLearner(tester.tasks)
    learning_params = Params(params['learning_params'])
    curriculum.total_steps = learning_params.step_unit * learning_params.total_units
    return tester, curriculum, params


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # EXAMPLE: python run.py --algorithm "qrm" --world "office" --seeds 0 1 2 3 4 --use_wandb

    # Getting params
    worlds = ["office", "craft", "water"]

    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='qrm', type=str,
                        help='This parameter indicated which RL algorithm to use.')
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0],
                        help='A list of random seeds to run experiments.')
    parser.add_argument('--use_rs', action='store_true',
                        help='Whether to use reward shaping of reward machines.')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to show experimental results on wandb or not.')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Whether to use cuda or not.')
    parser.add_argument('--load_rm_mode', default='files',
                        help='files: load rm from files; formulas: automatically generate rm by ltl progression.')
    parser.add_argument('--notes', default='', type=str,
                        help='Notes on the algorithm, shown in wandb.')

    # add learning params in command lines for tests
    parser.add_argument('--set_params', action='store_true',
                        help='Whether to set learning parameters in command lines.')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--buffer_size', default=1000, type=int, help='buffer size')
    parser.add_argument('--batch_size', default=1000, type=int, help='(mini) batch size')
    parser.add_argument('--n_updates', default=10, type=int, help='updated times for ppo')
    parser.add_argument('--e_coef', default=0, type=float, help='entropy loss coef')

    args = parser.parse_args()
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not (0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")

    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    use_wandb = args.use_wandb

    filename = "office.txt" if world == "office" else "%s_%d.txt" % (world, map_id)
    project_path = os.path.dirname(__file__)
    experiment = os.path.join(project_path, "..", "experiments", world, "tests", filename)

    for seed in args.seeds:
        tester, curriculum, params = get_tester_curriculum(world, experiment, args)
        learning_params = tester.learning_params
        # saver = Saver(alg_name, tester, curriculum)
        if use_wandb:
            wandb_config = copy.deepcopy(params)
            wandb_config['use_cuda'] = args.use_cuda
            wandb_config['seed'] = seed
            print(wandb_config)
            wandb.init(
                project="zxj-LLM-Research",
                notes=args.notes,
                group=world,
                name=alg_name,
                config=wandb_config
            )
        print("*" * 10, "seed:", seed, "*" * 10)
        print("alg_name:", alg_name)
        print(params)
        time_init = time.time()
        setup_seed(seed)
        run_experiment(args, tester, curriculum)
        print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")
        # saver.save_results(filename="seed%d.npy" % s)
        if use_wandb: wandb.finish()
