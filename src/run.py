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
    elif alg_name == "qsrm":
        from src.algos.qsrm import QSRMAlgo
        algo = QSRMAlgo(tester, curriculum, show_print, use_cuda, args.label_noise)
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
    elif alg_name == "sac":
        from src.algos.sac import SACAlgo
        algo = SACAlgo(tester, curriculum, show_print, use_cuda)
    elif alg_name == "sacrm":
        from src.algos.sacrm import SACRMAlgo
        algo = SACRMAlgo(tester, curriculum, show_print, use_cuda)
    elif alg_name in ["lifelong_qrm", "distill", "lsrm"]:
        from src.lifelong.lifelong_qrm import LifelongQRMAlgo
        algo = LifelongQRMAlgo(tester, curriculum, show_print, use_cuda)
    else:
        raise NotImplementedError("Algorithm:" + alg_name)

    algo.run_experiments()


def get_tester_curriculum(world, experiment, args):
    from src.tester.tester import Tester
    from src.tester.params import Params
    from src.common.curriculum import MultiTaskCurriculumLearner
    from src.common.curriculum import LifelongCurriculumLearner
    project_path = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(project_path, "params", world, args.algorithm + ".yaml")
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)

    # if args.value_com in ['average', 'max', 'left', 'right']:
    #     params['learning_params']['transfer_methods'] = "value_com"
    #     params['learning_params']['value_com'] = [args.value_com for _ in range(3)]
    # elif args.value_com == 'none':
    #     params['learning_params']['transfer_methods'] = "none"
    # elif args.value_com:
    #     raise ValueError(f"Unexpected value_com in commands: {args.value_com}")

    tester = Tester(params, experiment, args)
    learning_params = Params(params['learning_params'])
    total_steps = learning_params.step_unit * learning_params.total_units

    # Setting the curriculum learner
    if args.algorithm in ["lifelong_qrm", "distill", "lsrm"]:
        curriculum = LifelongCurriculumLearner(tester.tasks,
                                               tester.world.lifelong_curriculum,
                                               total_steps=total_steps
                                               )
    else:
        curriculum = MultiTaskCurriculumLearner(tester.tasks,
                                                total_steps=total_steps)

    return tester, curriculum, params


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # EXAMPLE: python run.py --algorithm "qrm" --world "office" --seeds 0 1 2 3 4 --use_wandb

    # Getting params
    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='qrm', type=str,
                        help='This parameter indicated which RL algorithm to use.')
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which world to solve.')
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
    parser.add_argument('--project', default='Default Project', type=str,
                        help='Project name in the wandb.')

    parser.add_argument('--label_noise', default=0.0, type=float,
                        help='Noise of the labelling function, between 0 and 1. For stochastic RM experiments only.')

    parser.add_argument('--wandb_name', default='', type=str,
                        help='The experiment name shown in wandb.')
    parser.add_argument('--save_model_name', default='', type=str,
                        help='The name of the saved model.')
    parser.add_argument('--load_model_name', default='', type=str,
                        help='The name of the loaded model.')
    parser.add_argument('--value_com', default='', type=str,
                        help='Value composition methods for lifelong learning.')

    args = parser.parse_args()
    if not (0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")
    assert 0 <= args.label_noise <= 1

    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    use_wandb = args.use_wandb

    filename = "%s_%d.txt" % (world, map_id)
    project_path = os.path.dirname(__file__)
    experiment = os.path.join(project_path, "..", "experiments", world, "tests", filename)

    tester, curriculum, params = get_tester_curriculum(world, experiment, args)
    for seed in args.seeds:
        learning_params = tester.learning_params
        # saver = Saver(alg_name, tester, curriculum)
        if use_wandb:
            wandb_config = copy.deepcopy(params)
            wandb_config['map'] = args.map
            wandb_config['use_cuda'] = args.use_cuda
            wandb_config['use_rs'] = args.use_rs
            wandb_config['seed'] = seed
            wandb_config['label_noise'] = args.label_noise
            wandb_config['load_rm_mode'] = args.load_rm_mode
            print(wandb_config)
            wandb.init(
                project=args.project,
                notes=args.notes,
                group=world,
                name=alg_name if args.wandb_name=="" else args.wandb_name,
                config=wandb_config
            )
        print("*" * 10, "Task Files", "*" * 10)
        for task in tester.tasks:
            print(task)
        print("*" * 10, "seed:", seed, "*" * 10)
        print("*" * 10, "label_noise:", args.label_noise, "*" * 10)
        print("alg_name:", alg_name)
        print(params)
        time_init = time.time()
        setup_seed(seed)
        run_experiment(args, tester, curriculum)
        print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")
        # saver.save_results(filename="seed%d.npy" % s)
        if use_wandb: wandb.finish()
