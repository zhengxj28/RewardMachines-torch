import argparse, os, sys

file_path = os.path.dirname(os.path.abspath(__file__))  # location of current file
# sys.path.append(file_path)
# sys.path.append(os.path.join(file_path, "worlds"))
sys.path.append(os.path.join(file_path, ".."))  # abs project path

from src.algos.qrm import run_qrm_experiments
from get_params import *

# The pickle library is asking me to have access to Ball and BallAgent from the main...
from src.worlds.water_world import Ball, BallAgent


def run_experiment(world, alg_name, num_times, use_rs, show_print):
    if world == "office":
        experiment = os.path.join("experiments", "office", "tests", "office.txt")
    else:
        experiment = os.path.join("experiments", world, "tests", "%s_%d.txt" % (world, map_id))
    project_path = os.path.dirname(__file__)
    experiment = os.path.join(project_path, "..", experiment)
    world += "world"

    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment, "num_times: " + str(num_times),
          show_print)

    if world == 'officeworld':
        tester, curriculum = get_params_office_world(experiment, use_rs)
    if world == 'craftworld':
        tester, curriculum = get_params_craft_world(experiment, use_rs)
    if world == 'waterworld':
        tester, curriculum = get_params_water_world(experiment, use_rs)

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
        run_qrm_experiments(alg_name, tester, curriculum, num_times, show_print)


if __name__ == "__main__":
    # EXAMPLE: python3 run.py --algorithm="qrm" --world="craft" --map=0 --num_times=1

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
    parser.add_argument('--num_times', default=1, type=int,
                        help='This parameter indicated which map to use. It must be a number greater or equal to 1')
    parser.add_argument("--verbosity", help="increase output verbosity")

    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError(
        "Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not (0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")
    if args.num_times < 1: raise NotImplementedError("num_times must be greater than 0")

    # Running the experiment
    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    num_times = args.num_times
    show_print = args.verbosity is not None
    # show_print = True
    use_rs = alg_name.endswith("-rs")
    run_experiment(world, alg_name, num_times, use_rs, show_print)
