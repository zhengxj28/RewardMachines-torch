from src.tester.tester_craft import TesterCraftWorld
from src.tester.tester_office import TesterOfficeWorld
from src.tester.tester_water import TesterWaterWorld
from src.tester.tester_mujoco import TesterMujoco
from src.reward_machines.reward_machine import RewardMachine
from src.reward_machines.stochastic_reward_machine import StochasticRewardMachine
from src.tester.test_utils import read_json, get_precentiles_str, get_precentiles_in_seconds, reward2steps
import numpy as np
import time, os
from src.tester.params import Params
from src.worlds.reward_machines_env import RewardMachinesEnv

class Tester:
    def __init__(self, params, experiment_file, args):
        self.params = params
        learning_params = Params(params.get('learning_params', dict()))
        testing_params = Params(params.get('testing_params', dict()))
        model_params = Params(params.get('model_params', dict()))
        self.learning_params = learning_params
        self.testing_params = testing_params
        self.model_params = model_params
        if args.value_com:
            self.learning_params.value_com = [args.value_com for _ in range(3)]
        # Reading the file
        self.args = args
        self.experiment = experiment_file
        self.use_wandb = args.use_wandb
        self.label_noise = args.label_noise
        self.last_test_time = time.time()  # last testing time

        self.save_model_name = args.save_model_name
        self.load_model_name = args.load_model_name

        f = open(experiment_file)
        lines = [l.rstrip() for l in f]
        f.close()

        # setting the right world environment
        self.game_type = eval(lines[0])
        if self.game_type == "officeworld":
            self.world = TesterOfficeWorld(experiment_file, learning_params.gamma)
        elif self.game_type == "craftworld":
            self.world = TesterCraftWorld(experiment_file, learning_params.tabular_case, learning_params.gamma)
        elif self.game_type == "waterworld":
            self.world = TesterWaterWorld(experiment_file, learning_params.use_random_maps)
        else:
            self.world = TesterMujoco(experiment_file)

        # Creating the reward machines for each task
        # Although some algorithms do not use RM, we create RMs to generate/reshape env rewards.
        self.reward_machines = []
        # a task is a file path of rm or ltl_formula (load_rm_mode=='files' or 'formulas')
        self.tasks = []
        self.task2rm_id = {}

        if args.load_rm_mode == 'files':
            self.tasks = self.world.rm_files
            for i, rm_file in enumerate(self.tasks):
                self.task2rm_id[rm_file] = i
                rm_file = os.path.join(os.path.dirname(__file__), "..", rm_file)
                self.reward_machines.append(RewardMachine(rm_file, args.use_rs, learning_params.gamma, False))
        elif args.load_rm_mode == 'formulas':
            self.tasks = self.world.ltl_files
            self.ltl_formulas = []
            for i, ltl_file in enumerate(self.tasks):
                self.task2rm_id[ltl_file] = i
                ltl_file = os.path.join(os.path.dirname(__file__), "..", ltl_file)
                with open(ltl_file) as f:
                    lines = [l.rstrip() for l in f]
                    self.ltl_formulas.append(eval(lines[1]))
                self.reward_machines.append(RewardMachine(ltl_file, args.use_rs, learning_params.gamma, True))
        elif args.load_rm_mode == 'srm':
            self.tasks = self.world.srm_files
            for i, srm_file in enumerate(self.tasks):
                self.task2rm_id[srm_file] = i
                self.reward_machines.append(StochasticRewardMachine(srm_file))
        else:
            raise NotImplementedError('Unexpected load_rm_mode:' + args.load_rm_mode)

        self.rm_id2task = dict([(rm_id, task) for task, rm_id in self.task2rm_id.items()])

        self.total_rewards = []

        env_aux = RewardMachinesEnv(self, self.tasks[0])
        self.num_features = env_aux.num_features
        self.num_actions = env_aux.num_actions
        self.action_space = env_aux.action_space

    def get_task_params(self, task_specification):
        return self.world.get_task_params(task_specification)

    def get_reward_machine_id_from_file(self, rm_file):
        return self.task2rm_id[rm_file]

    def get_all_task_files(self):
        return self.tasks

    def get_reward_machines(self):
        return self.reward_machines
