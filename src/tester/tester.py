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
        # Reading the file
        self.experiment = experiment_file
        self.use_wandb = args.use_wandb
        self.label_noise = args.label_noise
        self.last_test_time = time.time()  # last testing time
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

    # def get_world_name(self):
    #     return self.game_type.replace("world", "")

    def get_task_params(self, task_specification):
        return self.world.get_task_params(task_specification)

    def get_reward_machine_id_from_file(self, rm_file):
        return self.task2rm_id[rm_file]

    # def get_file_from_reward_machine_id(self, rm_id):
    #     return self.rm_id2task[rm_id]

    def get_rm_files(self):
        return self.task2rm_id.keys()

    # def get_reward_machine_id(self, task_specification):
    #     rm_file = self.world.get_task_rm_file(task_specification)
    #     return self.get_reward_machine_id_from_file(rm_file)

    def get_reward_machines(self):
        return self.reward_machines

    # def get_world_dictionary(self):
    #     return self.world.get_dictionary()

    # def get_task_specifications(self):
    #     # Returns the list with the task specifications (reward machine + env params)
    #     return self.world.get_task_specifications()

    # def get_tasks(self):
    #     # Returns only the reward machines that we are learning
    #     return self.world.get_reward_machine_files()

    # def get_optimal(self, task_specification):
    #     r = self.world.optimal[task_specification]
    #     return r if r > 0 else 1.0

    # def run_test(self, step, test_function, *test_args):
    #     t_init = time.time()
    #     # 'test_function' parameters should be (sess, task_params, learning_params, testing_params, *test_args)
    #     # and returns the reward
    #     reward_machines = self.get_reward_machines()
    #     rewards_of_each_task = []
    #     for task_specification in self.get_task_specifications():
    #         task_str = str(task_specification)
    #         task_params = self.get_task_params(task_specification)
    #         task_rm_id = self.get_reward_machine_id(task_specification)
    #         reward = test_function(reward_machines, task_params, task_rm_id, self.testing_params, *test_args)
    #         if step not in self.rewards[task_str]:
    #             self.rewards[task_str][step] = []
    #         if len(self.steps) == 0 or self.steps[-1] < step:
    #             self.steps.append(step)
    #         self.rewards[task_str][step].append(reward)
    #         rewards_of_each_task.append(reward)
    #
    #     total_reward = sum(rewards_of_each_task)
    #     self.total_rewards.append(total_reward)
    #
    #     cur_time = time.time()
    #     training_time = (cur_time-self.last_test_time)/60
    #     print("Training time: %0.2f minutes."%(training_time))
    #     self.last_test_time = cur_time
    #
    #     print("Steps: %d\tTesting: %0.1f" % (step, cur_time - t_init),
    #           "seconds\tTotal Reward: %0.1f" % total_reward)
    #     print("\t".join(["%0.1f" % (r) for r in rewards_of_each_task]))
    #
    #     log_reward = {"total": sum(rewards_of_each_task)}
    #     for i in range(len(rewards_of_each_task)):
    #         log_reward["task%d"%i] = rewards_of_each_task[i]
    #
    #     for key, value in self.loss_info.items():
    #         print("%s: %.4f"%(key, value), end='\t')
    #     print('\n')
    #
    #     if self.use_wandb:
    #         wandb.log({"reward": log_reward, "loss": self.loss_info, "training_time": training_time})

    # def save_loss_info(self, loss_info):
    #     self.loss_info = loss_info

    # def show_results(self):
    #     average_reward = {}
    #
    #     tasks = self.get_task_specifications()
    #
    #     # Showing perfomance per task
    #     for t in tasks:
    #         t_str = str(t)
    #         print("\n" + t_str + " --------------------")
    #         print("steps\tP25\t\tP50\t\tP75")
    #         for s in self.steps:
    #             normalized_rewards = [r / self.get_optimal(t) for r in self.rewards[t_str][s]]
    #             a = np.array(normalized_rewards)
    #             if s not in average_reward:
    #                 average_reward[s] = a
    #             else:
    #                 average_reward[s] = a + average_reward[s]
    #             p25, p50, p75 = get_precentiles_str(a)
    #             # p25, p50, p75 = get_precentiles_in_seconds(a)
    #             print(str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)
    #
    #     # Showing average perfomance across all the task
    #     print("\nAverage Reward --------------------")
    #     print("steps\tP25\t\tP50\t\tP75")
    #     num_tasks = float(len(tasks))
    #     for s in self.steps:
    #         normalized_rewards = average_reward[s] / num_tasks
    #         p25, p50, p75 = get_precentiles_str(normalized_rewards)
    #         # p25, p50, p75 = get_precentiles_in_seconds(normalized_rewards)
    #         print(str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)

    # def get_best_performance_per_task(self):
    #     # returns the best performance per task (this is relevant for reward normalization)
    #     ret = {}
    #     for t in self.get_task_specifications():
    #         t_str = str(t)
    #         ret[t_str] = max([max(self.rewards[t_str][s]) for s in self.steps])
    #     return ret

    # def get_result_summary(self):
    #     """
    #     Returns normalized average performance across all the tasks
    #     """
    #     average_reward = {}
    #     task_reward = {}
    #     task_reward_count = {}
    #     tasks = self.get_task_specifications()
    #
    #     # Computing average reward per task
    #     for task_specification in tasks:
    #         t_str = str(task_specification)
    #         task_rm = self.world.get_task_rm_file(task_specification)
    #         if task_rm not in task_reward:
    #             task_reward[task_rm] = {}
    #             task_reward_count[task_rm] = 0
    #         task_reward_count[task_rm] += 1
    #         for s in self.steps:
    #             normalized_rewards = [r / self.get_optimal(t_str) for r in self.rewards[t_str][s]]
    #             a = np.array(normalized_rewards)
    #             # adding to the average reward
    #             if s not in average_reward:
    #                 average_reward[s] = a
    #             else:
    #                 average_reward[s] = a + average_reward[s]
    #             # adding to the average reward per tas
    #             if s not in task_reward[task_rm]:
    #                 task_reward[task_rm][s] = a
    #             else:
    #                 task_reward[task_rm][s] = a + task_reward[task_rm][s]
    #
    #     # Computing average reward across all tasks
    #     ret = []
    #     ret_task = {}
    #     for task_rm in task_reward:
    #         ret_task[task_rm] = []
    #     num_tasks = float(len(tasks))
    #     for s in self.steps:
    #         normalized_rewards = average_reward[s] / num_tasks
    #         ret.append([s, normalized_rewards])
    #         for task_rm in task_reward:
    #             normalized_task_rewards = task_reward[task_rm][s] / float(task_reward_count[task_rm])
    #             ret_task[task_rm].append([s, normalized_task_rewards])
    #     ret_task["all"] = ret
    #
    #     return ret_task
