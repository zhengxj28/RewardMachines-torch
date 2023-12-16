import time
import wandb
from src.algos.base_algo import BaseAlgo
from src.worlds.game import Game


class NonMDPAlgo(BaseAlgo):
    def __init__(self, tester, curriculum):
        super().__init__(tester, curriculum)
        self.loss_info = {}


    def create_env(self, task):
        tester = self.tester
        reward_machines = tester.get_reward_machines()
        task_rm_id = tester.get_reward_machine_id_from_file(task)
        task_params = tester.get_task_params(task)

        rm = reward_machines[task_rm_id]
        env = Game(task_params, rm)

        return env

    def train_episode(self, *args):
        pass

    def evaluate(self, step):
        tester = self.tester
        t_init = time.time()
        reward_machines = tester.get_reward_machines()
        rewards_of_each_task = []

        for rm_file in tester.get_rm_files():
            reward = self.evaluate_episode(rm_file)
            rewards_of_each_task.append(reward)

        total_reward = sum(rewards_of_each_task)
        tester.total_rewards.append(total_reward)

        cur_time = time.time()
        training_time = (cur_time - tester.last_test_time) / 60
        print("Training time: %0.2f minutes." % (training_time))
        tester.last_test_time = cur_time

        print("Steps: %d\tTesting: %0.1f" % (step, cur_time - t_init),
              "seconds\tTotal Reward: %0.1f" % total_reward)
        print("\t".join(["%0.1f" % (r) for r in rewards_of_each_task]))

        log_reward = {"total": sum(rewards_of_each_task)}
        for i in range(len(rewards_of_each_task)):
            log_reward["task%d" % i] = rewards_of_each_task[i]

        for key, value in self.loss_info.items():
            print("%s: %.4f" % (key, value), end='\t')
        print('\n')

        if tester.use_wandb:
            wandb.log({"reward": log_reward, "loss": self.loss_info, "training_time": training_time})

