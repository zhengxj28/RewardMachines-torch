from abc import ABC, abstractmethod
from src.agents.base_rl_agent import DefaultAgent
from src.worlds.base_env import DefaultEnv
from src.worlds.reward_machines_env import RewardMachinesEnv
import wandb
import time


class BaseAlgo(ABC):
    def __init__(self, tester, curriculum, *args):
        self.tester = tester
        self.curriculum = curriculum
        # Resetting default values
        curriculum.restart()
        self.agent = DefaultAgent()
        self.loss_info = {}

    def run_experiments(self):
        curriculum = self.curriculum
        while not curriculum.stop_learning():
            task = curriculum.get_next_task()
            self.train_episode(task)

    def create_env(self, task):
        env = RewardMachinesEnv(self.tester, task)
        return env

    @abstractmethod
    def train_episode(self, *args):
        pass

    def evaluate(self, step):
        tester = self.tester
        t_init = time.time()
        reward_machines = tester.get_reward_machines()
        rewards_of_each_task = []
        testing_steps_of_each_task = []

        for task_file in tester.get_all_task_files():
            reward, testing_steps = self.evaluate_episode(task_file)
            rewards_of_each_task.append(reward)
            testing_steps_of_each_task.append(testing_steps)

        total_reward = sum(rewards_of_each_task)
        tester.total_rewards.append(total_reward)

        cur_time = time.time()
        training_time = (cur_time - tester.last_test_time) / 60
        print("Training time: %0.2f minutes." % (training_time))
        tester.last_test_time = cur_time

        print("Current steps: %d\tTesting: %0.1f" % (step, cur_time - t_init),
              "seconds\tTotal Reward: %0.1f" % total_reward)
        print("Rewards:      \t", "\t".join(["%0.1f" % (r) for r in rewards_of_each_task]))
        print("Testing Steps:\t", "\t".join(["%d" % (steps) for steps in testing_steps_of_each_task]))

        log_reward = {"total": sum(rewards_of_each_task)}
        log_steps = {"total": sum(testing_steps_of_each_task)}
        log_avg_reward = {"total": sum(rewards_of_each_task) / sum(testing_steps_of_each_task)}
        for i in range(len(rewards_of_each_task)):
            log_reward["task%d" % i] = rewards_of_each_task[i]
            log_steps["task%d" % i] = testing_steps_of_each_task[i]
            log_avg_reward["task%d" % i] = rewards_of_each_task[i] / testing_steps_of_each_task[i]

        for key, value in self.loss_info.items():
            print("%s: %.4f" % (key, value), end='\t')
        print('\n')

        if tester.use_wandb:
            wandb.log({"reward": log_reward,
                       "testing steps": log_steps,
                       "avg_reward": log_avg_reward,
                       "loss": self.loss_info,
                       "training_time": training_time})

    def evaluate_episode(self, task):
        env = self.create_env(task)
        s1 = env.reset()
        self.agent.reset_status(task, True)

        # Starting interaction with the environment
        r_total = 0
        testing_steps = 0
        for t in range(self.tester.testing_params.num_steps):
            testing_steps += 1
            a = self.agent.get_action(s1, True)
            s2, env_reward, done, infos = env.step(a)
            r_total += env_reward
            # Restarting the environment (Game Over)
            if done:
                break
            # Moving to the next state
            s1 = s2

        return r_total, testing_steps
