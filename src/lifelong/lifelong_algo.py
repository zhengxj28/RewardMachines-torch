from abc import ABC, abstractmethod
from src.algos.base_algo import BaseAlgo
import wandb
import time


class LifelongAlgo(BaseAlgo, ABC):
    def __init__(self, tester, curriculum, *args):
        super().__init__(tester, curriculum, args)
        self.current_phase = curriculum.current_phase

    def run_experiments(self):
        curriculum = self.curriculum
        self.start_new_phase()
        while not curriculum.stop_learning():
            phase_before = curriculum.current_phase
            task = curriculum.get_next_task()
            phase_after = curriculum.current_phase
            if phase_before != phase_after:
                self.start_new_phase()
            self.train_episode(task)

    @abstractmethod
    def start_new_phase(self):
        pass

    @abstractmethod
    def transfer_knowledge(self):
        pass

    @abstractmethod
    def train_episode(self, *args):
        pass

    def evaluate(self, step):
        tester = self.tester
        t_init = time.time()
        reward_machines = tester.get_reward_machines()
        rewards_of_each_task = []
        testing_steps_of_each_task = []

        for task_id in self.curriculum.current_curriculum:
            task = self.tester.tasks[task_id]
            reward, testing_steps = self.evaluate_episode(task)
            rewards_of_each_task.append(reward)
            testing_steps_of_each_task.append(testing_steps)
        print("Current curriculum (task id):", self.curriculum.current_curriculum)

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

        phase = self.curriculum.current_phase
        log_reward = {"phase%d" % phase: sum(rewards_of_each_task)}
        log_steps = {"phase%d" % phase: sum(testing_steps_of_each_task)}
        log_avg_reward = {"phase%d" % phase: sum(rewards_of_each_task) / sum(testing_steps_of_each_task)}
        for i, task_id in enumerate(self.curriculum.current_curriculum):
            log_reward["ph%dtask%d" % (phase, task_id)] = rewards_of_each_task[i]
            log_steps["ph%dtask%d" % (phase, task_id)] = testing_steps_of_each_task[i]
            log_avg_reward["ph%dtask%d" % (phase, task_id)] = rewards_of_each_task[i] / testing_steps_of_each_task[i]

        for key, value in self.loss_info.items():
            print("%s: %.4f" % (key, value), end='\t')
        print('\n')

        if tester.use_wandb:
            wandb.log({"reward": log_reward,
                       "testing steps": log_steps,
                       "avg_reward": log_avg_reward,
                       "loss": self.loss_info,
                       "training_time": training_time})
