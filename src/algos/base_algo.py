from abc import ABC, abstractmethod
from src.agents.base_rl_agent import DefaultAgent
from src.worlds.base_env import DefaultEnv
from src.worlds.reward_machines_env import RewardMachinesEnv

class BaseAlgo(ABC):
    def __init__(self, tester, curriculum, *args):
        self.tester = tester
        self.curriculum = curriculum
        # Resetting default values
        curriculum.restart()
        self.agent = DefaultAgent()

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

    def evaluate(self, *args):
        return self.evaluate_episode(None)

    def evaluate_episode(self, task):
        env = self.create_env(task)
        s1 = env.reset()
        self.agent.reset_status(task, True)

        # Starting interaction with the environment
        r_total = 0
        for t in range(self.tester.testing_params.num_steps):
            a = self.agent.get_action(s1, True)
            s2, env_reward, done, infos = env.step(a)
            self.agent.update(s1, a, s2, infos, done, True)
            r_total += env_reward
            # Restarting the environment (Game Over)
            if done:
                break
            # Moving to the next state
            s1 = s2

        return r_total
