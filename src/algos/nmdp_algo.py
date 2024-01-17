import time
import wandb
from src.algos.base_algo import BaseAlgo


class NonMDPAlgo(BaseAlgo):
    def __init__(self, tester, curriculum):
        super().__init__(tester, curriculum)

    def train_episode(self, *args):
        pass

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
            # update agent status to cope with non-MDP
            self.agent.update(s1, a, s2, infos, done, True)
            r_total += env_reward
            # Restarting the environment (Game Over)
            if done:
                break
            # Moving to the next state
            s1 = s2

        return r_total, testing_steps
