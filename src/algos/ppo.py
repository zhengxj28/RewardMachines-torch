import random
import time
import wandb

from src.algos.nmdp_algo import NonMDPAlgo
from src.agents.ppo_agent import PPOAgent


class PPOAlgo(NonMDPAlgo):
    def __init__(self, tester, curriculum, show_print, use_cuda):
        super().__init__(tester, curriculum)

        self.show_print = show_print
        self.use_cuda = use_cuda
        self.loss_info = {}

        num_features = tester.num_features
        num_actions = tester.num_actions

        learning_params = tester.learning_params
        model_params = tester.model_params
        self.agent = PPOAgent(num_features, num_actions,
                              learning_params,
                              model_params,
                              use_cuda,
                              curriculum)

    def train_episode(self, task):
        """
        This code runs one training episode.
            - rm_file: It is the path towards the RM machine to solve on this episode
        """
        tester = self.tester
        curriculum = self.curriculum
        agent = self.agent

        # Initializing parameters and the game
        learning_params = tester.learning_params
        testing_params = tester.testing_params

        # Getting the initial state of the environment and the reward machine
        env = self.create_env(task)
        s1 = env.reset()
        self.agent.reset_status(task, False)

        # Starting interaction with the environment
        num_steps = testing_params.num_steps
        t = 0
        while t < num_steps:
            curriculum.add_step()
            a, log_prob = agent.get_action(s1)
            # do not use reward from env to learn
            s2, env_reward, done, _ = env.step(a)

            agent.update(s1, a, s2, env_reward, log_prob, done)

            # Learning
            cur_step = curriculum.get_current_step()
            exit_episode = done or t >= num_steps or curriculum.stop_learning()
            if exit_episode:
                self.loss_info = agent.learn()
                agent.buffer.clear()  # Once learned, clear the data

            # Testing
            if testing_params.test and cur_step % testing_params.test_freq == 0:
                self.evaluate(cur_step)

            # Restarting the environment (Game Over)
            if exit_episode:
                break

            # Moving to the next state
            s1 = s2
            t += 1

        # truncated caused by max episode steps
        if t >= num_steps:
            self.loss_info = agent.learn()
            agent.buffer.clear()  # Once learned, clear the data

    def evaluate_episode(self, task):
        env = self.create_env(task)
        s1 = env.reset()
        self.agent.reset_status(task, True)

        # Starting interaction with the environment
        r_total = 0
        for t in range(self.tester.testing_params.num_steps):
            a = self.agent.get_action(s1, True)
            s2, env_reward, done, infos = env.step(a)
            # ppo do not need to update agent while evaluating
            # self.agent.update(s1, a, s2, infos, done, True)
            r_total += env_reward
            # Restarting the environment (Game Over)
            if done:
                break
            # Moving to the next state
            s1 = s2

        return r_total
