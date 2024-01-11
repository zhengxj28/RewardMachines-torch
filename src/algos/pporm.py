import random
import time
import wandb

from src.algos.ppo import PPOAlgo
from src.agents.pporm_agent import PPORMAgent


class PPORMAlgo(PPOAlgo):
    def __init__(self, tester, curriculum, show_print, use_cuda):
        super().__init__(tester, curriculum, show_print, use_cuda)

        self.show_print = show_print
        self.use_cuda = use_cuda

        num_features = tester.num_features
        num_actions = tester.num_actions

        learning_params = tester.learning_params
        model_params = tester.model_params
        self.agent = PPORMAgent(num_features, num_actions,
                                learning_params,
                                model_params,
                                tester.get_reward_machines(),
                                tester.task2rm_id,
                                use_cuda,
                                curriculum)

    def train_episode(self, task):
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
            s2, env_reward, done, info = env.step(a)

            agent.update(s1, a, s2, info, log_prob, done)

            # Learning
            cur_step = curriculum.get_current_step()
            exit_episode = done or t >= num_steps-1 or curriculum.stop_learning()
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
