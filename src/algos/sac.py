import random
import time
import wandb

from src.algos.base_algo import BaseAlgo
from src.agents.sac_agent import SACAgent


class SACAlgo(BaseAlgo):
    def __init__(self, tester, curriculum, show_print, use_cuda):
        super().__init__(tester, curriculum)

        self.show_print = show_print
        self.use_cuda = use_cuda
        self.loss_info = {}

        num_features = tester.num_features
        num_actions = tester.num_actions
        action_space = tester.action_space

        learning_params = tester.learning_params
        model_params = tester.model_params
        self.agent = SACAgent(num_features,
                              num_actions,
                              action_space,
                              learning_params,
                              model_params,
                              use_cuda)

    def train_episode(self, task):
        tester = self.tester
        curriculum = self.curriculum
        agent = self.agent
        show_print = self.show_print

        # Initializing parameters and the game
        learning_params = tester.learning_params
        testing_params = tester.testing_params

        training_reward = 0
        # Getting the initial state of the environment and the reward machine
        env = self.create_env(task)
        s1 = env.reset()
        self.agent.reset_status(task, False)

        # Starting interaction with the environment
        num_steps = testing_params.num_steps
        for t in range(num_steps):
            curriculum.add_step()
            cur_step = curriculum.get_current_step()
            a = agent.get_action(s1, choose_randomly=(cur_step < learning_params.learning_starts))
            s2, env_reward, done, _ = env.step(a)

            agent.update(s1, a, s2, env_reward, done)

            # Learning
            if cur_step >= learning_params.learning_starts:
                if cur_step % learning_params.train_freq == 0:
                    self.loss_info = agent.learn()
                # Updating the target network
                if cur_step % learning_params.target_network_update_freq == 0:
                    agent.update_target_network()

            # Printing
            training_reward += env_reward

            # Testing
            if testing_params.test and cur_step % testing_params.test_freq == 0:
                self.evaluate(cur_step)

            # Restarting the environment (Game Over)
            if done or curriculum.stop_learning():
                break
            # Moving to the next state
            s1 = s2
