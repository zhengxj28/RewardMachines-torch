import random
import time
import wandb

from src.algos.nmdp_algo import NonMDPAlgo
from src.lifelong.lifelong_algo import LifelongAlgo
from src.lifelong.lifelong_qrm_agent import LifelongQRMAgent


class LifelongQRMAlgo(NonMDPAlgo, LifelongAlgo):
    def __init__(self, tester, curriculum, show_print, use_cuda):
        LifelongAlgo.__init__(self, tester, curriculum)

        self.show_print = show_print
        self.use_cuda = use_cuda
        self.loss_info = {}

        num_features = tester.num_features
        num_actions = tester.num_actions

        learning_params = tester.learning_params
        model_params = tester.model_params
        self.agent = LifelongQRMAgent(num_features, num_actions,
                                      learning_params,
                                      model_params,
                                      tester.get_reward_machines(),
                                      tester.task2rm_id,
                                      use_cuda,
                                      curriculum.lifelong_curriculum)

    def train_episode(self, task):
        # self.agent.current_phase = self.curriculum.current_phase
        # # activate networks of current phase and freeze other networks to simulate "lifelong learning"
        # self.agent.activate_and_freeze_networks()

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
            a = agent.get_action(s1)
            # do not use reward from env to learn
            s2, env_reward, done, info = env.step(a)

            agent.update(s1, a, s2, info, done)

            # Learning
            cur_step = curriculum.get_current_step()
            if cur_step >= learning_params.learning_starts:
                if cur_step % learning_params.train_freq == 0:
                    self.loss_info = agent.learn()
                # Updating the target network
                if cur_step % learning_params.target_network_update_freq == 0:
                    agent.update_target_network()

            # Printing
            training_reward += env_reward
            # if show_print and (t + 1) % learning_params.print_freq == 0:
            #     print("Step:", t + 1, "\tTotal reward:", training_reward)

            # Testing
            if testing_params.test and cur_step % testing_params.test_freq == 0:
                self.evaluate(cur_step)

            # Restarting the environment (Game Over)
            if done or curriculum.stop_learning() or curriculum.stop_curriculum():
                break

            # Moving to the next state
            s1 = s2

    def evaluate_episode(self, task):
        return NonMDPAlgo.evaluate_episode(self, task)

    def start_new_phase(self):
        self.transfer_knowledge()
        self.current_phase = self.curriculum.current_phase
        # activate networks of current phase and freeze other networks to simulate "lifelong learning"
        self.activate_and_freeze_networks()
        # self.agent.buffer.clear()

    def activate_and_freeze_networks(self):
        # note that a policy may considered in different phase, due to "equivalent states" of RM
        # must freeze networks of other phases first, then activate networks of current phase
        # for phase, policies in enumerate(self.agent.policies_of_each_phase):
        #     if phase == self.agent.current_phase: continue
        #     self.agent.qrm_net.freeze(policies)
        activate_policies = self.agent.policies_of_each_phase[self.current_phase]
        freeze_policies = set([i for i in range(self.agent.num_policies)])-activate_policies
        self.agent.qrm_net.freeze(freeze_policies)
        self.agent.qrm_net.activate(activate_policies)

    def transfer_knowledge(self):
        self.agent.transfer_knowledge()
