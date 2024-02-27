import random
import time
import wandb

from src.algos.nmdp_algo import BaseAlgo
from src.agents.promptprog_agent import PromptProgAgent
from src.worlds.game_llm import LLMGame


class PromptProgAlgo(BaseAlgo):
    def __init__(self, tester, curriculum, show_print, use_cuda):
        super().__init__(tester, curriculum)
        # TODO: remove reward machine in tester & curriculum

        self.show_print = show_print
        self.use_cuda = use_cuda
        self.loss_info = {}

        num_features = tester.num_features
        num_actions = tester.num_actions
        
        learning_params = tester.learning_params
        model_params = tester.model_params
        self.agent = PromptProgAgent(num_features, num_actions,
                                     learning_params,
                                     model_params,
                                     tester.get_reward_machines(),
                                     tester.task2rm_id,
                                     use_cuda)

    def create_env(self, task):
        env = LLMGame(self.tester.get_task_params(task))
        return env

    def train_episode(self, task):
        """
        This code runs one training episode.
            - rm_file: It is the path towards the RM machine to solve on this episode
        """ 
        tester = self.tester
        curriculum = self.curriculum
        agent = self.agent
        show_print = self.show_print

        nl_file = tester.world.nl_files[curriculum.current_task]
        with open(nl_file, 'r') as file:
            nl_task = file.readline().replace("\n", "")
            
        # Initializing parameters and the game
        learning_params = tester.learning_params
        testing_params = tester.testing_params

        training_reward = 0
        # Getting the initial state of the environment and the reward machine
        env = self.create_env(task)
        s1 = env.reset()
        # self.agent.reset_status(task, False)

        # task = (reward_machine, map)
        # Starting interaction with the environment
        num_steps = testing_params.num_steps
        for t in range(num_steps):
            curriculum.add_step()
            s1_ = agent.add_task_embedding(s1, nl_task)
            a = agent.get_action(s1_)
            # do not use reward from env to learn
            s2, env_reward, done, events = env.step(a) # done defaults 0

            nl_task = agent.progression(nl_task, events)
            done = 1 if "solve" in nl_task else 0

            s2_ = agent.add_task_embedding(s2, nl_task)
            agent.update(s1_, a, s2_, events, done)

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
            if done or curriculum.stop_learning():
                break
                # s2 = env.reset()
                # agent.set_rm(task_rm_id)
                # if curriculum.stop_task(t):
                #     break

            # checking the steps time-out
            # if curriculum.stop_learning():
            #     break

            # Moving to the next state
            s1 = s2

        # if show_print: print("Done! Total reward:", training_reward)
    
    # def evaluate(self, step):
    #     tester = self.tester
    #     t_init = time.time()
    #     rewards_of_each_task = []

    #     for rm_file in tester.get_rm_files():
    #         reward = self.evaluate_episode(rm_file)
    #         rewards_of_each_task.append(reward)

    #     # for nl_file in tester.world.nl_files:
    #     #     with open(nl_file, 'r') as file:
    #     #         nl_task = file.readline().replace("\n", "")
    #     #     reward = self.evaluate_episode(nl_task)
    #     #     rewards_of_each_task.append(reward)

    #     total_reward = sum(rewards_of_each_task)
    #     tester.total_rewards.append(total_reward)

    #     cur_time = time.time()
    #     training_time = (cur_time - tester.last_test_time) / 60
    #     print("Training time: %0.2f minutes." % (training_time))
    #     tester.last_test_time = cur_time

    #     print("Steps: %d\tTesting: %0.1f" % (step, cur_time - t_init),
    #           "seconds\tTotal Reward: %0.1f" % total_reward)
    #     print("\t".join(["%0.1f" % (r) for r in rewards_of_each_task]))

    #     log_reward = {"total": sum(rewards_of_each_task)}
    #     for i in range(len(rewards_of_each_task)):
    #         log_reward["task%d" % i] = rewards_of_each_task[i]

    #     for key, value in self.loss_info.items():
    #         print("%s: %.4f" % (key, value), end='\t')
    #     print('\n')

    #     if tester.use_wandb:
    #         wandb.log({"reward": log_reward, "loss": self.loss_info, "training_time": training_time})

    def evaluate(self, step):
        tester = self.tester
        t_init = time.time()
        rewards_of_each_task = []

        for rm_file in tester.get_all_task_files():
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

    def evaluate_episode(self, task):
        env = self.create_env(task)
        s1 = env.reset()
        
        task_idx = self.tester.rm_files.index(task)
        nl_file = self.tester.world.nl_files[task_idx]
        with open(nl_file, 'r') as file:
            nl_task = file.readline().replace("\n", "")

        # Starting interaction with the environment
        r_total = 0
        for t in range(self.tester.testing_params.num_steps):
            s1_ = self.agent.add_task_embedding(s1, nl_task)
            a = self.agent.get_action(s1_, True)
            s2, env_reward, done, events = env.step(a)

            nl_task = self.agent.progression(nl_task, events)
            done = 1 if "solve" in nl_task else 0
            print("eval",  events)

            s2_ = self.agent.add_task_embedding(s2, nl_task)
            self.agent.update(s1_, a, s2_, events, done, True)
            r_total += env_reward
            # Restarting the environment (Game Over)
            if done:
                break
            # Moving to the next state
            s1 = s2

        return r_total