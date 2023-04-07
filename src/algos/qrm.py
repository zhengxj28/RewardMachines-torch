import random
import time
import copy

from src.agents.qrm_agent import QRMAgent
from src.tester.saver import Saver
from src.worlds.game import *


def run_qrm_experiments(alg_name, tester, curriculum, num_times, show_print):
    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(t)

        # Resetting default values
        curriculum.restart()

        task_aux = Game(tester.get_task_params(curriculum.get_current_task()), None)
        num_features = task_aux.num_features
        num_actions = task_aux.num_actions

        qrm_agent = QRMAgent(num_features, num_actions, learning_params, tester.get_reward_machines(), curriculum)

        # Task loop
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file = curriculum.get_next_task()
            # Running 'task_rm_id' for one episode

            run_qrm_task(rm_file, qrm_agent, tester, curriculum, show_print)

        # Backing up the results
        saver.save_results()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")


def run_qrm_task(rm_file, qrm_agent, tester, curriculum, show_print):
    """
    This code runs one training episode. 
        - rm_file: It is the path towards the RM machine to solve on this episode
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    reward_machines = tester.get_reward_machines()
    task_rm_id = tester.get_reward_machine_id_from_file(rm_file)
    task_params = tester.get_task_params(rm_file)
    num_steps = learning_params.max_timesteps_per_task
    rm = reward_machines[task_rm_id]
    env = Game(task_params, rm)
    qrm_agent.set_rm(task_rm_id)
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1 = env.reset()

    # Starting interaction with the environment
    if show_print: print("Executing Task", task_rm_id)
    for t in range(num_steps):
        curriculum.add_step()
        a = qrm_agent.get_action(s1)
        # do not use reward from env to learn
        s2, env_reward, done, events = env.step(a)

        # Getting rewards and next states for each reward machine to learn
        rewards, next_policies = [], []
        for j in range(len(reward_machines)):
            j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
            rewards.append(j_rewards)
            next_policies.append(j_next_states)
        # Mapping rewards and next states to specific policies in the policy bank
        rewards = qrm_agent.map_rewards(rewards)
        next_policies = qrm_agent.map_next_policies(next_policies)

        qrm_agent.update_rm_state(events)

        # Adding this experience to the experience replay buffer
        qrm_agent.buffer.add_data(s1, a, s2, rewards, next_policies, done)

        # Learning
        cur_step = curriculum.get_current_step()
        if cur_step > learning_params.learning_starts:
            if cur_step % learning_params.train_freq == 0:
                qrm_agent.learn()
            # Updating the target network
            if cur_step % learning_params.target_network_update_freq == 0:
                qrm_agent.update_target_network()

            # if learning_params.prioritized_replay:
            #     new_priorities = abs_td_errors + learning_params.prioritized_replay_eps
            #     replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Printing
        training_reward += env_reward
        if show_print and (t + 1) % learning_params.print_freq == 0:
            print("Step:", t + 1, "\tTotal reward:", training_reward)

        # Testing
        if testing_params.test and cur_step % testing_params.test_freq == 0:
            tester.run_test(cur_step, run_qrm_test, qrm_agent)

        # Restarting the environment (Game Over)
        if done:
            s2 = env.reset()
            qrm_agent.set_rm(task_rm_id)
            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1 = s2

    if show_print: print("Done! Total reward:", training_reward)


def run_qrm_test(reward_machines, task_params, task_rm_id, testing_params, qrm_agent):
    # Initializing parameters
    rm = reward_machines[task_rm_id]
    env = Game(task_params, rm)
    s1 = env.reset()
    qrm_agent.set_rm(task_rm_id, eval_mode=True)

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        a = qrm_agent.get_action(s1, eval_mode=True)
        s2, env_reward, done, events = env.step(a)
        qrm_agent.update_rm_state(events, eval_mode=True)
        r_total += env_reward
        # Restarting the environment (Game Over)
        if done:
            break
        # Moving to the next state
        s1 = s2

    return r_total
