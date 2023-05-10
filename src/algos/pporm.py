import random
import time
import wandb

from src.agents.pporm_agent import PPORMAgent
from src.worlds.game import *

def run_pporm_experiments(tester, curriculum, show_print, use_cuda):
    learning_params = tester.learning_params

    # Resetting default values
    curriculum.restart()

    task_aux = Game(tester.get_task_params(curriculum.get_current_task()), None)
    num_features = task_aux.num_features
    num_actions = task_aux.num_actions

    pporm_agent = PPORMAgent(num_features, num_actions, learning_params, tester.get_reward_machines(), curriculum, use_cuda)

    # Task loop
    while not curriculum.stop_learning():
        if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
        rm_file = curriculum.get_next_task()
        # Running 'task_rm_id' for one episode
        run_pporm_task(rm_file, pporm_agent, tester, curriculum, show_print)


def run_pporm_task(rm_file, agent, tester, curriculum, show_print):
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
    agent.set_rm(task_rm_id)
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1 = env.reset()

    # Starting interaction with the environment
    if show_print: print("Executing Task", task_rm_id)
    for t in range(num_steps):
        curriculum.add_step()
        a, log_prob = agent.get_action(s1)
        # do not use reward from env to learn
        s2, env_reward, done, events = env.step(a)

        agent.update(s1, a, s2, events, log_prob, done)

        # Learning
        cur_step = curriculum.get_current_step()
        if cur_step > learning_params.learning_starts:
            # if cur_step % learning_params.train_freq == 0:
            if agent.buffer.is_full() or done:
                policy_loss, value_loss = agent.learn()
                agent.buffer.clear()  # Once learned, clear the data

        # Printing
        training_reward += env_reward
        if show_print and (t + 1) % learning_params.print_freq == 0:
            print("Step:", t + 1, "\tTotal reward:", training_reward)

        # Testing
        if testing_params.test and cur_step % testing_params.test_freq == 0:
            tester.run_test(cur_step, run_pporm_test, agent)

        # Restarting the environment (Game Over)
        if done:
            s2 = env.reset()
            agent.set_rm(task_rm_id)
            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1 = s2

    if show_print: print("Done! Total reward:", training_reward)


def run_pporm_test(reward_machines, task_params, task_rm_id, testing_params, rm_agent):
    """
    run a test (single task) for algorithms that use reward machines.
    """
    # Initializing parameters
    rm = reward_machines[task_rm_id]
    env = Game(task_params, rm)
    s1 = env.reset()
    rm_agent.set_rm(task_rm_id, eval_mode=True)

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        a, _ = rm_agent.get_action(s1, eval_mode=True)
        s2, env_reward, done, events = env.step(a)
        rm_agent.update_rm_state(events, eval_mode=True)
        r_total += env_reward
        # Restarting the environment (Game Over)
        if done:
            break
        # Moving to the next state
        s1 = s2

    return r_total