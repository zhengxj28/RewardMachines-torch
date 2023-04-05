import random, time
from src.worlds.game import *
from src.agents.qrm_agent import QRMAgent
from src.tester.saver import Saver


def run_qrm_experiments(alg_name, tester, curriculum, num_times, show_print):
    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(t)

        # Reseting default values
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
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1 = env.reset()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)
    for t in range(num_steps):

        # Choosing an action to perform
        a = qrm_agent.get_action(task_rm_id, u1, s1)

        # updating the curriculum
        curriculum.add_step()

        # Executing the action
        s2, env_reward, done, _ = env.step(a)  # do not use reward from env

        events = env.get_true_propositions()
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1, u2, s1, a, s2)
        training_reward += reward

        # Getting rewards and next states for each reward machine
        rewards, next_states = [], []
        for j in range(len(reward_machines)):
            j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
            rewards.append(j_rewards)
            next_states.append(j_next_states)
        # Mapping rewards and next states to specific policies in the policy bank
        rewards = qrm_agent.map_rewards(rewards)
        next_policies = qrm_agent.map_next_policies(next_states)

        # Adding this experience to the experience replay buffer
        qrm_agent.buffer.add_data(s1, a, s2, rewards, next_policies, done)

        # Learning
        cur_step = curriculum.get_current_step()
        if cur_step > learning_params.learning_starts and cur_step % learning_params.train_freq == 0:
            qrm_agent.learn()

            # if learning_params.prioritized_replay:
            #     new_priorities = abs_td_errors + learning_params.prioritized_replay_eps
            #     replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Updating the target network
        if cur_step > learning_params.learning_starts and cur_step % learning_params.target_network_update_freq == 0:
            qrm_agent.update_target_network()

        # Printing
        if show_print and (t + 1) % learning_params.print_freq == 0:
            print("Step:", t + 1, "\tTotal reward:", training_reward)

        # Testing
        # if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
        #     tester.run_test(curriculum.get_current_step(), sess, run_qrm_test, policy_bank, num_features)

        # Restarting the environment (Game Over)
        if done:
            s2 = env.reset()
            u2 = rm.get_initial_state()

            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1, u1 = s2, u2

    if show_print: print("Done! Total reward:", training_reward)


def run_qrm_test(reward_machines, task_params, task_rm_id, learning_params, testing_params, policy_bank,
                 num_features):
    # Initializing parameters
    task = Game(task_params)
    rm = reward_machines[task_rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Choosing an action using the right policy
        a = policy_bank.get_best_action(task_rm_id, u1, s1_features.reshape((1, num_features)), add_noise=False)

        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        u2 = rm.get_next_state(u1, task.get_true_propositions())
        r = rm.get_reward(u1, u2, s1, a, s2, is_training=False)

        r_total += r * learning_params.gamma ** t

        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2):
            break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

    return r_total
