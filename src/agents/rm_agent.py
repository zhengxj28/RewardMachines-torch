import time, random
import numpy as np


class RMAgent:
    """
    Agent with reward machines only, without RL module.
    """

    def __init__(self, reward_machines, task2rm_id):
        self.reward_machines = reward_machines  # all reward machines
        self.task2rm_id = task2rm_id
        self.rm_id = None  # current reward machine
        self.u = None  # current state of current reward machine
        self.rm_id_eval = None  # current reward machine while evaluating
        self.u_eval = None  # current state of current reward machine while evaluating

        # Decomposing reward machines: We learn one policy per state in a reward machine
        t_i = time.time()
        self.state2policy = {}
        policies_to_add = self._decompose_reward_machines(reward_machines)
        if reward_machines[0].generate_by_ltl:
            self.policy2ltl = {}
            for state, policy in self.state2policy.items():
                rm_id, rm_state = state
                rm = reward_machines[rm_id]
                self.policy2ltl[policy] = rm.state2ltl[rm_state]
            self.ltl2policy = dict([(v, k) for k, v in self.policy2ltl.items()])

        print("Decomposing RMs is done! (in %0.2f minutes)" % ((time.time() - t_i) / 60))
        num_policies = len(policies_to_add)
        self.num_policies = num_policies

    def _decompose_reward_machines(self, reward_machines):
        # Some machine states might have equivalent Q-functions
        # In those cases, we learn only one policy for them
        policies_to_add = []
        # We add_data one constant policy for every terminal state
        policies_to_add.append("constant")  # terminal policy has id '0'
        # Associating policies to each machine state
        for i, rm in enumerate(reward_machines):
            for ui in range(len(rm.get_states())):
                if rm.is_terminal_state(ui):
                    # terminal states goes to the constant policy
                    self.state2policy[(i, ui)] = 0
                else:
                    # associating a policy for this reward machine state
                    policy_id = None
                    for j, uj in self.state2policy:
                        # checking if we already have a policy for an equivalent reward machine
                        if rm.is_this_machine_equivalent(ui, reward_machines[j], uj):
                            print(
                                "Match: reward machine %d from state %d is equivalent to reward machine %d from state "
                                "%d" % (i, ui, j, uj))
                            policy_id = self.state2policy[(j, uj)]
                            break
                    if policy_id is None:
                        # creating a new policy for this node
                        policy_id = len(policies_to_add)
                        policies_to_add.append("machine" + str(i) + "_state" + str(ui))
                    self.state2policy[(i, ui)] = policy_id
        return policies_to_add

    def _map_rewards(self, rewards):
        """
        reward format:
           [R0, ..., Rn] where Ri is the list of rewards gotten by each state on the reward machine 'i'
        returns a single vector with the corresponding rewards given to every policy
        """
        policy_rewards = np.zeros(self.num_policies, dtype=np.float64)
        done = set()
        for i in range(len(rewards)):
            for j in range(len(rewards[i])):
                policy_id = self.state2policy[(i, j)]
                if policy_id not in done:
                    policy_rewards[policy_id] = rewards[i][j]
                #     done.add(policy_id)
                # elif policy_rewards[policy_id] != rewards[i][j]:
                #     print("(%d,%d) -> pos %d" % (i, j, policy_id))
                #     print("reward discrepancy:", policy_rewards[policy_id], "vs", rewards[i][j])
                #     print("state2policy", self.state2policy)
                #     print("rewards", rewards)
                #     raise ValueError("Error! equivalent policies are receiving different rewards!")
        return policy_rewards

    def _map_next_policies(self, next_states):
        """
        next_states format:
           [U0, ..., Un] where Ui is the list of next states for each state on the reward machine 'i'
        returns a single vector with the corresponding next policy per each policy
        """
        next_policies = np.zeros(self.num_policies, dtype=np.float64)
        done = set()
        for i in range(len(next_states)):
            for j in range(len(next_states[i])):
                u = self.state2policy[(i, j)]
                u_next = self.state2policy[(i, next_states[i][j])]
                if u not in done:
                    next_policies[u] = u_next
        return next_policies

    def set_rm(self, rm_id, eval_mode=False, u=None):
        if eval_mode:
            self.rm_id_eval = rm_id
            self.u_eval = self.reward_machines[self.rm_id_eval].get_initial_state() if u is None else u
        else:
            self.rm_id = rm_id
            self.u = self.reward_machines[self.rm_id].get_initial_state() if u is None else u

    def update_rm_state(self, events, eval_mode=False):
        if eval_mode:
            self.u_eval = self.reward_machines[self.rm_id_eval].get_next_state(self.u_eval, events)
        else:
            self.u = self.reward_machines[self.rm_id].get_next_state(self.u, events)

    def get_rewards_and_next_policies(self, s1, a, s2, info):
        # Getting rewards and next states for each reward machine to learn
        rewards, next_policies = [], []
        reward_machines = self.reward_machines
        for j in range(len(reward_machines)):
            j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, info)
            rewards.append(j_rewards)
            next_policies.append(j_next_states)
        # Mapping rewards and next states to specific policies in the policy bank
        rewards = self._map_rewards(rewards)
        next_policies = self._map_next_policies(next_policies)
        return rewards, next_policies


class SRMAgent(RMAgent):
    """
    Agent for Stochastic Reward Machines
    """

    def __init__(self, reward_machines, task2rm_id, label_noise):
        RMAgent.__init__(self, reward_machines, task2rm_id)
        self.label_noise = label_noise
        self.num_states_of_rm = [len(rm.U) for rm in reward_machines]
        all_events = []  # all events of all rm
        for rm in reward_machines:
            all_events += rm.all_events

        all_events = list(set(all_events))
        all_events.sort()
        self.events2id = dict([(events, i) for i, events in enumerate(all_events)])
        self.num_events = len(all_events)
        rm_event_id2ag_event_id = {}
        for i, rm in enumerate(reward_machines):
            for rm_event_id, events in enumerate(rm.all_events):
                rm_event_id2ag_event_id[(i, rm_event_id)] = self.events2id[events]

        self.belief_u = None
        self.belief_u_eval = None
        self.belief_policy = np.zeros(self.num_policies)
        self.belief_policy_eval = np.zeros(self.num_policies)

        # assemble transition matrices and reward matrices of each rm into one matrix
        # self.policy_tran_matrix[l,u1,u2]=prob from policy u1 to policy u2 under events l
        self.policy_tran_matrix = np.zeros([self.num_events, self.num_policies, self.num_policies])
        for i, rm in enumerate(reward_machines):
            for rm_event_id, events in enumerate(rm.all_events):
                ag_event_id = rm_event_id2ag_event_id[(i, rm_event_id)]
                for rm_u1 in range(rm.delta_u.shape[1]):
                    ag_u1 = self.state2policy[(i, rm_u1)]
                    for rm_u2 in range(rm.delta_u.shape[2]):
                        ag_u2 = self.state2policy[(i, rm_u2)]
                        # need to assert: different states do not map to the same policy
                        # i.e. policy_tran_matrix[ag_event_id, ag_u1, ag_u2] do not have 2 or more values
                        self.policy_tran_matrix[ag_event_id, ag_u1, ag_u2] = rm.delta_u[rm_event_id, rm_u1, rm_u2]

        all_reward_components = set()
        for i, rm in enumerate(reward_machines):
            for com_name in rm.reward_components.keys():
                all_reward_components.add(com_name)
        all_reward_components = list(all_reward_components)
        all_reward_components.sort()
        self.reward_components = dict([(com_name, i) for i, com_name in enumerate(all_reward_components)])
        self.num_reward_components = len(all_reward_components)
        self.reward_matrix = np.zeros([self.num_reward_components, self.num_policies, self.num_policies])
        for i, rm in enumerate(reward_machines):
            for com_name, com_id in rm.reward_components.items():
                for rm_u1 in range(rm.delta_u.shape[1]):
                    ag_u1 = self.state2policy[(i, rm_u1)]
                    for rm_u2 in range(rm.delta_u.shape[2]):
                        ag_u2 = self.state2policy[(i, rm_u2)]
                        ag_com_id = self.reward_components[com_name]
                        self.reward_matrix[ag_com_id, ag_u1, ag_u2] = rm.reward_matrix[com_id, rm_u1, rm_u2]

        self.terminal_policy_array = np.zeros([self.num_policies])
        for i in range(self.num_policies):
            if i in self.terminal_policy:
                self.terminal_policy_array[i] = 1

    def _decompose_reward_machines(self, reward_machines):
        """
        it is overwritten for stochastic rm, which is different from deterministic rm
        """
        # Some machine states might have equivalent Q-functions
        # In those cases, we learn only one policy for them
        policies_to_add = []
        # We add_data one constant policy for every terminal state
        policies_to_add.append("positive")  # positive terminal policy has id '0'
        policies_to_add.append("negative")  # negative terminal policy has id '1'
        self.terminal_policy = {0, 1}
        # Associating policies to each machine state
        for i, rm in enumerate(reward_machines):
            for ui in range(len(rm.get_states())):
                if rm.is_pos_terminal_state(ui):
                    # terminal states goes to the constant policy
                    self.state2policy[(i, ui)] = 0
                elif rm.is_neg_terminal_state(ui):
                    self.state2policy[(i, ui)] = 1
                else:
                    # associating a policy for this reward machine state
                    policy_id = None
                    for j, uj in self.state2policy:
                        # checking if we already have a policy for an equivalent reward machine
                        if rm.is_this_machine_equivalent(ui, reward_machines[j], uj):
                            print(
                                "Match: reward machine %d from state %d is equivalent to reward machine %d from state "
                                "%d" % (i, ui, j, uj))
                            policy_id = self.state2policy[(j, uj)]
                            break
                    if policy_id is None:
                        # creating a new policy for this node
                        policy_id = len(policies_to_add)
                        policies_to_add.append("machine" + str(i) + "_state" + str(ui))
                    self.state2policy[(i, ui)] = policy_id
        return policies_to_add

    def set_rm(self, rm_id, eval_mode=False, u=None):
        RMAgent.set_rm(self, rm_id, eval_mode, u)
        if eval_mode:
            self.belief_u_eval = np.zeros(self.num_states_of_rm[rm_id])
            self.belief_u_eval[0] = 1
            self.belief_policy_eval = np.zeros(self.num_policies)
            self.belief_policy_eval[self.state2policy[(rm_id, 0)]] = 1
        else:
            self.belief_u = np.zeros(self.num_states_of_rm[rm_id])
            self.belief_u[0] = 1
            self.belief_policy = np.zeros(self.num_policies)
            self.belief_policy[self.state2policy[(rm_id, 0)]] = 1

    def update_belief_state(self, events, eval_mode=False):
        def get_tran_matrix(rm, events):
            if events in rm.events2id:
                events_id = rm.events2id[events]
                return rm.delta_u[events_id]
            else:
                # if rm do not consider `events`, then do not transit, i.e. return identity matrix
                return np.eye(len(rm.U))

        if eval_mode:
            cur_rm = self.reward_machines[self.rm_id_eval]
            tran_matrix = get_tran_matrix(cur_rm, events)
            self.belief_u_eval = np.matmul(self.belief_u_eval, tran_matrix)
        else:
            cur_rm = self.reward_machines[self.rm_id]
            tran_matrix = get_tran_matrix(cur_rm, events)
            self.belief_u = np.matmul(self.belief_u, tran_matrix)

    def update_belief_policy(self, label_prob, eval_mode=False):
        belief_policy = self.belief_policy_eval if eval_mode else self.belief_policy
        next_belief_policy = np.zeros(self.num_policies)
        for l, prob in enumerate(label_prob):
            next_belief_policy += prob * np.matmul(belief_policy, self.policy_tran_matrix[l])
        if eval_mode:
            self.belief_policy_eval = next_belief_policy
        else:
            self.belief_policy = next_belief_policy

    def belief_state2policy(self, eval_mode):
        if eval_mode:
            cur_rm_id = self.rm_id_eval
            cur_belief_u = self.belief_u_eval
        else:
            cur_rm_id = self.rm_id
            cur_belief_u = self.belief_u
        belief_policy = np.zeros(self.num_policies)
        for i, p in enumerate(cur_belief_u):
            # different rm states may map to the same policy, so add the probability `p`
            belief_policy[self.state2policy[(cur_rm_id, i)]] += p
        return belief_policy

    def get_label_prob(self, info):
        label_prob = self.label_noise*np.ones(self.num_events)/self.num_events
        if info['events'] in self.events2id:
            events = info['events']
        else:
            # TODO: cope with possible events in the env which is not considered in rm
            events = ''
        label_prob[self.events2id[events]] += (1-self.label_noise)
        return label_prob

    def get_srm_rewards(self, info):
        """
        return a reward matrix `srm_rewards`
        srm_rewards[u1,u2] is the reward when transit from policy u1 to policy u2
        """
        srm_rewards = np.zeros([self.num_policies, self.num_policies])
        for component, value in info.items():
            if component in self.reward_components:
                com_id = self.reward_components[component]
                srm_rewards += value*self.reward_matrix[com_id]
        return srm_rewards