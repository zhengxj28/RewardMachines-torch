from src.worlds.water_world import Ball, BallAgent, WaterWorldParams, WaterWorld
from src.worlds.craft_world import CraftWorldParams, CraftWorld
from src.worlds.office_world import OfficeWorldParams, OfficeWorld
# from src.worlds.half_cheetah import LabellingHalfCheetahEnv
from src.worlds.wrapped_mujoco import WrappedMujoco
from src.worlds.base_env import BaseEnv
import random

toy_game_types = ["craftworld", "waterworld", "officeworld"]


class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """

    def __init__(self, game_type, game_params):
        self.game_type = game_type
        self.game_params = game_params


class RewardMachinesEnv(BaseEnv):
    def __init__(self, tester, task=None):
        super().__init__()
        reward_machines = tester.get_reward_machines()
        if task is not None:
            task_rm_id = tester.get_reward_machine_id_from_file(task)
            env_params = tester.get_task_params(task).game_params
            rm = reward_machines[task_rm_id]
        else:
            env_params = tester.get_task_params(task).game_params
            rm = None

        self.env_params = env_params
        self.s = None
        self.label_noise = tester.label_noise

        env_name = tester.game_type
        self.env_name = env_name
        if env_name in toy_game_types:
            if env_name == "craftworld":
                self.world = CraftWorld(env_params)
            if env_name == "waterworld":
                self.world = WaterWorld(env_params)
            if env_name == "officeworld":
                self.world = OfficeWorld(env_params)
            self.num_features = len(self.world.get_features())
            self.num_actions = len(self.world.get_actions())
            self.action_space = None
        elif env_name in ["half_cheetah"]:
            self.world = WrappedMujoco(env_name)
            self.num_features = self.world.observation_space.shape[0]
            self.num_actions = self.world.action_space.shape[0]
            self.action_space = self.world.action_space

        if rm is not None:
            self.rm = rm
            self.u = rm.get_initial_state()  # RM state

    def reset(self):
        if self.env_name in toy_game_types:
            if self.env_name == "craftworld":
                self.world = CraftWorld(self.env_params)
            if self.env_name == "waterworld":
                self.world = WaterWorld(self.env_params)
            if self.env_name == "officeworld":
                self.world = OfficeWorld(self.env_params)
            self.s = self.world.get_features()
            return self.s
        else:
            self.s = self.world.reset()
            return self.s

    def step(self, a):
        if self.env_name in toy_game_types:
            """
            We execute 'action' in the game
            Returns the reward
            """
            s1 = self.world.get_features()  # current state
            self.world.execute_action(a)  # the state of `world` is modified
            s2 = self.world.get_features()  # next state
            origin_reward = 0
            done = self.world.env_game_over
            info = {}
        else:
            s1 = self.s
            s2, origin_reward, done, truncated, info = self.world.step(a)
            self.s = s2
            done = done or truncated
        true_events = self.get_true_events()
        stochastic_events = self.get_stochastic_events()
        if self.rm is not None:
            u1 = self.u
            u2 = self.rm.get_next_state(u1, true_events)
            self.u = u2
            info['events'] = stochastic_events
            rm_reward = self.rm.get_reward(u1, u2, s1, a, s2, info, eval_mode=True)
            done = done or self.rm.is_terminal_state(u2)
        else:
            rm_reward = origin_reward
        # for stochastic rm methods
        info['constant'] = 1

        return s2, rm_reward, done, info

    def get_true_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.world.get_true_propositions()

    def get_stochastic_events(self):
        if random.random() < self.label_noise:
            return random.choice(self.world.id2events)
        else:
            return self.get_true_events()
