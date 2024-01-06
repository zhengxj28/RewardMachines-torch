from src.worlds.water_world import Ball, BallAgent, WaterWorldParams, WaterWorld
from src.worlds.craft_world import CraftWorldParams, CraftWorld
from src.worlds.office_world import OfficeWorldParams, OfficeWorld
from src.worlds.half_cheetah import LabellingHalfCheetahEnv
from src.worlds.base_env import BaseEnv

toy_game_types = ["craftworld", "waterworld", "officeworld"]

class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """

    def __init__(self, game_type, game_params):
        self.game_type = game_type
        self.game_params = game_params
        if self.game_type not in toy_game_types:
            raise NotImplementedError(self.game_type + " is not currently supported")



class RewardMachinesEnv(BaseEnv):
    def __init__(self, tester, task):
        reward_machines = tester.get_reward_machines()
        task_rm_id = tester.get_reward_machine_id_from_file(task)
        env_params = tester.get_task_params(task).game_params
        self.env_params = env_params

        rm = reward_machines[task_rm_id]
        super().__init__()
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
        elif env_name in ["half_cheetah"]:
            self.world = LabellingHalfCheetahEnv()

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
            return self.world.get_features()
        else:
            return self.world.reset()


    def step(self, a):
        if self.env_name in toy_game_types:
            """
            We execute 'action' in the game
            Returns the reward
            """
            s1 = self.world.get_features()  # current state
            self.world.execute_action(a)  # the state of `world` is modified
            s2 = self.world.get_features()  # next state
            events = self.world.get_true_propositions()
            u1 = self.u
            u2 = self.rm.get_next_state(u1, events)
            self.u = u2
            r = self.rm.get_reward(u1, u2, s1, a, s2, eval_mode=True)
            done = self.rm.is_terminal_state(u2) or self.world.env_game_over
            if r == 0: r = -0.01
            events = self.get_true_propositions()
        else:
            s2, r, done, info = self.world.step(a)

        return s2, r, done, events

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.world.get_true_propositions()
