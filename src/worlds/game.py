from src.worlds.water_world import Ball, BallAgent, WaterWorldParams, WaterWorld
from src.worlds.craft_world import CraftWorldParams, CraftWorld
from src.worlds.office_world import OfficeWorldParams, OfficeWorld


class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """

    def __init__(self, game_type, game_params):
        self.game_type = game_type
        self.game_params = game_params
        if self.game_type not in ["craftworld", "waterworld", "officeworld"]:
            print(self.game_type, "is not currently supported")
            exit()


"""
The class `Game` has been modified.
Game = World (transition model) + RM (reward function)
"""


class Game:
    def __init__(self, params, rm):
        self.params = params
        if params.game_type == "craftworld":
            self.world = CraftWorld(params.game_params)
        if params.game_type == "waterworld":
            self.world = WaterWorld(params.game_params)
        if params.game_type == "officeworld":
            self.world = OfficeWorld(params.game_params)
        self.num_features = len(self.world.get_features())
        self.num_actions = len(self.world.get_actions())
        if rm:
            self.rm = rm
            self.u = rm.get_initial_state()  # RM state

    def reset(self):
        params = self.params
        # TODO: add `reset()` in `worlds` to simplify codes
        if params.game_type == "craftworld":
            self.world = CraftWorld(params.game_params)
        if params.game_type == "waterworld":
            self.world = WaterWorld(params.game_params)
        if params.game_type == "officeworld":
            self.world = OfficeWorld(params.game_params)
        return self.world.get_features()

    def step(self, a):
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
        r = self.rm.get_reward(u1, u2, s1, a, s2)
        done = self.rm.is_terminal_state(u2) or self.world.env_game_over
        events = self.get_true_propositions()
        if r == 0: r = -0.01
        return s2, r, done, events

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.world.get_true_propositions()
