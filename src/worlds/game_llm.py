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



class LLMGame:
    def __init__(self, params):
        self.params = params
        if params.game_type == "craftworld":
        self.num_actions = len(self.world.get_actions())

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
        done = 0 # TODO
        events = self.get_true_propositions()
        r = -0.01 # TODO
        return s2, r, done, events

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.world.get_true_propositions()
