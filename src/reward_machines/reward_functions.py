import math
from abc import ABC, abstractmethod

class RewardFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_reward(self, s1, a, s2, info):
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def get_type(self):
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def compare_to(self, other):
        raise NotImplementedError("To be implemented")

class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def compare_to(self, other):
        return self.get_type() == other.get_type() and self.c == other.c

    def get_reward(self, s1, a, s2, info):
        return self.c

class RewardControl(RewardFunction):
    def __init__(self, direction=1):
        super().__init__()
        self.direction = direction

    def get_type(self):
        return "dense"

    def compare_to(self, other):
        return False

    def get_reward(self, s1, a, s2, info):
        return self.direction*info['reward_run']+info['reward_ctrl']
