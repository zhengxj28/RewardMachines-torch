from abc import ABC, abstractmethod
class BaseEnv(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def reset(self, *args):
        return None

    @abstractmethod
    def step(self, a):
        pass


class DefaultEnv(BaseEnv):
    def reset(self, *args):
        return None

    def step(self, a):
        next_state, reward, done, info = None, None, None, None
        return next_state, reward, done, info