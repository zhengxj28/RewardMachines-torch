from abc import ABC, abstractmethod

class BaseRLAgent(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def learn(self):
        # one-step of the agent
        pass

    @abstractmethod
    def get_action(self, s, eval_mode):
        # choose an action under the state s
        pass

    @abstractmethod
    def update(self, *trajectory):
        # update the status of agent using trajectory
        # for example: update replay buffer, update bool flag
        pass

    def reset_status(self, *args):
        # reset high-level status, i.e. RM state, bool flag
        pass

class DefaultAgent(BaseRLAgent):
    def learn(self):
        pass

    def get_action(self, s, eval_mode):
        return 0

    def update(self, *trajectory):
        pass
