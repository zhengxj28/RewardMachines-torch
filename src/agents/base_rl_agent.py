from abc import ABC, abstractmethod

class BaseRLAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        # one-step of the agent
        pass

    @abstractmethod
    def get_action(self, s):
        # choose an action under the state s
        pass

    @abstractmethod
    def update(self, *trajectory):
        # update the status of agent using trajectory
        # for example: update replay buffer, update bool flag
        pass