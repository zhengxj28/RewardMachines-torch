from abc import ABC, abstractmethod
from src.agents.base_rl_agent import BaseRLAgent

class BaseAlgo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_experiments(self):
        pass

    @abstractmethod
    def run_episode(self, *args):
        pass

    @abstractmethod
    def evaluate(self, *args):
        pass

