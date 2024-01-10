import time
import wandb
from src.algos.base_algo import BaseAlgo


class NonMDPAlgo(BaseAlgo):
    def __init__(self, tester, curriculum):
        super().__init__(tester, curriculum)



    def train_episode(self, *args):
        pass


