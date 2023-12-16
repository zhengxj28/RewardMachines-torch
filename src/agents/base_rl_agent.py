from abc import ABC, abstractmethod
from src.common.auto_gpu import auto_gpu
import torch

class BaseRLAgent(ABC):
    def __init__(self, use_cuda=False):
        if use_cuda:
            if torch.cuda.is_available():
                gpu_id = auto_gpu()
                self.device = torch.device(f'cuda:{gpu_id}')
                print(f"Using device: cuda:{gpu_id}")
            else:
                raise ValueError("Cuda is not available, but set use_cuda=True.")
        else:
            self.device = torch.device('cpu')

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
