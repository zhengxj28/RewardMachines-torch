import random
import time
import wandb

from src.algos.qrm import QRMAlgo
from src.agents.qsrm_agent import QSRMAgent


class QSRMAlgo(QRMAlgo):
    def __init__(self, tester, curriculum, show_print, use_cuda, label_noise):
        super().__init__(tester, curriculum, show_print, use_cuda)

        self.show_print = show_print
        self.use_cuda = use_cuda
        self.loss_info = {}

        num_features = tester.num_features
        num_actions = tester.num_actions

        learning_params = tester.learning_params
        model_params = tester.model_params
        self.agent = QSRMAgent(num_features, num_actions,
                               learning_params,
                               model_params,
                               tester.get_reward_machines(),
                               tester.task2rm_id,
                               use_cuda,
                               label_noise)
