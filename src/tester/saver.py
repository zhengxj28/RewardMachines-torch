import os
import numpy as np
from src.tester.test_utils import save_json

class Saver:
    def __init__(self, alg_name, tester, curriculum):
        # folder = "../tmp/"
        folder = os.path.join("..", "data")

        # e.g. tester.experiment = "../experiments/craft/tests/craft_0.txt"
        exp_dir = os.path.join(folder, tester.game_type)
        self.file_out_path = os.path.join(exp_dir, alg_name)
        if not os.path.exists(self.file_out_path):
            os.makedirs(self.file_out_path)
        self.tester = tester

    def save_results(self, filename):
        np_results = np.array(self.tester.total_rewards)
        np.save(os.path.join(self.file_out_path, filename), np_results)
        # results = {}
        # results['game_type'] = self.tester.game_type
        # results['steps'] = self.tester.steps
        # results['results'] = self.tester.results
        # results['world'] = self.tester.get_world_dictionary()
        # save_json(self.file_out, results)
