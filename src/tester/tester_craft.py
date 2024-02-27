from src.worlds.reward_machines_env import GameParams
from src.worlds.craft_world import CraftWorldParams


class TesterCraftWorld:
    def __init__(self, experiment, tabular_case, gamma, data=None):
        if data is None:
            # Reading the file
            self.experiment = experiment
            self.tabular_case = tabular_case
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()
            # setting the test attributes
            self.map = eval(lines[1])
            self.consider_night = eval(lines[2])
            self.rm_files = eval(lines[3])
            self.ltl_files = eval(lines[4])
            self.srm_files = eval(lines[5])
            self.lifelong_curriculum = eval(lines[6])

            # I compute the optimal reward
            # self.optimal = {}
            # for i in range(len(self.tasks)):
            #     self.optimal[self.tasks[i]] = gamma ** (float(optimal_aux[i]) - 1)
        else:
            self.experiment = data["experiment"]
            self.map = data["map"]
            self.rm_files = data["tasks"]
            self.consider_night = data["consider_night"]
            # self.optimal = data["optimal"]

    def get_dictionary(self):
        d = {}
        d["experiment"] = self.experiment
        d["map"] = self.map
        d["tasks"] = self.rm_files
        d["consider_night"] = self.consider_night
        d["optimal"] = self.optimal
        return d

    def get_reward_machine_files(self):
        return self.rm_files

    def get_task_specifications(self):
        return self.rm_files

    def get_task_params(self, task_specification):
        params = CraftWorldParams(self.map, self.tabular_case, self.consider_night)
        return GameParams("craftworld", params)

    def get_task_rm_file(self, task_specification):
        return task_specification
