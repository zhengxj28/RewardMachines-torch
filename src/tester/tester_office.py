from src.worlds.reward_machines_env import GameParams
from src.worlds.office_world import OfficeWorldParams

class TesterOfficeWorld:
    def __init__(self, experiment, gamma, data=None):
        if data is None:
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()
            # setting the test attributes
            self.rm_files = eval(lines[1])
            optimal_aux = eval(lines[2])
            self.ltl_files = eval(lines[3])
            self.srm_files = eval(lines[4])
            self.lifelong_curriculum = eval(lines[5])

            # I compute the optimal reward
            self.optimal = {}
            for i in range(len(self.rm_files)):
                self.optimal[self.rm_files[i]] = gamma ** (float(optimal_aux[i]) - 1)
        else:
            self.experiment = data["experiment"]
            self.rm_files   = data["tasks"]
            self.optimal = data["optimal"]

    def get_dictionary(self):
        d = {}
        d["experiment"] = self.experiment
        d["tasks"] = self.rm_files
        d["optimal"] = self.optimal
        return d

    # def get_reward_machine_files(self):
    #     return self.rm_files

    # def get_task_specifications(self):
    #     return self.rm_files

    def get_task_params(self, task_specification):
        return GameParams("officeworld", OfficeWorldParams())

    def get_task_rm_file(self, task_specification):
        return task_specification
