from src.worlds.reward_machines_env import GameParams


class TesterMujoco:
    def __init__(self, experiment):
        # Reading the file
        self.experiment = experiment
        f = open(experiment)
        lines = [l.rstrip() for l in f]
        f.close()
        self.rm_files = eval(lines[1])
        self.ltl_files = eval(lines[2])

        # self.tasks = [(t, m) for t in self.rm_files for m in self.maps]
        # self.current_map = 0
        # # NOTE: Update the optimal value per task when you know it...
        # self.optimal = {}
        # for i in range(len(self.tasks)):
        #     self.optimal[self.tasks[i]] = 1

    # def get_reward_machine_files(self):
    #     return self.rm_files
    #
    # def get_task_specifications(self):
    #     return self.tasks

    def get_task_params(self, task_specification):
        # if type(task_specification) == tuple: _, map_file = task_specification

        # params = WaterWorldParams(map_file, max_x=self.max_x, max_y=self.max_y, b_radius=self.b_radius,
        #                           b_num_per_color=self.b_num_per_color,
        #                           use_velocities=self.use_velocities,
        #                           ball_disappear=self.ball_disappear)

        return GameParams("mujoco", None)

    def get_task_rm_file(self, task_specification):
        if type(task_specification) == tuple:
            rm_task, _ = task_specification
        if type(task_specification) == str:
            rm_task = task_specification
        return rm_task
