from abc import ABC, abstractmethod


class BaseCurriculumLearner(ABC):
    def __init__(self, tasks, total_steps, *args):
        self.tasks = tasks
        self.current_task = 0
        self.total_steps = total_steps
        self.current_step = 0
        self.current_episode = 0

    def get_tasks(self):
        return self.tasks

    def get_current_step(self):
        return self.current_step

    def add_step(self):
        self.current_step += 1

    def add_episode(self):
        self.current_episode += 1

    def get_current_task_file(self):
        return self.tasks[self.current_task]

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def stop_learning(self):
        pass

    @abstractmethod
    def get_next_task(self):
        pass


class MultiTaskCurriculumLearner(BaseCurriculumLearner):
    """
    Decides when to stop one task and which to execute next
    In addition, it controls how many steps the agent has given so far
    """

    def __init__(self, tasks, total_steps):
        super().__init__(tasks, total_steps)

    def restart(self):
        self.current_step = 0
        self.current_task = -1

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def get_next_task(self):
        self.last_restart = -1
        self.current_task = (self.current_task + 1) % len(self.tasks)
        return self.get_current_task_file()


class SingleTaskCurriculumLearner(BaseCurriculumLearner):
    def __init__(self, tasks, total_steps):
        super().__init__(tasks, total_steps)

    def restart(self):
        self.current_step = 0
        self.current_task = 0

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def get_next_task(self):
        return self.tasks[0]


class LifelongCurriculumLearner(BaseCurriculumLearner):
    def __init__(self, tasks, lifelong_curriculum, total_steps):
        super().__init__(tasks, total_steps)
        self.current_task_in_curriculum = 0
        self.num_phases = len(lifelong_curriculum)
        assert total_steps%self.num_phases==0
        self.phase_total_steps = total_steps // self.num_phases
        self.current_phase = 0
        self.current_step_of_phase = 0
        # lifelong_curriculum[i] is a list of tasks in phase i
        self.lifelong_curriculum = lifelong_curriculum
        self.current_curriculum = []

    def add_step(self):
        self.current_step += 1
        self.current_step_of_phase += 1

    def restart(self):
        self.current_step = 0
        self.current_step_of_phase = 0
        self.current_phase = 0
        self.current_curriculum = self.lifelong_curriculum[0]
        # relative task id in the current_curriculum
        self.current_task_in_curriculum = 0
        # absolute task id in the list of all tasks
        self.current_task = self.current_curriculum[0]

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def stop_curriculum(self):
        # return self.current_step - self.current_phase*self.phase_total_steps > self.phase_total_steps
        return self.current_step_of_phase >= self.phase_total_steps

    def get_next_task(self):
        if self.stop_curriculum():
            # new phase
            self.current_phase += 1
            self.current_step_of_phase = 0
            self.current_curriculum = self.lifelong_curriculum[self.current_phase]
            self.current_task_in_curriculum = 0
        else:
            self.current_task_in_curriculum = (self.current_task_in_curriculum + 1) % len(self.current_curriculum)

        self.current_task = self.current_curriculum[self.current_task_in_curriculum]
        return self.get_current_task_file()
