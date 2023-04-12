from tester.tester import Tester
from tester.testing_params import TestingParameters
from common.curriculum import CurriculumLearner
from tester.learning_params import LearningParameters


def get_params_craft_world(experiment, use_rs, use_wandb):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters(
        test=True,
        test_freq=10 * step_unit,
        num_steps=1000
    )

    # configuration of learning params
    learning_params = LearningParameters(
        gamma=0.9,
        print_freq=step_unit,
        train_freq=1,
        tabular_case=True,
        max_timesteps_per_task=testing_params.num_steps,
        lr=1,
        batch_size=1,
        buffer_size=1,
        learning_starts=1,
    )

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs, use_wandb)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 1000 * step_unit
    curriculum.min_steps = 1

    print("Craft World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("epsilon: ", learning_params.epsilon)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)

    return tester, curriculum


def get_params_office_world(experiment, use_rs, use_wandb):
    step_unit = 500

    # configuration of testing params
    testing_params = TestingParameters(
        test=True,
        test_freq=1 * step_unit,
        num_steps=1000
    )

    # configuration of learning params
    learning_params = LearningParameters(
        gamma=0.9, tabular_case=True,
        max_timesteps_per_task=testing_params.num_steps,
        epsilon=0.1,
        lr=1.0,
        batch_size=1,
        learning_starts=1,
        buffer_size=1,
        target_network_update_freq=1
    )

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs, use_wandb)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps  # 100
    curriculum.total_steps = 100 * step_unit
    # curriculum.total_steps = 10 * step_unit  # for test only
    curriculum.min_steps = 1

    print("Office World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("epsilon: ", learning_params.epsilon)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)

    return tester, curriculum


def get_params_water_world(experiment, use_rs, use_wandb):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters(
        test=True,
        test_freq=10 * step_unit,
        num_steps=600
    )

    # configuration of learning params
    learning_params = LearningParameters(
        lr=1e-5,  # 5e-5 seems to be better than 1e-4
        gamma=0.9,
        max_timesteps_per_task=testing_params.num_steps,
        buffer_size=50000,
        print_freq=step_unit,
        train_freq=1,
        batch_size=32,
        target_network_update_freq=100,  # obs: 500 makes learning more stable, but slower
        learning_starts=1000,
        tabular_case=False,
        use_random_maps=False,
        use_double_dqn=True,
        prioritized_replay=True,
        num_hidden_layers=6,
        num_neurons=64
    )

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs, use_wandb)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = 300
    curriculum.total_steps = 2000 * step_unit
    curriculum.min_steps = 1

    print("Water World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return tester, curriculum
