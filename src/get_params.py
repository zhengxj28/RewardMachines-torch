from tester.tester import Tester
from tester.testing_params import TestingParameters
from common.curriculum import MultiTaskCurriculumLearner
from tester.learning_params import LearningParameters

def get_tester_curriculum(learning_params, testing_params, experiment, use_rs, use_wandb):
    tester = Tester(learning_params, testing_params, experiment, use_rs, use_wandb)

    # Setting the curriculum learner
    curriculum = MultiTaskCurriculumLearner(tester.get_task_rms())
    curriculum.total_steps = learning_params.step_unit*learning_params.total_units
    return tester, curriculum

def get_params_craft_world(alg_name, experiment, use_rs, use_wandb):
    step_unit = 1000
    # configuration of testing params
    testing_params = TestingParameters(
        test=True,
        test_freq=10 * step_unit,
        num_steps=1000
    )

    if alg_name in ["qrm", "qrm-rs"]:
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
    if alg_name in ["pporm", "pporm-rs"]:
        # configuration of learning params
        learning_params = LearningParameters(
            gamma=0.9, tabular_case=True,
            max_timesteps_per_task=testing_params.num_steps,
            lr=1e-1,
            clip_rate=0.1,
            lam=0.8,
            n_updates=10,
            learning_starts=1,
            buffer_size=testing_params.num_steps,
            batch_size=8
        )

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs, use_wandb)

    # Setting the curriculum learner
    curriculum = MultiTaskCurriculumLearner(tester.get_task_rms())
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


def get_params_office_world(alg_name, experiment, use_rs, use_wandb):

    # configuration of testing params
    testing_params = TestingParameters(
        test=True,
        test_freq=1 * 500,
        num_steps=1000
    )

    if alg_name in ["qrm", "qrm-rs"]:
        # configuration of learning params
        learning_params = LearningParameters(
            total_units=100,
            step_unit=500,
            gamma=0.9, tabular_case=True,
            # max_timesteps_per_task=1000,
            epsilon=0.1,
            lr=1,
            batch_size=1,
            learning_starts=1,
            buffer_size=1,
            target_network_update_freq=1
        )

    # if alg_name in ["pporm", "pporm-rs"]:
    #     # configuration of learning params
    #     learning_params = LearningParameters(
    #         gamma=0.9, tabular_case=True,
    #         max_timesteps_per_task=testing_params.num_steps,
    #         lr=1,
    #         clip_rate=0.1,
    #         lam=0.8,
    #         policy_loss_coef=1e-4,
    #         value_loss_coef=1,
    #         entropy_loss_coef=1,
    #         n_updates=4,
    #         learning_starts=1,
    #         buffer_size=100,
    #         batch_size=1  # mini batch size
    #     )

    tester, curriculum = get_tester_curriculum(learning_params, testing_params, experiment, use_rs, use_wandb)

    print("Office World ----------")
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)

    return tester, curriculum


def get_params_water_world(alg_name, experiment, use_rs, use_wandb):
    step_unit = 1  # 1000
    testing_params = TestingParameters(
        test=True,
        test_freq=10 * step_unit,
        num_steps=600
    )

    if alg_name in ["qrm", "qrm-rs"]:
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
    if alg_name in ["pporm", "pporm-rs"]:
        learning_params = LearningParameters(
            lr=1e-5,  # 5e-5 seems to be better than 1e-4
            gamma=0.9,
            max_timesteps_per_task=testing_params.num_steps,
            buffer_size=testing_params.num_steps,
            batch_size=64,  # mini batch size
            print_freq=step_unit,
            clip_rate=0.1,
            lam=0.8,
            n_updates=10,
            policy_loss_coef=1.0,
            value_loss_coef=1.0,
            entropy_loss_coef=0.01,
            learning_starts=1,
            tabular_case=False,
            use_random_maps=False,
            num_hidden_layers=6,
            num_neurons=64
        )

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment, use_rs, use_wandb)

    # Setting the curriculum learner
    curriculum = MultiTaskCurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = 300
    curriculum.total_steps = 2000 * step_unit
    curriculum.min_steps = 1

    print("Water World ----------")
    # print("lr:", learning_params.lr)
    # print("batch_size:", learning_params.batch_size)
    # print("num_hidden_layers:", learning_params.num_hidden_layers)
    # print("target_network_update_freq:", learning_params.target_network_update_freq)
    # print("TRAIN gamma:", learning_params.gamma)
    # print("Total steps:", curriculum.total_steps)
    # print("tabular_case:", learning_params.tabular_case)
    # print("use_double_dqn:", learning_params.use_double_dqn)
    # print("prioritized_replay:", learning_params.prioritized_replay)
    # print("use_random_maps:", learning_params.use_random_maps)

    return tester, curriculum
