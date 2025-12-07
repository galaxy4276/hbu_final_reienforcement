"""
DQN configuration for Atari Breakout-v5 with CNN architecture.
Includes hyperparameters and training settings for expert policy training.
Now supports both legacy ModelV2 and new RLModule APIs.

SageMaker compatible version - uses relative imports.
"""
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from utils.cnn_models import (
    AtariCNN, DuelingAtariCNN,
    get_rl_module_class
)


def register_custom_models():
    """Register custom CNN models with RLlib."""
    ModelCatalog.register_custom_model("atari_cnn", AtariCNN)
    ModelCatalog.register_custom_model("atari_dueling_cnn", DuelingAtariCNN)


def get_dqn_config(
    env_name: str = "ALE/Breakout-v5",
    model_type: str = "cnn",
    framework: str = "torch",
    num_gpus: int = 0,
    num_workers: int = 4,
    train_batch_size: int = 32,
    target_network_update_freq: int = 1000,
    exploration_config: dict = None,
    use_dueling: bool = False,
    double_q: bool = True,
    prioritized_replay: bool = False,
    buffer_size: int = 1000000,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    num_steps_sampled_before_learning_starts: int = 50000,
    min_iter_time_s: int = 60,
    use_new_api_stack: bool = True,
) -> DQNConfig:
    """
    Get DQN configuration for Atari Breakout.

    Args:
        env_name: Environment name
        model_type: Type of CNN model ("cnn" or "dueling")
        framework: Deep learning framework ("torch" or "tf")
        num_gpus: Number of GPUs to use
        num_workers: Number of rollout workers
        train_batch_size: Training batch size
        target_network_update_freq: Target network update frequency
        exploration_config: Exploration configuration
        use_dueling: Whether to use dueling networks
        double_q: Whether to use double Q-learning
        prioritized_replay: Whether to use prioritized replay
        buffer_size: Replay buffer size
        learning_rate: Learning rate
        gamma: Discount factor
        num_steps_sampled_before_learning_starts: Start learning after this many steps
        min_iter_time_s: Minimum time per iteration in seconds
        use_new_api_stack: Whether to use new RLModule API (True) or legacy ModelV2 API (False)

    Returns:
        Configured DQNConfig
    """
    if exploration_config is None:
        exploration_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 1000000,
        }

    config = (
        DQNConfig()
        .environment(
            env=env_name,
            env_config={
                "frameskip": 4,
                "repeat_action_probability": 0.25,
            },
        )
        .api_stack(
            enable_rl_module_and_learner=use_new_api_stack,
            enable_env_runner_and_connector_v2=use_new_api_stack,
        )
    )

    if use_new_api_stack:
        if model_type == "dueling" or use_dueling:
            model_type_for_api = "dueling"
        else:
            model_type_for_api = "cnn"

        rl_module_class = get_rl_module_class(model_type_for_api)

        config = config.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=rl_module_class,
            )
        )

        config = config.training(
            lr=learning_rate,
            gamma=gamma,
            train_batch_size=train_batch_size,
            target_network_update_freq=target_network_update_freq,
            double_q=double_q,
            dueling=use_dueling,
            num_steps_sampled_before_learning_starts=num_steps_sampled_before_learning_starts,
            replay_buffer_config={
                'type': 'MultiAgentReplayBuffer',
            }
        )
    else:
        register_custom_models()

        if model_type == "dueling" or use_dueling:
            custom_model = "atari_dueling_cnn"
            model_config = {"use_dueling": True}
        else:
            custom_model = "atari_cnn"
            model_config = {}

        config = config.training(
            lr=learning_rate,
            gamma=gamma,
            train_batch_size=train_batch_size,
            target_network_update_freq=target_network_update_freq,
            model={
                "custom_model": custom_model,
                "custom_model_config": model_config,
                "conv_filters": None,
                "fcnet_hiddens": None,
                "post_fcnet_hiddens": None,
            },
            double_q=double_q,
            dueling=use_dueling,
            num_steps_sampled_before_learning_starts=num_steps_sampled_before_learning_starts,
            replay_buffer_config={
                'type': 'MultiAgentReplayBuffer',
            }
        )

    config = (
        config
        .env_runners(
            explore=True,
            exploration_config=exploration_config,
            num_env_runners=num_workers,
            num_cpus_per_env_runner=1,
            rollout_fragment_length="auto",
            create_env_on_local_worker=True,
        )
        .resources(
            num_gpus=num_gpus,
        )
        .framework(framework)
        .offline_data(
            output_write_episodes=False,
        )
    )

    return config


def get_expert_training_config(
    stop_criteria: dict = None,
    evaluation_config: dict = None,
) -> dict:
    """
    Get expert training configuration with evaluation setup.
    """
    if stop_criteria is None:
        stop_criteria = {
            "timesteps_total": 10000000,
            "evaluation/episode_reward_mean": 300.0,
            "training_iteration": 5000,
        }

    if evaluation_config is None:
        evaluation_config = {
            "evaluation_interval": 20,
            "evaluation_duration": 10,
            "evaluation_parallel_to_training": True,
            "evaluation_num_workers": 2,
            "evaluation_config": {
                "explore": False,
            },
        }

    return {
        "stop": stop_criteria,
        "evaluation": evaluation_config,
        "checkpoint_at_end": True,
        "keep_checkpoints_num": 5,
        "checkpoint_score_attribute": "evaluation/episode_reward_mean",
        "checkpoint_score_order": "max",
    }


def get_full_config(
    env_name: str = "ALE/Breakout-v5",
    model_type: str = "cnn",
    use_dueling: bool = False,
    num_gpus: int = 0,
    resume: bool = False,
    checkpoint_path: str = None,
    use_new_api_stack: bool = True,
) -> tuple:
    """
    Get full DQN configuration for expert training.
    """
    config = get_dqn_config(
        env_name=env_name,
        model_type=model_type,
        use_dueling=use_dueling,
        num_gpus=num_gpus,
        num_workers=4,
        train_batch_size=32,
        learning_rate=2.5e-4,
        buffer_size=1000000,
        use_new_api_stack=use_new_api_stack,
    )

    expert_config = get_expert_training_config()

    restore_config = {}
    if resume and checkpoint_path:
        restore_config["restart"] = True

    return config, expert_config, restore_config
