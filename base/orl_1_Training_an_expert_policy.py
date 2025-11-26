from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray import tune

# Configure the PPO algorithm.
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
    .env_runners(
        num_env_runners=12,
        num_envs_per_env_runner=5,
        observation_filter="MeanStdFilter",
    )
    .training(
        # Learning Parameters
        lr=[[0, 3e-4], [2_000_000, 1e-5]],  # lr=3e-4,
        train_batch_size=4096,
        minibatch_size=512,
        num_epochs=10,
        # PPO Standard Parameters
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.0,
        grad_clip=0.5,
        gamma=0.99,
        lambda_=0.95,
        # Model Parameters
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
        },
    )
    .evaluation(
        evaluation_interval=10,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
    )
)

# Define the metric to use for stopping.
metric = f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
# metric = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"

# Define the Tuner.
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=tune.RunConfig(
        stop={
            metric: 3000.0,  # HalfCheetah expert usually gets > 3000
        },
        name="halfcheetah_expert_ppo",
        verbose=2,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
    ),
)
results = tuner.fit()

# Store the best checkpoint to use it later for recording
# an expert policy.
best_checkpoint = results.get_best_result(metric=metric, mode="max").checkpoint.path

print(f"Best checkpoint path: {best_checkpoint}")
