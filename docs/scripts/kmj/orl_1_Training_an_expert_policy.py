from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray import tune
from pathlib import Path

save_dir = Path("/home/user/hbu_final_reinforcement/results")
    
# Configure the PPO algorithm.
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .framework("torch")
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        num_epochs=20,
        train_batch_size=8192,
        minibatch_size=1024,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.001,
        grad_clip=0.5,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[512, 512, 256],
            fcnet_activation="tanh",
            vf_share_layers=False,
            free_log_std=True,
        )
    )
    .env_runners(
        batch_mode="complete_episodes",
        num_env_runners=5,
        num_envs_per_env_runner=4,
        observation_filter="MeanStdFilter"
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_parallel_to_training=True,
        evaluation_duration=5,
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
        name="HalfCheetah_expert",
        storage_path=str(save_dir),
        stop={metric: 4000},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True,
        ),
        verbose=2,
    ),
)
results = tuner.fit()

# Store the best checkpoint to use it later for recording
# an expert policy.
best_checkpoint = (
    results
    .get_best_result(
        metric=metric,
        mode="max"
    )
    .checkpoint.path
)

print(f"Best checkpoint path: {best_checkpoint}")