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
    .environment("CartPole-v1")
    .training(
        lr=0.0003,
        # Run 6 SGD minibatch iterations on a batch.
        num_epochs=6,
        # Weigh the value function loss smaller than
        # the policy loss.
        vf_loss_coeff=0.01,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[32],
            fcnet_activation="linear",
            # Share encoder layers between value network
            # and policy.
            vf_share_layers=True,
        ),
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
            metric: 450.0,
        },
        name="docs_rllib_offline_pretrain_ppo",
        verbose=2,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=1,
            checkpoint_at_end=True,
        ),
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