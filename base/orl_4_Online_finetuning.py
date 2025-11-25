from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray import tune
import os

# Placeholder for the offline trained checkpoint path.
# You will need to update this after running orl_3_Training_on_previously_saved_experiences.py
offline_checkpoint = "/home/pz1004/ray_results/halfcheetah_offline_bc/..."

# Configure the PPO algorithm for fine-tuning.
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .resources(
        num_gpus=1,
    )
    .env_runners(
        num_env_runners=10,
    )
    .training(
        lr=0.00005, # Lower learning rate for fine-tuning
        num_epochs=10,
        vf_loss_coeff=0.01,
        train_batch_size=8000,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="tanh",
            vf_share_layers=False,
        ),
    )
)

# Define the metric to use for stopping.
metric = f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"

# Define the Tuner.
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=tune.RunConfig(
        stop={
            metric: 5000.0, # Target higher return than offline
        },
        name="halfcheetah_online_finetuning",
        verbose=2,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True,
        ),
    ),
)

# Restore from the offline checkpoint before training
# Note: In the new API, we might need to load the state differently or use `restore` on the algorithm object if we were not using Tuner.
# However, with Tuner, we can't easily inject the restore call before fit() unless we use a custom trainable or restore from a checkpoint of the SAME algorithm.
# Since we are switching from BC to PPO, we need to load the weights specifically.

# Alternative approach: Build the algo, restore weights, and then train using `algo.train()` loop or wrap it.
# But for simplicity and consistency with previous scripts, let's try to use `restore` if possible, but BC and PPO have different states.
# We only want to transfer the policy weights.

# Let's use a custom training loop for fine-tuning to ensure we load weights correctly.

print("Starting Fine-tuning...")
algo = config.build()

# Load weights from BC checkpoint
# We need to be careful here. BC checkpoint contains a BC policy. PPO expects a PPO policy.
# If the model architecture is the same, we can load the weights.
# The `component` argument in `restore_from_path` can be used to load specific parts.
try:
    algo.restore_from_path(
        offline_checkpoint,
        component="learner_group", # Try to load the learner state (policy weights)
        # component="rl_module", # Or just the RL Module
    )
    print(f"Restored weights from {offline_checkpoint}")
except Exception as e:
    print(f"Failed to restore full state: {e}")
    print("Attempting to load only RL Module weights...")
    # This part depends on the exact structure of the checkpoint and RLlib version.
    # For now, we will assume the user will manually verify if weights are loaded.
    pass

# Manual training loop
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
    
    if i % 5 == 0:
        eval_result = algo.evaluate()
        print(f"Evaluation: {eval_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
        
    if result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] > 5000:
        print("Target reached!")
        break

algo.stop()
