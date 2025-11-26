from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray import tune
import os

# Find the latest checkpoint from the expert training
import glob

def get_latest_checkpoint(base_dir):
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "PPO_HalfCheetah-v5_*"))
    if not exp_dirs:
        raise ValueError(f"No experiment directories found in {base_dir}")
    
    # Sort by modification time (latest first)
    latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
    print(f"Found latest experiment directory: {latest_exp_dir}")
    
    # Find the best checkpoint in this directory (assuming we want the one at the end or best metric)
    # Since we don't have the Tuner object here easily to query 'best', we'll take the last checkpoint directory
    # usually named 'checkpoint_000XXX'
    checkpoint_dirs = glob.glob(os.path.join(latest_exp_dir, "checkpoint_*"))
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {latest_exp_dir}")
        
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint

# Base directory for expert results
expert_results_dir = os.path.expanduser("~/ray_results/halfcheetah_expert_ppo")
# Directory for finetuning results
finetuning_dir = os.path.expanduser("~/ray_results/halfcheetah_online_finetuning")

checkpoint_to_restore = None

# 1. Try to find a checkpoint from a previous finetuning run
try:
    print(f"Checking for existing finetuning checkpoints in {finetuning_dir}...")
    checkpoint_to_restore = get_latest_checkpoint(finetuning_dir)
    print(f"Resuming from finetuning checkpoint: {checkpoint_to_restore}")
except (ValueError, FileNotFoundError):
    print("No finetuning checkpoint found. Falling back to expert checkpoint.")
    # 2. If no finetuning checkpoint, load the expert checkpoint
    try:
        checkpoint_to_restore = get_latest_checkpoint(expert_results_dir)
        print(f"Starting fresh from expert checkpoint: {checkpoint_to_restore}")
    except Exception as e:
        print(f"Error finding expert checkpoint: {e}")
        # Fallback
        checkpoint_to_restore = "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_09c7e_00000_0_2025-11-25_15-43-23/" 

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
        # Run 10 SGD minibatch iterations on a batch.
        train_batch_size=2048,
        minibatch_size=64,
        num_epochs=10,
        # Weigh the value function loss smaller than
        # the policy loss.
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.0,
        grad_clip=0.5,
        gamma=0.99,
        lambda_=0.95,
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
            # Share encoder layers between value network
            # and policy.
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

# Try to restore the whole algorithm state
if checkpoint_to_restore:
    try:
        algo.restore(checkpoint_to_restore)
        print(f"Restored state from {checkpoint_to_restore}")
    except Exception as e:
        print(f"Failed to restore state: {e}")
        pass

# Manual training loop
for i in range(100):
    result = algo.train()
    
    # Debug keys if metric is missing
    if ENV_RUNNER_RESULTS in result and EPISODE_RETURN_MEAN not in result[ENV_RUNNER_RESULTS]:
        print(f"Available keys in result[{ENV_RUNNER_RESULTS}]: {result[ENV_RUNNER_RESULTS].keys()}")
    
    # Handle potential missing keys gracefully or debug
    try:
        mean_return = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        print(f"Iteration {i}: {mean_return}")
        
        # Check if evaluation results are available in the training result
        if EVALUATION_RESULTS in result:
            eval_metrics = result[EVALUATION_RESULTS]
            # Depending on the API, it might be nested or direct
            if ENV_RUNNER_RESULTS in eval_metrics:
                eval_return = eval_metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
                print(f"Evaluation: {eval_return}")
            else:
                # Fallback or just print keys if structure is different
                # print(f"Evaluation keys: {eval_metrics.keys()}")
                pass
            
        # Save checkpoint every 5 iterations
        if i % 5 == 0:
            save_dir = algo.save(checkpoint_dir=finetuning_dir)
            print(f"Checkpoint saved at {save_dir}")
            
        if mean_return > 5000:
            print("Target reached!")
            break
    except KeyError as e:
        print(f"KeyError accessing metrics: {e}")
        print(f"Result keys: {result.keys()}")
        if ENV_RUNNER_RESULTS in result:
             print(f"Env Runner Results keys: {result[ENV_RUNNER_RESULTS].keys()}")
        break

algo.stop()
