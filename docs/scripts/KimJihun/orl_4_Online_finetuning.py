from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
import os
import glob
import shutil

# Base directory for expert results
expert_results_dir = os.path.expanduser(
    "~/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_35463_00000_0_2025-11-27_06-16-43"
)
# Directory for finetuning results (we might want to continue in the same dir or a new one)
# Since we are resuming, we can just continue saving to the new dir, but restore from the old one.
finetuning_dir = os.path.expanduser("~/ray_results/halfcheetah_online_finetuning")


def get_latest_checkpoint(base_dir):
    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(base_dir, "checkpoint_*"))
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {base_dir}")

    # Sort by modification time (latest first)
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint


print(f"Searching for checkpoints in {expert_results_dir}...")
try:
    checkpoint_to_restore = get_latest_checkpoint(expert_results_dir)
    print(f"Found latest checkpoint: {checkpoint_to_restore}")
except Exception as e:
    print(f"Error finding checkpoint: {e}")
    exit(1)

# Configure the PPO algorithm matching orl_1 exactly
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .framework("torch")
    .training(
        # Learning Parameters - Tuned for 5000+ Score
        lr=[[0, 3e-4], [10_000_000, 0.0]],  # Linear decay to 0
        train_batch_size=8192,
        minibatch_size=1024,
        num_epochs=20,
        # PPO Standard Parameters
        clip_param=0.2,
        vf_loss_coeff=1.0,  # Increased for better value estimation
        entropy_coeff=0.001,  # Slight exploration
        grad_clip=0.8,  # Relaxed clipping for larger model
        gamma=0.99,
        lambda_=0.95,
        # Model Parameters - Deep & Wide
        model={
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
            "free_log_std": True,
        },
    )
    .env_runners(
        num_env_runners=5,
        num_envs_per_env_runner=4,  # Total 20 envs
        observation_filter="MeanStdFilter",
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
        evaluation_parallel_to_training=True,
        evaluation_duration_unit="episodes",
    )
)

print("Building algorithm...")
algo = config.build()

# Strategy:
# 1. Try to resume from a previous FINETUNING checkpoint (full state restore).
# 2. If no finetuning checkpoint, load weights from EXPERT checkpoint (fresh optimizer).
#    This avoids the 'beta1 as Tensor' error caused by incompatible optimizer state in the expert checkpoint.

finetuning_checkpoint = None
try:
    finetuning_checkpoint = get_latest_checkpoint(finetuning_dir)
    print(f"Found existing finetuning checkpoint: {finetuning_checkpoint}")
except (ValueError, FileNotFoundError):
    print("No existing finetuning checkpoint found.")

if finetuning_checkpoint:
    print(f"Resuming training from finetuning checkpoint: {finetuning_checkpoint}")
    try:
        algo.restore(finetuning_checkpoint)
        print("Successfully restored from finetuning checkpoint.")
    except Exception as e:
        print(f"Failed to restore finetuning checkpoint: {e}")
        print("Will attempt to start fresh from expert weights.")
        finetuning_checkpoint = None

if not finetuning_checkpoint:
    print(f"Loading expert checkpoint: {checkpoint_to_restore}")
    try:
        # Standard restore to get weights AND MeanStdFilter state
        algo.restore(checkpoint_to_restore)
        print("Successfully restored checkpoint (weights + filter + optimizer).")

        # PATCH: Fix optimizer state (betas as Tensors) which causes errors in newer PyTorch versions
        def fix_optimizer_state(learner):
            import torch
            import torch.optim as optim

            print(f"DEBUG: Inspecting learner {type(learner)}")
            print(f"DEBUG: Learner keys: {learner.__dict__.keys()}")

            optimizers = []

            # Recursive search for optimizers in the learner object
            visited = set()

            def find_optimizers(obj, depth=0):
                if depth > 3:
                    return
                if id(obj) in visited:
                    return
                visited.add(id(obj))

                if isinstance(obj, optim.Optimizer):
                    optimizers.append(obj)
                    return

                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        find_optimizers(item, depth + 1)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        find_optimizers(v, depth + 1)
                elif hasattr(obj, "__dict__"):
                    for k, v in obj.__dict__.items():
                        # Skip private ray attributes to avoid recursion hell
                        if k.startswith("_ray"):
                            continue
                        find_optimizers(v, depth + 1)

            find_optimizers(learner)

            unique_optimizers = list(set(optimizers))
            print(
                f"DEBUG: Found {len(unique_optimizers)} unique optimizers via recursive search."
            )

            fixed_count = 0
            for i, opt in enumerate(unique_optimizers):
                print(f"DEBUG: Checking optimizer {i}: {type(opt)}")
                for group_idx, group in enumerate(opt.param_groups):
                    if "betas" in group:
                        b1, b2 = group["betas"]
                        updated = False
                        if isinstance(b1, torch.Tensor):
                            b1 = b1.item()
                            updated = True
                        if isinstance(b2, torch.Tensor):
                            b2 = b2.item()
                            updated = True

                        if updated:
                            group["betas"] = (b1, b2)
                            fixed_count += 1
                            print(
                                f"DEBUG: FIXED betas in optimizer {i} group {group_idx}"
                            )
            return fixed_count

        print("Applying optimizer patch to fix tensor betas...")
        results = algo.learner_group.foreach_learner(fix_optimizer_state)
        print(f"Optimizer patch results: {results}")

    except Exception as e:
        print(f"Failed to restore expert checkpoint: {e}")
        print("CRITICAL: Could not load expert checkpoint.")
        exit(1)


print("Starting training loop...")

# Manual training loop
# We'll run for a large number of iterations, but stop if we hit the target
for i in range(10000):
    result = algo.train()

    # Extract and print relevant metrics
    mean_return = None
    total_steps = result["num_env_steps_sampled_lifetime"]

    if (
        EVALUATION_RESULTS in result
        and ENV_RUNNER_RESULTS in result[EVALUATION_RESULTS]
        and EPISODE_RETURN_MEAN in result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS]
    ):
        mean_return = result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
            EPISODE_RETURN_MEAN
        ]
        print(
            f"Iter: {i} | Eval Mean Return: {mean_return:.2f} | Total Env Steps: {total_steps}"
        )

    elif (
        ENV_RUNNER_RESULTS in result
        and EPISODE_RETURN_MEAN in result[ENV_RUNNER_RESULTS]
    ):
        mean_return = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        print(
            f"Iter: {i} | Train Mean Return: {mean_return:.2f} | Total Env Steps: {total_steps}"
        )
    else:
        print(f"Iter: {i} | Metrics not available yet")

    if mean_return is not None and mean_return > 5000:
        print("Target reached! Stopping.")
        break

    # Save checkpoint occasionally
    if i % 20 == 0:
        save_dir = algo.save(checkpoint_dir=finetuning_dir)
        print(f"Checkpoint saved at {save_dir}")

algo.stop()
