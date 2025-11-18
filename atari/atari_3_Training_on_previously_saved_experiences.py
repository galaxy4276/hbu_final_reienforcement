"""
Stage 3: Offline Behavior Cloning Training on Expert Data

This script loads the recorded expert episodes from Stage 2 and trains a new agent
using offline behavior cloning (supervised learning). The agent learns to imitate
the expert's behavior without interacting with the environment.

IMPORTANT: Update the input_data_dir path with the output from Stage 2.

Usage:
    python atari/atari_3_Training_on_previously_saved_experiences.py

Expected output:
    - Trained behavior cloning checkpoint
    - Performance comparison with expert
"""

import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path

import ray
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.bc import BC
from ray.tune.registry import register_env

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from atari.configs.dqn_breakout_config import register_custom_models
from atari.utils.preprocess import FrameStack, validate_observation_shape
from atari.utils.cnn_models import BCCNN, get_model


def create_atari_env(env_config: dict = None):
    """
    Create Atari environment with frame stacking wrapper for evaluation.
    """
    if env_config is None:
        env_config = {}

    # Create base environment
    env = gym.make("ALE/Breakout-v5", **env_config)

    # Apply frame stacking wrapper
    env = FrameStack(env, stack_size=4)

    return env


def validate_input_data(data_dir: str) -> bool:
    """
    Validate that the input data directory exists and contains recorded episodes.

    Args:
        data_dir: Directory containing recorded episodes

    Returns:
        True if data is valid, False otherwise
    """
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False

    # Look for Parquet files
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))

    if not parquet_files:
        print(f"❌ No Parquet files found in: {data_dir}")
        return False

    print(f"✅ Found {len(parquet_files)} Parquet files")
    print(f"📁 Data directory: {data_dir}")

    # Try to read one file to validate format
    try:
        import pandas as pd
        sample_df = pd.read_parquet(parquet_files[0])
        required_columns = ["obs", "actions", "rewards", "dones", "episode_id"]

        missing_columns = [col for col in required_columns if col not in sample_df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        print(f"✅ Data format validation successful")
        print(f"   Sample episodes: {len(sample_df['episode_id'].unique())}")
        print(f"   Sample steps: {len(sample_df)}")

        return True

    except Exception as e:
        print(f"❌ Error reading data files: {e}")
        return False


def get_bc_config(
    data_dir: str,
    framework: str = "torch",
    num_gpus: int = 0,
    train_batch_size: int = 1024,
    learning_rate: float = 1e-4,
    model_type: str = "bc",
) -> BC:
    """
    Create BC configuration for offline training.

    Args:
        data_dir: Directory containing recorded episodes
        framework: Deep learning framework
        num_gpus: Number of GPUs to use
        train_batch_size: Training batch size
        learning_rate: Learning rate
        model_type: Type of model to use

    Returns:
        Configured BC algorithm
    """
    # Register custom environment and models
    register_env("atari_breakout", create_atari_env)
    register_custom_models()

    # Create BC algorithm
    config = (
        BC()
        .offline_data(
            input_=(f"{data_dir}/*.parquet"),  # Input data pattern
            input_config={
                "format": "parquet",
                "shuffle_episodes": True,
            },
        )
        .training(
            lr=learning_rate,
            train_batch_size=train_batch_size,
            model={
                "custom_model": "atari_bc_cnn",  # We'll use BCCNN
                "custom_model_config": {},
                "fcnet_hiddens": None,  # Using custom CNN
            },
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_for_driver=1,
        )
        .framework(framework)
        .evaluation(
            evaluation_interval=5,  # Evaluate every 5 iterations
            evaluation_duration=10,  # 10 episodes per evaluation
            evaluation_config={
                "explore": False,  # Greedy evaluation
                "input": f"{data_dir}/*.parquet",  # Use same data for evaluation
            },
            off_policy_estimation_methods={
                "ope": {"is": {"action_space": gym.spaces.Discrete(4)}}
            },
        )
        .reporting(
            metrics_num_episodes_for_smoothing=20,
            min_time_s_per_iteration=30,
        )
    )

    return config


def train_behavior_cloning(
    data_dir: str,
    experiment_name: str = "bc_breakout_offline",
    num_iterations: int = 200,
    num_gpus: int = 0,
    resume: bool = False,
    checkpoint_path: str = None,
) -> str:
    """
    Train behavior cloning model on recorded expert data.

    Args:
        data_dir: Directory containing recorded episodes
        experiment_name: Name for the experiment
        num_iterations: Maximum number of training iterations
        num_gpus: Number of GPUs to use
        resume: Whether to resume from checkpoint
        checkpoint_path: Path to checkpoint (if resuming)

    Returns:
        Path to the final checkpoint
    """
    print(f"🧠 Starting behavior cloning training...")
    print(f"📁 Input Data: {data_dir}")
    print(f"🎯 Experiment Name: {experiment_name}")
    print(f"📚 Max Iterations: {num_iterations}")
    print(f"⚡ GPUs: {num_gpus}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Get BC configuration
    algorithm = get_bc_config(
        data_dir=data_dir,
        framework="torch",
        num_gpus=num_gpus,
        train_batch_size=512,  # Smaller batch for better convergence
        learning_rate=5e-4,    # Higher learning rate for imitation
        model_type="bc",
    )

    # Create trainer
    trainer = BC(config=algorithm)

    # Restore from checkpoint if specified
    if resume and checkpoint_path:
        print(f"🔄 Restoring from checkpoint: {checkpoint_path}")
        trainer.restore(checkpoint_path)

    print(f"\n🚀 Starting offline training...")
    print(f"💾 Checkpoints will be saved to: ~/ray_results/{experiment_name}/")
    print(f"📈 Monitor with: tensorboard --logdir=~/ray_results/{experiment_name}/")

    # Training loop
    start_time = time.time()
    training_metrics = []

    try:
        for iteration in range(num_iterations):
            iteration += 1

            # Train one iteration
            result = trainer.train()

            # Collect metrics
            training_metrics.append({
                "iteration": iteration,
                "training_loss": result.get("policy_loss", 0),
                "episode_reward_mean": result.get("evaluation", {}).get("episode_reward_mean", 0),
                "episode_reward_max": result.get("evaluation", {}).get("episode_reward_max", 0),
                "episode_reward_min": result.get("evaluation", {}).get("episode_reward_min", 0),
                "time_elapsed": time.time() - start_time,
            })

            # Print progress every 5 iterations
            if iteration % 5 == 0:
                elapsed_time = time.time() - start_time
                policy_loss = result.get("policy_loss", 0)
                eval_reward = result.get("evaluation", {}).get("episode_reward_mean", "N/A")

                print(f"\n📊 Iteration {iteration}:")
                print(f"  Training Loss: {policy_loss:.6f}")
                print(f"  Evaluation Reward Mean: {eval_reward}")
                print(f"  Time Elapsed: {elapsed_time:.1f}s")
                print(f"  Samples Trained: {result.get('samples_trained', 0):,}")

                # Save checkpoint every 50 iterations
                if iteration % 50 == 0:
                    checkpoint_dir = trainer.save()
                    print(f"💾 Checkpoint saved: {checkpoint_dir}")

            # Check convergence
            if iteration >= 20:  # After enough iterations
                recent_rewards = [m["episode_reward_mean"] for m in training_metrics[-10:]]
                if len(recent_rewards) >= 10:
                    # Check if performance has stabilized
                    reward_variance = np.var(recent_rewards)
                    if reward_variance < 1.0:  # Very low variance means convergence
                        mean_reward = np.mean(recent_rewards)
                        print(f"\n🎯 Training converged!")
                        print(f"   Final Performance: {mean_reward:.2f}")
                        print(f"   Recent Variance: {reward_variance:.4f}")
                        break

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    finally:
        # Save final checkpoint
        final_checkpoint = trainer.save()
        print(f"\n✅ Training completed!")
        print(f"🎯 Final checkpoint saved: {final_checkpoint}")

        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

    # Plot training progress
    plot_training_results(training_metrics, experiment_name)

    return final_checkpoint


def plot_training_results(metrics: list, experiment_name: str):
    """
    Plot training progress and save to file.

    Args:
        metrics: List of training metrics
        experiment_name: Name of the experiment
    """
    if not metrics:
        print("⚠️  No metrics to plot")
        return

    try:
        iterations = [m["iteration"] for m in metrics]
        losses = [m["training_loss"] for m in metrics]
        rewards = [m["episode_reward_mean"] for m in metrics]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot training loss
        ax1.plot(iterations, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Behavior Cloning Training Loss')
        ax1.grid(True, alpha=0.3)

        # Plot evaluation rewards
        ax2.plot(iterations, rewards, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Episode Reward Mean')
        ax2.set_title('Evaluation Performance')
        ax2.grid(True, alpha=0.3)

        # Add horizontal line for human-level performance (approx. 300)
        ax2.axhline(y=300, color='r', linestyle='--', alpha=0.7, label='Human-level')
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plot_path = f"{experiment_name}_training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 Training plot saved: {plot_path}")

        plt.close()

    except Exception as e:
        print(f"⚠️  Failed to create training plot: {e}")


def evaluate_trained_model(checkpoint_path: str, num_episodes: int = 50):
    """
    Evaluate the trained behavior cloning model.

    Args:
        checkpoint_path: Path to trained checkpoint
        num_episodes: Number of evaluation episodes
    """
    print(f"\n🎯 Evaluating trained model...")
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"🎮 Evaluation Episodes: {num_episodes}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    try:
        # Load trained model
        trainer = BC.from_checkpoint(checkpoint_path)

        # Evaluate model
        eval_result = trainer.evaluate(
            evaluation_config={
                "evaluation_duration": num_episodes,
                "explore": False,  # Greedy evaluation
            }
        )

        # Print evaluation results
        print(f"\n📊 Evaluation Results:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Reward Mean: {eval_result['evaluation']['episode_reward_mean']:.2f}")
        print(f"  Reward Max: {eval_result['evaluation']['episode_reward_max']:.2f}")
        print(f"  Reward Min: {eval_result['evaluation']['episode_reward_min']:.2f}")
        print(f"  Reward Std: {eval_result['evaluation']['episode_reward_std']:.2f}")
        print(f"  Episode Length Mean: {eval_result['evaluation']['episode_length_mean']:.1f}")

        # Performance assessment
        reward_mean = eval_result['evaluation']['episode_reward_mean']
        if reward_mean >= 200:
            print(f"🏆 EXCELLENT: Expert-level performance!")
        elif reward_mean >= 100:
            print(f"✅ GOOD: Strong intermediate performance!")
        elif reward_mean >= 50:
            print(f"🟡 DECENT: Basic learning achieved!")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: Consider more training or better data.")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Train behavior cloning on expert data")
    parser.add_argument("--input-data-dir", type=str,
                       default="docs_rllib_offline_pretrain_ppo/ale-breakout-v5",
                       help="Directory containing recorded expert episodes")
    parser.add_argument("--experiment-name", type=str,
                       default="bc_breakout_offline",
                       help="Name for this experiment")
    parser.add_argument("--num-iterations", type=int, default=200,
                       help="Maximum number of training iterations")
    parser.add_argument("--num-gpus", type=int, default=0,
                       help="Number of GPUs to use")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint-path", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate, don't train")
    parser.add_argument("--num-eval-episodes", type=int, default=50,
                       help="Number of evaluation episodes")

    args = parser.parse_args()

    print(f"🧠 Atari Breakout - Stage 3: Offline Behavior Cloning")

    # Validate input data
    if not validate_input_data(args.input_data_dir):
        sys.exit(1)

    if args.evaluate_only:
        # Only evaluate existing model
        if not args.checkpoint_path:
            print("❌ Checkpoint path required for evaluation only mode")
            sys.exit(1)

        evaluate_trained_model(args.checkpoint_path, args.num_eval_episodes)
    else:
        # Train new model
        final_checkpoint = train_behavior_cloning(
            data_dir=args.input_data_dir,
            experiment_name=args.experiment_name,
            num_iterations=args.num_iterations,
            num_gpus=args.num_gpus,
            resume=args.resume,
            checkpoint_path=args.checkpoint_path,
        )

        # Evaluate final model
        evaluate_trained_model(final_checkpoint, args.num_eval_episodes)

        print(f"\n🎉 Stage 3 completed successfully!")
        print(f"📦 Final checkpoint: {final_checkpoint}")
        print(f"📈 Check results with: tensorboard --logdir=~/ray_results/{args.experiment_name}/")


if __name__ == "__main__":
    main()