"""
Stage 1: Train DQN Expert Policy for Atari Breakout-v5

This script trains an expert DQN agent on ALE/Breakout-v5 using CNN architecture.
The trained checkpoint will be used in Stage 2 for data collection.

Usage:
    python atari/atari_1_Training_DQN_expert.py [--resume] [--checkpoint-path PATH]

Arguments:
    --resume: Resume training from checkpoint
    --checkpoint-path: Path to checkpoint to resume from
"""

import os
import sys
import argparse
import time
import tempfile
from pathlib import Path

import ray
import gymnasium as gym
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from atari.configs.dqn_breakout_config import get_full_config
from atari.utils.preprocess import FrameStack, validate_observation_shape


def create_atari_env(env_config: dict = None):
    """
    Create Atari environment with frame stacking wrapper.

    Args:
        env_config: Environment configuration

    Returns:
        Wrapped Atari environment
    """
    if env_config is None:
        env_config = {}

    # Create base environment
    env = gym.make("ALE/Breakout-v5", **env_config)

    # Apply frame stacking wrapper
    env = FrameStack(env, stack_size=4)

    return env


def validate_environment():
    """Validate that the Atari environment is working correctly."""
    print("Validating Atari Breakout environment...")

    try:
        # Create environment
        env = create_atari_env()

        # Test reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        # Validate observation shape
        assert validate_observation_shape(obs), f"Invalid observation shape: {obs.shape}"

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step observation shape: {obs.shape}")
        print(f"Sample reward: {reward}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.action_space}")

        # Validate observation shape after step
        assert validate_observation_shape(obs), f"Invalid observation shape after step: {obs.shape}"

        env.close()
        print("✅ Environment validation successful!")
        return True

    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train DQN expert on Atari Breakout-v5")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint")
    parser.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "dueling"],
                       help="Type of CNN model")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--experiment-name", type=str, default="dqn_breakout_expert",
                       help="Experiment name for TensorBoard")

    args = parser.parse_args()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Register custom environment
    register_env("atari_breakout", create_atari_env)

    # Get configuration
    config, expert_config, restore_config = get_full_config(
        env_name="atari_breakout",
        model_type=args.model_type,
        use_dueling=(args.model_type == "dueling"),
        num_gpus=args.num_gpus,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
    )

    print("🎮 DQN Expert Training Configuration:")
    print(f"  Environment: ALE/Breakout-v5")
    print(f"  Model Type: {args.model_type}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Resume: {args.resume}")
    if args.checkpoint_path:
        print(f"  Checkpoint: {args.checkpoint_path}")

    print("\n📊 Training Parameters:")
    print(f"  Train Batch Size: {config.train_batch_size}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Buffer Size: {config.buffer_size}")
    print(f"  Target Network Update: {config.target_network_update_freq}")
    print(f"  Double Q-Learning: {config.double_q}")
    print(f"  Dueling Networks: {config.dueling}")

    print("\n🎯 Stopping Criteria:")
    for key, value in expert_config["stop"].items():
        print(f"  {key}: {value}")

    # Create trainer
    trainer = DQN(config=config)

    # Restore from checkpoint if specified
    if args.resume and args.checkpoint_path:
        print(f"\n🔄 Restoring from checkpoint: {args.checkpoint_path}")
        trainer.restore(args.checkpoint_path)
    elif args.resume:
        print("⚠️  Resume requested but no checkpoint path provided. Starting fresh training.")

    print(f"\n🚀 Starting DQN expert training...")
    print(f"💾 Checkpoints will be saved to: ~/ray_results/{args.experiment_name}/")
    print(f"📈 Monitor with: tensorboard --logdir=~/ray_results/{args.experiment_name}/")

    # Training loop
    start_time = time.time()
    iteration = 0

    try:
        while True:
            iteration += 1

            # Train one iteration
            result = trainer.train()

            # Print progress every 10 iterations
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time
                timesteps = result.get("timesteps_total", 0)
                episode_reward_mean = result.get("episode_reward_mean", 0)
                evaluation_reward_mean = result.get("evaluation", {}).get("episode_reward_mean", "N/A")

                print(f"\n📊 Iteration {iteration}:")
                print(f"  Timesteps: {timesteps:,}")
                print(f"  Training Reward Mean: {episode_reward_mean:.2f}")
                print(f"  Evaluation Reward Mean: {evaluation_reward_mean}")
                print(f"  Time Elapsed: {elapsed_time:.1f}s")
                print(f"  Loss: {result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {}).get('mean_td_error', 'N/A')}")

                # Save checkpoint every 100 iterations
                if iteration % 100 == 0:
                    checkpoint_dir = trainer.save()
                    print(f"💾 Checkpoint saved: {checkpoint_dir}")

            # Check stopping criteria
            should_stop = False
            for criterion, threshold in expert_config["stop"].items():
                current_value = result.get(criterion)
                if current_value is not None:
                    if criterion == "evaluation/episode_reward_mean":
                        # For evaluation metrics, check if we've reached the threshold
                        if current_value >= threshold:
                            print(f"\n🎯 Stopping criterion met: {criterion} = {current_value:.2f} >= {threshold}")
                            should_stop = True
                    elif criterion == "timesteps_total":
                        if current_value >= threshold:
                            print(f"\n🎯 Stopping criterion met: {criterion} = {current_value:,} >= {threshold:,}")
                            should_stop = True
                    elif criterion == "training_iteration":
                        if iteration >= threshold:
                            print(f"\n🎯 Stopping criterion met: {criterion} = {iteration} >= {threshold}")
                            should_stop = True

            if should_stop:
                break

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    finally:
        # Save final checkpoint
        final_checkpoint = trainer.save()
        print(f"\n✅ Training completed!")
        print(f"🎯 Final checkpoint saved: {final_checkpoint}")

        # Print training summary
        total_time = time.time() - start_time
        final_result = trainer.train()  # Get final metrics

        print(f"\n📈 Training Summary:")
        print(f"  Total Iterations: {iteration}")
        print(f"  Total Timesteps: {final_result.get('timesteps_total', 0):,}")
        print(f"  Final Training Reward: {final_result.get('episode_reward_mean', 0):.2f}")
        print(f"  Final Evaluation Reward: {final_result.get('evaluation', {}).get('episode_reward_mean', 'N/A')}")
        print(f"  Total Training Time: {total_time:.1f}s ({total_time/60:.1f}m)")

        # Shutdown Ray
        ray.shutdown()

        # Update this path for Stage 2
        print(f"\n📝 FOR STAGE 2: Update checkpoint path in atari_2_Record_visual_data.py:")
        print(f"   best_checkpoint = \"{final_checkpoint}\"")


if __name__ == "__main__":
    main()