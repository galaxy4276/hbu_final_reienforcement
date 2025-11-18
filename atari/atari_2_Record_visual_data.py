"""
Stage 2: Record Expert Data from DQN Policy for Atari Breakout-v5

This script loads the trained DQN expert checkpoint and records expert episodes
to local storage for offline behavior cloning. The data is saved as RLlib
Episode objects in Parquet format.

IMPORTANT: Update the best_checkpoint path before running this script.

Usage:
    python atari/atari_2_Record_visual_data.py

Expected output:
    - Parquet files in docs_rllib_offline_pretrain_ppo/ale-breakout-v5/
    - ~500 expert episodes with visual observations and actions
"""

import os
import sys
import argparse
import time
import pandas as pd
from pathlib import Path

import ray
import gymnasium as gym
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env
from ray.rllib.offline import JsonReader, JsonWriter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from atari.configs.dqn_breakout_config import register_custom_models
from atari.utils.preprocess import FrameStack, validate_observation_shape, preprocess_observation


def create_atari_env(env_config: dict = None):
    """
    Create Atari environment with frame stacking wrapper.
    """
    if env_config is None:
        env_config = {}

    # Create base environment
    env = gym.make("ALE/Breakout-v5", **env_config)

    # Apply frame stacking wrapper
    env = FrameStack(env, stack_size=4)

    return env


def validate_expert_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that the expert checkpoint exists and can be loaded.

    Args:
        checkpoint_path: Path to the expert checkpoint

    Returns:
        True if checkpoint is valid, False otherwise
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    try:
        print(f"🔍 Validating expert checkpoint: {checkpoint_path}")

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # Register environment and models
        register_env("atari_breakout", create_atari_env)
        register_custom_models()

        # Load trainer to validate checkpoint
        trainer = DQN.from_checkpoint(checkpoint_path)

        # Get training stats
        training_iteration = trainer.training_iteration()
        episode_reward_mean = trainer.evaluate()["evaluation"]["episode_reward_mean"]

        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Training Iteration: {training_iteration}")
        print(f"   Evaluation Reward Mean: {episode_reward_mean:.2f}")

        # Shutdown temporary Ray instance
        if ray.is_initialized():
            ray.shutdown()

        return True

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False


def record_expert_episodes(
    checkpoint_path: str,
    output_dir: str,
    num_episodes: int = 500,
    num_workers: int = 2,
    episodes_per_worker: int = 50,
    max_file_size: int = 5,  # Reduced for visual data
) -> str:
    """
    Record expert episodes using the trained DQN policy.

    Args:
        checkpoint_path: Path to the expert checkpoint
        output_dir: Directory to save recorded episodes
        num_episodes: Total number of episodes to record
        num_workers: Number of parallel workers
        episodes_per_worker: Episodes per worker per evaluation iteration
        max_file_size: Maximum rows per output Parquet file

    Returns:
        Path to the output directory
    """
    print(f"🎮 Recording expert episodes...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output Directory: {output_dir}")
    print(f"   Episodes to Record: {num_episodes}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Register environment and models
    register_env("atari_breakout", create_atari_env)
    register_custom_models()

    # Load trained trainer
    print(f"📦 Loading expert policy...")
    trainer = DQN.from_checkpoint(checkpoint_path)

    # Configure trainer for evaluation (no training)
    trainer.config.train_batch_size = 0  # Disable training
    trainer.config.training(num_sgd_iter=0)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Configure episode recording
    episodes_per_iteration = min(episodes_per_worker, num_episodes // num_workers)
    num_iterations = (num_episodes + episodes_per_iteration - 1) // episodes_per_iteration

    print(f"📊 Recording Configuration:")
    print(f"   Episodes per Iteration: {episodes_per_iteration}")
    print(f"   Number of Iterations: {num_iterations}")
    print(f"   Max Rows per File: {max_file_size}")

    episode_count = 0
    start_time = time.time()

    try:
        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")

            # Evaluate the policy and record episodes
            eval_result = trainer.evaluate(
                evaluation_config={
                    "explore": False,  # Use greedy policy
                    "evaluation_duration": episodes_per_iteration,
                    "evaluation_num_workers": num_workers,
                    "evaluation_parallel_to_training": True,
                    "output_write_episodes": True,  # Record episodes
                    "output_config": {
                        "output_format": "parquet",
                        "output_dir": output_dir,
                        "output_max_file_size": max_file_size,
                    },
                }
            )

            # Update episode count
            episode_count += episodes_per_iteration

            # Print progress
            elapsed_time = time.time() - start_time
            reward_mean = eval_result["evaluation"]["episode_reward_mean"]
            reward_max = eval_result["evaluation"]["episode_reward_max"]
            reward_min = eval_result["evaluation"]["episode_reward_min"]

            print(f"   Episodes Recorded: {episode_count}/{num_episodes}")
            print(f"   Reward Mean: {reward_mean:.2f}")
            print(f"   Reward Range: [{reward_min:.2f}, {reward_max:.2f}]")
            print(f"   Time Elapsed: {elapsed_time:.1f}s")

            # Check if we've recorded enough episodes
            if episode_count >= num_episodes:
                print(f"\n✅ Recording completed! Total episodes: {episode_count}")
                break

    except KeyboardInterrupt:
        print(f"\n⚠️  Recording interrupted by user. Episodes recorded: {episode_count}")

    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

    # Analyze recorded data
    analyze_recorded_data(output_dir)

    return output_dir


def analyze_recorded_data(data_dir: str):
    """
    Analyze the recorded expert data.

    Args:
        data_dir: Directory containing recorded episodes
    """
    print(f"\n📈 Analyzing recorded data in: {data_dir}")

    # List all Parquet files
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))

    if not parquet_files:
        print("⚠️  No Parquet files found!")
        return

    print(f"📄 Found {len(parquet_files)} Parquet files")

    total_episodes = 0
    total_steps = 0
    reward_stats = []

    # Analyze each file
    for i, file_path in enumerate(parquet_files[:10]):  # Limit to first 10 files for speed
        try:
            df = pd.read_parquet(file_path)

            if "episode_reward" in df.columns:
                episodes_in_file = len(df["episode_id"].unique())
                total_episodes += episodes_in_file
                total_steps += len(df)

                rewards = df.groupby("episode_id")["episode_reward"].last()
                reward_stats.extend(rewards.tolist())

                print(f"   File {i+1}: {episodes_in_file} episodes, {len(df)} steps")

        except Exception as e:
            print(f"   ⚠️  Error reading {os.path.basename(file_path)}: {e}")

    # Print summary statistics
    if reward_stats:
        print(f"\n📊 Recording Summary:")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Steps: {total_steps:,}")
        print(f"   Steps per Episode: {total_steps / total_episodes:.1f}")
        print(f"   Reward Statistics:")
        print(f"     Mean: {sum(reward_stats) / len(reward_stats):.2f}")
        print(f"     Max: {max(reward_stats):.2f}")
        print(f"     Min: {min(reward_stats):.2f}")
        print(f"     Std: {(sum((r - sum(reward_stats)/len(reward_stats))**2 for r in reward_stats) / len(reward_stats))**0.5:.2f}")

    print(f"✅ Data analysis completed!")


def create_sample_episodic_reader(data_dir: str):
    """
    Create a sample script to read and analyze recorded episodes.

    Args:
        data_dir: Directory containing recorded episodes
    """
    sample_script = f"""
# Sample script to read and analyze recorded expert episodes
# Save this as a separate file and run to explore the data

import ray
from ray.rllib.offline import JsonReader, JsonWriter

# Initialize Ray
ray.init()

# Create reader for recorded data
input = "{data_dir}/output*.json"
reader = JsonReader(input)

print("Reading recorded episodes...")

# Read all episodes
episodes = []
for episode in reader.read_all_episodes():
    episodes.append(episode)
    print(f"Episode {{len(episodes)}}: {{episode.episode_id}}")
    print(f"  Length: {{len(episode)}} steps")
    print(f"  Total Reward: {{episode.total_reward}}")
    print(f"  Observations shape: {{episode.obs.shape}}")
    print(f"  Actions shape: {{episode.actions.shape}}")
    print(f"  Rewards shape: {{episode.rewards.shape}}")
    print("  First few actions:", episode.actions[:10])
    print("  First few rewards:", episode.rewards[:10])
    print()

    if len(episodes) >= 5:  # Show first 5 episodes
        break

print(f"\\nTotal episodes read: {{len(episodes)}}")

ray.shutdown()
"""

    script_path = os.path.join(data_dir, "read_episodes_sample.py")
    with open(script_path, "w") as f:
        f.write(sample_script)

    print(f"📝 Sample analysis script created: {script_path}")


def main():
    parser = argparse.ArgumentParser(description="Record expert data from DQN policy for Atari Breakout-v5")
    parser.add_argument("--checkpoint-path", type=str, required=False,
                       help="Path to expert checkpoint (required if not set in script)")
    parser.add_argument("--output-dir", type=str,
                       default="docs_rllib_offline_pretrain_ppo/ale-breakout-v5",
                       help="Output directory for recorded episodes")
    parser.add_argument("--num-episodes", type=int, default=500,
                       help="Number of episodes to record")
    parser.add_argument("--num-workers", type=int, default=2,
                       help="Number of parallel workers")
    parser.add_argument("--episodes-per-worker", type=int, default=50,
                       help="Episodes per worker per iteration")
    parser.add_argument("--max-file-size", type=int, default=5,
                       help="Maximum rows per output Parquet file")

    args = parser.parse_args()

    # IMPORTANT: UPDATE THIS PATH with your actual checkpoint
    # This path comes from Stage 1 training output
    best_checkpoint = None  # UPDATE THIS LINE with actual path

    # Use checkpoint from command line or script
    checkpoint_path = args.checkpoint_path or best_checkpoint

    if not checkpoint_path:
        print("❌ No checkpoint path provided!")
        print("Please either:")
        print("1. Use --checkpoint-path argument")
        print("2. Update best_checkpoint variable in this script")
        print(f"\\nThe checkpoint path should look like:")
        print(f"~/ray_results/dqn_breakout_expert/DQN_ale-breakout-v5_*/checkpoint_XXXX")
        sys.exit(1)

    # Validate checkpoint
    if not validate_expert_checkpoint(os.path.expanduser(checkpoint_path)):
        sys.exit(1)

    print(f"\n🎮 Starting expert data recording...")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🎯 Episodes to Record: {args.num_episodes}")
    print(f"⚡ Workers: {args.num_workers} x {args.episodes_per_worker} episodes each")

    # Record expert episodes
    output_dir = record_expert_episodes(
        checkpoint_path=os.path.expanduser(checkpoint_path),
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        num_workers=args.num_workers,
        episodes_per_worker=args.episodes_per_worker,
        max_file_size=args.max_file_size,
    )

    # Create sample analysis script
    create_sample_episodic_reader(output_dir)

    print(f"\n✅ Stage 2 completed successfully!")
    print(f"📁 Recorded data saved to: {output_dir}")
    print(f"\n📝 FOR STAGE 3: Update data path in atari_3_Offline_behavior_cloning.py:")
    print(f"   input_data_dir = \"{output_dir}\"")


if __name__ == "__main__":
    main()