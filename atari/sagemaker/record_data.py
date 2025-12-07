"""
Stage 2: Record Expert Data from DQN Policy for Atari Breakout-v5 on AWS SageMaker

This script loads the trained DQN expert checkpoint and records expert episodes
to S3-compatible storage for offline behavior cloning.

Usage (SageMaker):
    Entry point for SageMaker processing job

Usage (Local):
    python record_data.py --checkpoint-path /path/to/checkpoint --num-episodes 500
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path

import ray
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_sagemaker_env():
    return os.environ.get('SM_MODEL_DIR') is not None


def get_sagemaker_paths():
    if is_sagemaker_env():
        return {
            'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
            'input_dir': os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/input/data/model'),
            'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
        }
    else:
        home = os.path.expanduser('~')
        base_dir = os.path.join(home, 'ray_results', 'atari_record_data')
        return {
            'model_dir': os.path.join(base_dir, 'model'),
            'input_dir': os.path.join(home, 'ray_results', 'atari_dqn_expert', 'model'),
            'output_dir': os.path.join(base_dir, 'expert_data'),
        }


def setup_atari_roms():
    logger.info("Setting up Atari ROMs...")
    try:
        import subprocess
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'autorom[accept-rom-license]', '-q'
        ], check=True)
        subprocess.run(['AutoROM', '--accept-license'], check=True)
        logger.info("Atari ROMs installed successfully")
    except Exception as e:
        logger.warning(f"ROM installation warning: {e}")


def preprocess_observation(obs):
    import cv2
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0


class FrameStack:
    def __init__(self, env, stack_size=4):
        self.env = env
        self.stack_size = stack_size
        self.frame_buffer = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = preprocess_observation(obs)
        self.frame_buffer = [processed] * self.stack_size
        return np.stack(self.frame_buffer, axis=-1).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = preprocess_observation(obs)
        self.frame_buffer.append(processed)
        if len(self.frame_buffer) > self.stack_size:
            self.frame_buffer.pop(0)
        stacked = np.stack(self.frame_buffer, axis=-1).astype(np.float32)
        return stacked, reward, terminated, truncated, info


def create_atari_env(env_config=None):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", **(env_config or {}))
    return FrameStack(env, stack_size=4)


def record_episodes(checkpoint_path, output_dir, num_episodes=500, batch_size=50):
    logger.info("=" * 60)
    logger.info("Recording Expert Episodes - SageMaker Edition")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Episodes: {num_episodes}")

    os.makedirs(output_dir, exist_ok=True)

    ray.init(
        num_cpus=4,
        num_gpus=1 if torch.cuda.is_available() else 0,
        ignore_reinit_error=True,
    )

    register_env("atari_breakout", create_atari_env)

    logger.info("Loading expert policy...")
    trainer = DQN.from_checkpoint(checkpoint_path)
    policy = trainer.get_policy()
    logger.info("Expert policy loaded successfully")

    env = create_atari_env()

    all_episodes = []
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()

    for episode_idx in range(num_episodes):
        obs, info = env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }

        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            if obs_tensor.shape[-1] == 4:
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)

            with torch.no_grad():
                action = policy.compute_single_action(obs)[0]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data['observations'].append(obs.astype(np.float16))
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)

            obs = next_obs
            episode_reward += reward
            step_count += 1

        episode_data['episode_id'] = episode_idx
        episode_data['episode_reward'] = episode_reward
        episode_data['episode_length'] = step_count
        all_episodes.append(episode_data)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        if (episode_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(
                f"Episode {episode_idx + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg(10): {avg_reward:.1f} | "
                f"Time: {elapsed:.1f}s"
            )

        if (episode_idx + 1) % batch_size == 0:
            batch_num = (episode_idx + 1) // batch_size
            save_episodes_to_parquet(
                all_episodes[-batch_size:],
                output_dir,
                batch_num
            )

    remaining = len(all_episodes) % batch_size
    if remaining > 0:
        batch_num = (num_episodes // batch_size) + 1
        save_episodes_to_parquet(
            all_episodes[-remaining:],
            output_dir,
            batch_num
        )

    env.close()

    summary = {
        'total_episodes': num_episodes,
        'total_steps': sum(episode_lengths),
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'recording_time_seconds': time.time() - start_time,
    }

    summary_path = os.path.join(output_dir, 'recording_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Recording Complete!")
    logger.info(f"  Total episodes: {num_episodes}")
    logger.info(f"  Total steps: {sum(episode_lengths):,}")
    logger.info(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    logger.info(f"  Max reward: {np.max(episode_rewards):.2f}")
    logger.info(f"  Recording time: {(time.time() - start_time)/60:.1f} min")
    logger.info("=" * 60)

    ray.shutdown()
    return summary


def save_episodes_to_parquet(episodes, output_dir, batch_num):
    rows = []
    for ep in episodes:
        for step_idx in range(len(ep['actions'])):
            rows.append({
                'episode_id': ep['episode_id'],
                'step': step_idx,
                'obs': ep['observations'][step_idx].tobytes(),
                'action': ep['actions'][step_idx],
                'reward': ep['rewards'][step_idx],
                'done': ep['dones'][step_idx],
                'episode_reward': ep['episode_reward'],
            })

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(output_dir, f'episodes_batch_{batch_num:04d}.parquet')
    df.to_parquet(parquet_path, index=False, compression='snappy')
    logger.info(f"Saved batch {batch_num} to {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Record Expert Data for Offline RL")
    parser.add_argument("--checkpoint-path", type=str, required=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)

    args = parser.parse_args()

    if is_sagemaker_env():
        paths = get_sagemaker_paths()
        checkpoint_path = args.checkpoint_path or os.path.join(
            paths['input_dir'], 'best_checkpoint'
        )
        output_dir = args.output_dir or os.path.join(paths['output_dir'], 'expert_data')
    else:
        checkpoint_path = args.checkpoint_path
        output_dir = args.output_dir or './expert_data'

        if not checkpoint_path:
            logger.error("--checkpoint-path required for local execution")
            sys.exit(1)

    setup_atari_roms()
    record_episodes(checkpoint_path, output_dir, args.num_episodes, args.batch_size)


if __name__ == "__main__":
    main()
