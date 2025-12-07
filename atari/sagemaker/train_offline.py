"""
Stage 3: Offline Behavior Cloning Training on Expert Data on AWS SageMaker

This script loads the recorded expert episodes from Stage 2 and trains a new agent
using offline behavior cloning (supervised learning).

Usage (SageMaker):
    Entry point for SageMaker training job

Usage (Local):
    python train_offline.py --input-data-dir /path/to/expert_data
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from ray.tune.registry import register_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_sagemaker_env():
    return os.environ.get('SM_MODEL_DIR') is not None


def get_sagemaker_paths():
    if is_sagemaker_env():
        return {
            'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
            'input_dir': os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'),
            'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
            'tensorboard_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data') + '/tensorboard',
        }
    else:
        home = os.path.expanduser('~')
        base_dir = os.path.join(home, 'ray_results', 'atari_bc_offline')
        return {
            'model_dir': os.path.join(base_dir, 'model'),
            'input_dir': os.path.join(home, 'ray_results', 'atari_record_data', 'expert_data'),
            'output_dir': os.path.join(base_dir, 'output'),
            'tensorboard_dir': os.path.join(base_dir, 'tensorboard'),
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


class AtariCNN(nn.Module):
    def __init__(self, input_channels=4, action_dim=4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, action_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class ExpertDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.observations = []
        self.actions = []

        parquet_files = sorted(Path(data_dir).glob("*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files")

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                obs = np.frombuffer(row['obs'], dtype=np.float16).reshape(84, 84, 4)
                self.observations.append(obs.astype(np.float32))
                self.actions.append(row['action'])

                if max_samples and len(self.observations) >= max_samples:
                    break
            if max_samples and len(self.observations) >= max_samples:
                break

        logger.info(f"Loaded {len(self.observations)} samples")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx])
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, action


def evaluate_policy(model, num_episodes=10, device='cuda'):
    model.eval()
    env = create_atari_env()

    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    env.close()
    model.train()
    return np.mean(rewards), np.std(rewards), np.max(rewards)


def train_bc(data_dir, output_dir, tensorboard_dir, args):
    logger.info("=" * 60)
    logger.info("Behavior Cloning Training - SageMaker Edition")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    setup_atari_roms()

    logger.info("Loading expert data...")
    dataset = ExpertDataset(data_dir, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = AtariCNN(input_channels=4, action_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Dataset size: {len(dataset)}")

    start_time = time.time()
    best_reward = float('-inf')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for obs, actions in dataloader:
            obs = obs.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = model(obs)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)

        writer.add_scalar('Training/Loss', avg_loss, epoch)
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.eval_interval == 0:
            mean_reward, std_reward, max_reward = evaluate_policy(
                model, num_episodes=10, device=device
            )

            writer.add_scalar('Evaluation/Mean_Reward', mean_reward, epoch)
            writer.add_scalar('Evaluation/Max_Reward', max_reward, epoch)
            writer.add_scalar('Evaluation/Std_Reward', std_reward, epoch)

            training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'max_reward': max_reward,
            })

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Reward: {mean_reward:.1f} +/- {std_reward:.1f} | "
                f"Max: {max_reward:.1f}"
            )

            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                logger.info(f"New best model saved! Reward: {best_reward:.1f}")
        else:
            logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))

    logger.info("Final evaluation...")
    mean_reward, std_reward, max_reward = evaluate_policy(
        model, num_episodes=50, device=device
    )

    summary = {
        'total_epochs': args.epochs,
        'final_loss': avg_loss,
        'best_reward': best_reward,
        'final_mean_reward': mean_reward,
        'final_std_reward': std_reward,
        'final_max_reward': max_reward,
        'training_time_seconds': time.time() - start_time,
    }

    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    writer.add_hparams(
        {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
        },
        {
            'hparam/best_reward': best_reward,
            'hparam/final_reward': mean_reward,
            'hparam/final_loss': avg_loss,
        }
    )
    writer.close()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total epochs: {args.epochs}")
    logger.info(f"  Best reward: {best_reward:.2f}")
    logger.info(f"  Final reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info(f"  Training time: {(time.time() - start_time)/60:.1f} min")
    logger.info(f"  TensorBoard logs: {tensorboard_dir}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train BC on Expert Data")
    parser.add_argument("--input-data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    paths = get_sagemaker_paths()

    if is_sagemaker_env():
        data_dir = args.input_data_dir or paths['input_dir']
        output_dir = args.output_dir or paths['model_dir']
        tensorboard_dir = paths['tensorboard_dir']

        hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
            for key, value in hyperparams.items():
                key_name = key.replace('-', '_')
                if hasattr(args, key_name):
                    current_type = type(getattr(args, key_name))
                    if current_type == bool:
                        setattr(args, key_name, value.lower() == 'true')
                    elif current_type != type(None):
                        setattr(args, key_name, current_type(value))
    else:
        data_dir = args.input_data_dir or paths['input_dir']
        output_dir = args.output_dir or paths['model_dir']
        tensorboard_dir = paths['tensorboard_dir']

        if not data_dir or not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            logger.error("Run Stage 2 (record_data.py) first or specify --input-data-dir")
            sys.exit(1)

    train_bc(data_dir, output_dir, tensorboard_dir, args)


if __name__ == "__main__":
    main()
