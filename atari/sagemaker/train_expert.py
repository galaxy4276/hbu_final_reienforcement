"""
Stage 1: Train DQN Expert Policy for Atari Breakout-v5 on AWS SageMaker

This script trains an expert DQN agent on ALE/Breakout-v5 using CNN architecture.
Optimized for SageMaker GPU instances with S3 integration and TensorBoard logging.

Usage (SageMaker):
    Entry point for SageMaker training job

Usage (Local):
    python train_expert.py --num-gpus 1
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
import gymnasium as gym
import ale_py
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_sagemaker_env():
    return os.environ.get('SM_MODEL_DIR') is not None


def get_sagemaker_paths():
    if is_sagemaker_env():
        return {
            'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
            'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
            'checkpoint_dir': os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints'),
            'tensorboard_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data') + '/tensorboard',
        }
    else:
        home = os.path.expanduser('~')
        base_dir = os.path.join(home, 'ray_results', 'atari_dqn_expert')
        return {
            'model_dir': os.path.join(base_dir, 'model'),
            'output_dir': os.path.join(base_dir, 'output'),
            'checkpoint_dir': os.path.join(base_dir, 'checkpoints'),
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
        logger.info("Atari ROMs installed successfully")
    except Exception as e:
        logger.warning(f"ROM installation warning: {e}")


def preprocess_observation(obs):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0


class FrameStack(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frame_buffer = []

        import numpy as np
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(84, 84, stack_size),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        import numpy as np
        obs, info = self.env.reset(**kwargs)
        processed = preprocess_observation(obs)
        self.frame_buffer = [processed] * self.stack_size
        return np.stack(self.frame_buffer, axis=-1).astype(np.float32), info

    def step(self, action):
        import numpy as np
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


def get_dqn_config(num_gpus=1, num_workers=4):
    config = (
        DQNConfig()
        .environment(
            env="atari_breakout",
            env_config={"frameskip": 4, "repeat_action_probability": 0.25},
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            lr=2.5e-4,
            gamma=0.99,
            train_batch_size=32,
            target_network_update_freq=1000,
            double_q=True,
            dueling=False,
            num_steps_sampled_before_learning_starts=50000,
        )
        .env_runners(
            num_env_runners=num_workers,
            num_cpus_per_env_runner=1,
            rollout_fragment_length="auto",
            create_env_on_local_worker=True,
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_for_main_process=1,
        )
        .framework("torch")
        .evaluation(
            evaluation_interval=20,
            evaluation_duration=10,
            evaluation_num_env_runners=2,
            evaluation_config={"explore": False},
        )
    )

    config.replay_buffer_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
    }

    return config


def train(args):
    logger.info("=" * 60)
    logger.info("DQN Expert Training - SageMaker Edition")
    logger.info("=" * 60)

    paths = get_sagemaker_paths()
    logger.info(f"Model directory: {paths['model_dir']}")
    logger.info(f"Output directory: {paths['output_dir']}")
    logger.info(f"Checkpoint directory: {paths['checkpoint_dir']}")
    logger.info(f"TensorBoard directory: {paths['tensorboard_dir']}")

    os.makedirs(paths['model_dir'], exist_ok=True)
    os.makedirs(paths['output_dir'], exist_ok=True)
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)
    os.makedirs(paths['tensorboard_dir'], exist_ok=True)

    writer = SummaryWriter(log_dir=paths['tensorboard_dir'])
    logger.info(f"TensorBoard logging enabled: {paths['tensorboard_dir']}")

    setup_atari_roms()

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")

    ray.init(
        num_cpus=args.num_workers + 4,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True,
        logging_level=logging.INFO,
    )

    register_env("atari_breakout", create_atari_env)

    config = get_dqn_config(num_gpus=args.num_gpus, num_workers=args.num_workers)
    trainer = DQN(config=config)

    logger.info("Training Configuration:")
    logger.info(f"  GPUs: {args.num_gpus}")
    logger.info(f"  Workers: {args.num_workers}")
    logger.info(f"  Max iterations: {args.max_iterations}")
    logger.info(f"  Target reward: {args.target_reward}")

    start_time = time.time()
    best_reward = float('-inf')
    best_checkpoint = None
    training_history = []
    iteration = 0
    timesteps = 0

    try:
        for iteration in range(1, args.max_iterations + 1):
            result = trainer.train()

            timesteps = result.get("timesteps_total", 0)
            episode_reward = result.get("episode_reward_mean", 0)
            eval_reward = result.get("evaluation", {}).get("episode_reward_mean", None)

            training_history.append({
                "iteration": iteration,
                "timesteps": timesteps,
                "episode_reward_mean": episode_reward,
                "evaluation_reward_mean": eval_reward,
                "time_elapsed": time.time() - start_time,
            })

            writer.add_scalar('Training/Episode_Reward_Mean', episode_reward or 0, iteration)
            writer.add_scalar('Training/Timesteps', timesteps, iteration)
            if eval_reward is not None:
                writer.add_scalar('Evaluation/Episode_Reward_Mean', eval_reward, iteration)

            loss_info = result.get("info", {}).get("learner", {})
            if loss_info:
                for policy_id, policy_info in loss_info.items():
                    if isinstance(policy_info, dict):
                        td_error = policy_info.get("learner_stats", {}).get("mean_td_error")
                        if td_error is not None:
                            writer.add_scalar(f'Training/TD_Error_{policy_id}', td_error, iteration)

            epsilon = result.get("info", {}).get("exploration_infos", [{}])
            if epsilon and isinstance(epsilon, list) and len(epsilon) > 0:
                eps_value = epsilon[0].get("cur_epsilon")
                if eps_value is not None:
                    writer.add_scalar('Training/Epsilon', eps_value, iteration)

            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Iteration {iteration}:")
                logger.info(f"  Timesteps: {timesteps:,}")
                logger.info(f"  Training Reward: {episode_reward:.2f}")
                logger.info(f"  Evaluation Reward: {eval_reward}")
                logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

            current_reward = eval_reward if eval_reward else episode_reward
            if current_reward and current_reward > best_reward:
                best_reward = current_reward
                checkpoint_path = os.path.join(
                    paths['checkpoint_dir'], f"checkpoint_{iteration:05d}"
                )
                trainer.save(checkpoint_path)
                best_checkpoint = checkpoint_path
                logger.info(f"New best reward: {best_reward:.2f}, saved to {checkpoint_path}")

            if iteration % 100 == 0:
                checkpoint_path = os.path.join(
                    paths['checkpoint_dir'], f"checkpoint_{iteration:05d}"
                )
                trainer.save(checkpoint_path)

            if eval_reward and eval_reward >= args.target_reward:
                logger.info(f"Target reward {args.target_reward} reached!")
                break

            if timesteps >= args.max_timesteps:
                logger.info(f"Max timesteps {args.max_timesteps:,} reached!")
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted")

    finally:
        final_checkpoint = trainer.save(os.path.join(paths['model_dir'], "final_checkpoint"))
        logger.info(f"Final checkpoint saved: {final_checkpoint}")

        if best_checkpoint:
            import shutil
            best_dest = os.path.join(paths['model_dir'], "best_checkpoint")
            shutil.copytree(best_checkpoint, best_dest, dirs_exist_ok=True)
            logger.info(f"Best checkpoint copied to: {best_dest}")

        history_path = os.path.join(paths['output_dir'], "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")

        summary = {
            "total_iterations": iteration,
            "total_timesteps": timesteps,
            "best_reward": best_reward,
            "training_time_seconds": time.time() - start_time,
            "final_checkpoint": final_checkpoint,
            "best_checkpoint": best_dest if best_checkpoint else final_checkpoint,
        }
        summary_path = os.path.join(paths['output_dir'], "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        writer.add_hparams(
            {
                'num_gpus': args.num_gpus,
                'num_workers': args.num_workers,
                'learning_rate': 2.5e-4,
                'target_reward': args.target_reward,
            },
            {
                'hparam/best_reward': best_reward,
                'hparam/total_iterations': iteration,
                'hparam/training_time_hours': (time.time() - start_time) / 3600,
            }
        )
        writer.close()

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"  Total iterations: {iteration}")
        logger.info(f"  Total timesteps: {timesteps:,}")
        logger.info(f"  Best reward: {best_reward:.2f}")
        logger.info(f"  Training time: {(time.time() - start_time)/3600:.2f} hours")
        logger.info(f"  TensorBoard logs: {paths['tensorboard_dir']}")
        logger.info("=" * 60)

        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Train DQN Expert on Atari Breakout")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--max-timesteps", type=int, default=10000000)
    parser.add_argument("--target-reward", type=float, default=300.0)

    if is_sagemaker_env():
        hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
            for key, value in hyperparams.items():
                if hasattr(parser.parse_args(), key.replace('-', '_')):
                    parser.set_defaults(**{key.replace('-', '_'): type(getattr(parser.parse_args(), key.replace('-', '_')))(value)})

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
