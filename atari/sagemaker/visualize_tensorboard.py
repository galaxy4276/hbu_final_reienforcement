"""
TensorBoard Visualization Script for SageMaker Training Results

This script downloads TensorBoard logs from S3 and launches TensorBoard locally.
It can also generate static visualizations from training history JSON files.

Usage:
    # Download from S3 and launch TensorBoard
    python visualize_tensorboard.py --s3-uri s3://bucket/path/to/output --launch

    # Use local logs
    python visualize_tensorboard.py --local-dir ~/ray_results/atari_dqn_expert/tensorboard --launch

    # Generate static plots from JSON
    python visualize_tensorboard.py --json-file training_history.json --output-dir ./plots
"""

import os
import sys
import json
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


def download_from_s3(s3_uri, local_dir):
    if not HAS_BOTO3:
        print("boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    print(f"Downloading from S3: {s3_uri}")
    print(f"Local directory: {local_dir}")

    os.makedirs(local_dir, exist_ok=True)

    cmd = ['aws', 's3', 'sync', s3_uri, local_dir, '--exclude', '*.tar.gz']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error downloading from S3: {result.stderr}")
        sys.exit(1)

    print(f"Downloaded successfully to: {local_dir}")
    return local_dir


def find_tensorboard_logs(base_dir):
    base_path = Path(base_dir)
    tb_dirs = []

    for pattern in ['**/tensorboard', '**/events.out.tfevents*']:
        for path in base_path.glob(pattern):
            if path.is_dir():
                tb_dirs.append(str(path))
            else:
                tb_dirs.append(str(path.parent))

    return list(set(tb_dirs))


def launch_tensorboard(log_dirs, port=6006):
    if isinstance(log_dirs, list):
        log_dir = ','.join(log_dirs)
    else:
        log_dir = log_dirs

    print(f"Launching TensorBoard...")
    print(f"Log directory: {log_dir}")
    print(f"Port: {port}")

    cmd = ['tensorboard', '--logdir', log_dir, '--port', str(port), '--bind_all']

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)

        url = f"http://localhost:{port}"
        print(f"\nTensorBoard is running at: {url}")
        print("Press Ctrl+C to stop\n")

        webbrowser.open(url)
        process.wait()

    except KeyboardInterrupt:
        print("\nShutting down TensorBoard...")
        process.terminate()
    except FileNotFoundError:
        print("TensorBoard not found. Install with: pip install tensorboard")
        sys.exit(1)


def plot_training_history(json_file, output_dir):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    print(f"Loading training history from: {json_file}")

    with open(json_file, 'r') as f:
        history = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    if 'iteration' in history[0]:
        plot_expert_training(history, output_dir)
    elif 'epoch' in history[0]:
        plot_bc_training(history, output_dir)
    else:
        print("Unknown training history format")


def plot_expert_training(history, output_dir):
    iterations = [h['iteration'] for h in history]
    timesteps = [h['timesteps'] for h in history]
    rewards = [h.get('episode_reward_mean', 0) for h in history]
    eval_rewards = [h.get('evaluation_reward_mean') for h in history]
    eval_rewards = [r if r is not None else np.nan for r in eval_rewards]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(iterations, rewards, 'b-', alpha=0.7, label='Training')
    valid_eval = [(i, r) for i, r in zip(iterations, eval_rewards) if not np.isnan(r)]
    if valid_eval:
        eval_iters, eval_vals = zip(*valid_eval)
        ax1.plot(eval_iters, eval_vals, 'r-o', label='Evaluation')
    ax1.axhline(y=300, color='g', linestyle='--', alpha=0.5, label='Target (300)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Episode Reward Mean')
    ax1.set_title('Reward over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(iterations, timesteps, 'g-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Timesteps')
    ax2.set_title('Timesteps Progress')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    times = [h.get('time_elapsed', 0) / 60 for h in history]
    ax3.plot(times, rewards, 'b-')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Episode Reward Mean')
    ax3.set_title('Reward over Time')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if valid_eval:
        eval_iters, eval_vals = zip(*valid_eval)
        ax4.bar(eval_iters, eval_vals, alpha=0.7, color='blue')
        ax4.axhline(y=300, color='r', linestyle='--', label='Target')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Evaluation Reward')
    ax4.set_title('Evaluation Results')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'expert_training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_bc_training(history, output_dir):
    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    rewards = [h.get('mean_reward', 0) for h in history]
    max_rewards = [h.get('max_reward', 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(epochs, losses, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(epochs, rewards, 'g-', label='Mean')
    ax2.plot(epochs, max_rewards, 'r--', alpha=0.7, label='Max')
    ax2.axhline(y=287, color='orange', linestyle=':', alpha=0.7, label='Expert Mean (287)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reward')
    ax2.set_title('Evaluation Rewards')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(5)/5, mode='valid')
        ax3.plot(epochs[4:], smoothed, 'g-', linewidth=2)
    else:
        ax3.plot(epochs, rewards, 'g-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Smoothed Reward')
    ax3.set_title('Smoothed Reward Trend')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if rewards:
        performance_ratio = [r / 287.4 * 100 for r in rewards]
        ax4.bar(epochs, performance_ratio, alpha=0.7, color='purple')
        ax4.axhline(y=100, color='r', linestyle='--', label='Expert Level')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Performance Ratio (%)')
        ax4.set_title('Performance vs Expert')
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bc_training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_comparison_plot(expert_json, bc_json, output_dir):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed")
        return

    with open(expert_json, 'r') as f:
        expert_history = json.load(f)

    with open(bc_json, 'r') as f:
        bc_history = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))

    expert_rewards = [h.get('evaluation_reward_mean') or h.get('episode_reward_mean', 0)
                      for h in expert_history]
    bc_rewards = [h.get('mean_reward', 0) for h in bc_history]

    expert_time = [h.get('time_elapsed', 0) / 3600 for h in expert_history]
    bc_time = [i * 0.02 for i in range(len(bc_rewards))]

    ax.plot(expert_time, expert_rewards, 'b-', label='DQN Expert', linewidth=2)
    ax.plot([t + max(expert_time) for t in bc_time], bc_rewards, 'g-',
            label='BC Offline', linewidth=2)

    ax.axhline(y=300, color='r', linestyle='--', alpha=0.5, label='Target (300)')
    ax.axvline(x=max(expert_time), color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Training Time (hours)')
    ax.set_ylabel('Reward')
    ax.set_title('Expert vs Offline Learning Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'expert_vs_bc_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="TensorBoard Visualization for SageMaker Results")

    parser.add_argument("--s3-uri", type=str, help="S3 URI to download TensorBoard logs")
    parser.add_argument("--local-dir", type=str, help="Local TensorBoard log directory")
    parser.add_argument("--launch", action="store_true", help="Launch TensorBoard")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--json-file", type=str, help="Training history JSON file")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--expert-json", type=str, help="Expert training history JSON")
    parser.add_argument("--bc-json", type=str, help="BC training history JSON")

    args = parser.parse_args()

    if args.s3_uri:
        local_dir = args.local_dir or os.path.expanduser('~/sagemaker_tensorboard')
        download_from_s3(args.s3_uri, local_dir)
        tb_dirs = find_tensorboard_logs(local_dir)
        if not tb_dirs:
            print("No TensorBoard logs found")
            sys.exit(1)
        if args.launch:
            launch_tensorboard(tb_dirs, args.port)

    elif args.local_dir:
        if args.launch:
            launch_tensorboard(args.local_dir, args.port)
        else:
            tb_dirs = find_tensorboard_logs(args.local_dir)
            print(f"Found TensorBoard logs: {tb_dirs}")

    elif args.json_file:
        plot_training_history(args.json_file, args.output_dir)

    elif args.expert_json and args.bc_json:
        create_comparison_plot(args.expert_json, args.bc_json, args.output_dir)

    else:
        home = os.path.expanduser('~')
        default_dirs = [
            os.path.join(home, 'ray_results', 'atari_dqn_expert', 'tensorboard'),
            os.path.join(home, 'ray_results', 'atari_bc_offline', 'tensorboard'),
        ]
        existing_dirs = [d for d in default_dirs if os.path.exists(d)]

        if existing_dirs:
            print(f"Found local TensorBoard logs: {existing_dirs}")
            if args.launch:
                launch_tensorboard(existing_dirs, args.port)
            else:
                print("Use --launch to start TensorBoard")
        else:
            print("No TensorBoard logs found. Options:")
            print("  --s3-uri s3://bucket/path  : Download from S3")
            print("  --local-dir /path/to/logs  : Use local logs")
            print("  --json-file history.json   : Generate plots from JSON")


if __name__ == "__main__":
    main()
