"""
SageMaker Job Launcher for Atari RL Pipeline

This script launches SageMaker training jobs for the complete RL pipeline:
1. Stage 1: Train DQN Expert
2. Stage 2: Record Expert Data
3. Stage 3: Train Offline BC

Usage:
    # From your local machine with AWS credentials configured
    python launch_sagemaker.py --stage 1  # Train expert
    python launch_sagemaker.py --stage 2 --checkpoint-s3-uri s3://...  # Record data
    python launch_sagemaker.py --stage 3 --data-s3-uri s3://...  # Train BC
    python launch_sagemaker.py --stage all  # Run complete pipeline
"""

import os
import argparse
import time
from datetime import datetime

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.processing import ScriptProcessor


def get_execution_role():
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        role = os.environ.get('SAGEMAKER_ROLE')
        if not role:
            raise ValueError(
                "SAGEMAKER_ROLE environment variable not set. "
                "Set it to your SageMaker execution role ARN."
            )
    return role


def create_estimator(
    entry_point,
    role,
    instance_type,
    instance_count=1,
    hyperparameters=None,
    output_path=None,
    base_job_name=None,
    max_run=86400,
):
    return PyTorch(
        entry_point=entry_point,
        source_dir='.',
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters=hyperparameters or {},
        output_path=output_path,
        base_job_name=base_job_name,
        max_run=max_run,
        keep_alive_period_in_seconds=0,
        disable_profiler=True,
        debugger_hook_config=False,
    )


def launch_stage1_training(args, role, session):
    print("=" * 60)
    print("Stage 1: Training DQN Expert")
    print("=" * 60)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"atari-dqn-expert-{timestamp}"

    output_path = f"s3://{args.bucket}/atari-rl/stage1-expert/"

    hyperparameters = {
        'num-gpus': 1,
        'num-workers': 4,
        'max-iterations': args.max_iterations,
        'max-timesteps': args.max_timesteps,
        'target-reward': args.target_reward,
    }

    estimator = create_estimator(
        entry_point='train_expert.py',
        role=role,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name='atari-dqn-expert',
        max_run=args.max_run_hours * 3600,
    )

    print(f"Starting training job: {job_name}")
    print(f"Instance type: {args.instance_type}")
    print(f"Output path: {output_path}")

    estimator.fit(wait=args.wait)

    if args.wait:
        model_path = estimator.model_data
        print(f"Training complete! Model saved to: {model_path}")
        return model_path
    else:
        print(f"Training job started: {estimator.latest_training_job.name}")
        return None


def launch_stage2_recording(args, role, session):
    print("=" * 60)
    print("Stage 2: Recording Expert Data")
    print("=" * 60)

    if not args.checkpoint_s3_uri:
        raise ValueError("--checkpoint-s3-uri required for stage 2")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = f"s3://{args.bucket}/atari-rl/stage2-data/"

    hyperparameters = {
        'num-episodes': args.num_episodes,
        'batch-size': 50,
    }

    estimator = create_estimator(
        entry_point='record_data.py',
        role=role,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name='atari-record-data',
        max_run=4 * 3600,
    )

    print(f"Starting recording job")
    print(f"Checkpoint: {args.checkpoint_s3_uri}")
    print(f"Output path: {output_path}")

    estimator.fit(
        inputs={'model': args.checkpoint_s3_uri},
        wait=args.wait,
    )

    if args.wait:
        data_path = f"{output_path}{estimator.latest_training_job.name}/output/"
        print(f"Recording complete! Data saved to: {data_path}")
        return data_path
    else:
        print(f"Recording job started: {estimator.latest_training_job.name}")
        return None


def launch_stage3_training(args, role, session):
    print("=" * 60)
    print("Stage 3: Training Offline BC")
    print("=" * 60)

    if not args.data_s3_uri:
        raise ValueError("--data-s3-uri required for stage 3")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = f"s3://{args.bucket}/atari-rl/stage3-bc/"

    hyperparameters = {
        'batch-size': args.batch_size,
        'learning-rate': args.learning_rate,
        'epochs': args.epochs,
        'eval-interval': 5,
    }

    estimator = create_estimator(
        entry_point='train_offline.py',
        role=role,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name='atari-bc-offline',
        max_run=4 * 3600,
    )

    print(f"Starting BC training job")
    print(f"Data: {args.data_s3_uri}")
    print(f"Output path: {output_path}")

    estimator.fit(
        inputs={'training': args.data_s3_uri},
        wait=args.wait,
    )

    if args.wait:
        model_path = estimator.model_data
        print(f"BC training complete! Model saved to: {model_path}")
        return model_path
    else:
        print(f"BC training job started: {estimator.latest_training_job.name}")
        return None


def launch_complete_pipeline(args, role, session):
    print("=" * 60)
    print("Running Complete Pipeline")
    print("=" * 60)

    args.wait = True

    print("\n[1/3] Stage 1: Training Expert...")
    model_path = launch_stage1_training(args, role, session)

    print("\n[2/3] Stage 2: Recording Data...")
    args.checkpoint_s3_uri = model_path
    data_path = launch_stage2_recording(args, role, session)

    print("\n[3/3] Stage 3: Training BC...")
    args.data_s3_uri = data_path
    bc_model_path = launch_stage3_training(args, role, session)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Expert model: {model_path}")
    print(f"Expert data: {data_path}")
    print(f"BC model: {bc_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker Atari RL Jobs")

    parser.add_argument("--stage", type=str, required=True,
                        choices=['1', '2', '3', 'all'],
                        help="Stage to run (1, 2, 3, or all)")
    parser.add_argument("--bucket", type=str, required=True,
                        help="S3 bucket for storing outputs")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge",
                        help="SageMaker instance type")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job completion")

    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--max-timesteps", type=int, default=10000000)
    parser.add_argument("--target-reward", type=float, default=300.0)
    parser.add_argument("--max-run-hours", type=int, default=12)

    parser.add_argument("--checkpoint-s3-uri", type=str,
                        help="S3 URI of expert checkpoint (for stage 2)")
    parser.add_argument("--num-episodes", type=int, default=500)

    parser.add_argument("--data-s3-uri", type=str,
                        help="S3 URI of expert data (for stage 3)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    role = get_execution_role()
    session = sagemaker.Session()

    print(f"SageMaker Role: {role}")
    print(f"S3 Bucket: {args.bucket}")
    print(f"Instance Type: {args.instance_type}")

    if args.stage == '1':
        launch_stage1_training(args, role, session)
    elif args.stage == '2':
        launch_stage2_recording(args, role, session)
    elif args.stage == '3':
        launch_stage3_training(args, role, session)
    elif args.stage == 'all':
        launch_complete_pipeline(args, role, session)


if __name__ == "__main__":
    main()
