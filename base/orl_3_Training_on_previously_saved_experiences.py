from ray import tune
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
from pathlib import Path
import os

# 절대 경로로 변환하여 데이터 경로 설정
# orl_2에서 생성된 데이터는 halfcheetah_expert_ppo 하위에 저장됨
script_dir = Path(__file__).parent
data_path = (script_dir / "halfcheetah_expert_ppo" / "HalfCheetah-v5").as_posix()

print(f"Using offline data from: {data_path}")
print(f"Data exists: {os.path.exists(data_path)}")

# Setup the config for behavior cloning.
config = (
    BCConfig()
    .environment(
        # Use the `HalfCheetah-v5` environment from which the
        # data was recorded. This is merely for receiving
        # action and observation spaces and to use it during
        # evaluation.
        env="HalfCheetah-v5",
    )
    .learners(
        # Use a single local learner.
        num_learners=0,
        num_gpus_per_learner=1,
    )
    .training(
        # This has to be defined in the new offline RL API.
        train_batch_size_per_learner=1024,
    )
    .offline_data(
        # Link the data.
        input_=[data_path],
        # You want to read in RLlib's episode format b/c this
        # is how you recorded data.
        input_read_episodes=True,
        # Read smaller batches from the data than the learner
        # trains on. Note, each batch element is an episode
        # with multiple timesteps.
        input_read_batch_size=512,
        # Create exactly 2 `DataWorkers` that transform
        # the data on-the-fly. Give each of them a single
        # CPU.
        map_batches_kwargs={
            "concurrency": 2,
            "num_cpus": 1,
        },
        # When iterating over the data, prefetch two batches
        # to improve the data pipeline. Don't shuffle the
        # buffer (the data is too small).
        iter_batches_kwargs={
            "prefetch_batches": 2,
            "local_shuffle_buffer_size": None,
        },
        # You must set this for single-learner setups.
        dataset_num_iters_per_learner=1,
    )
    .evaluation(
        # Run evaluation to see how well the learned policy
        # performs. Run every 3rd training iteration an evaluation.
        evaluation_interval=3,
        # Use a single `EnvRunner` for evaluation.
        evaluation_num_env_runners=1,
        # In each evaluation rollout, collect 5 episodes of data.
        evaluation_duration=5,
        # Evaluate the policy parallel to training.
        evaluation_parallel_to_training=True,
    )
)

# Set the stopping metric to be the evaluation episode return mean.
metric = f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"

# Configure Ray Tune.
tuner = tune.Tuner(
    "BC",
    param_space=config,
    run_config=tune.RunConfig(
        name="halfcheetah_offline_bc",
        # Stop behavior cloning when we reach 3000 in return.
        stop={metric: 3000.0},
        checkpoint_config=tune.CheckpointConfig(
            # Only checkpoint at the end to be faster.
            checkpoint_frequency=0,
            checkpoint_at_end=True,
        ),
        verbose=2,
    )
)
# Run the experiment.
analysis = tuner.fit()