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
# orl_2에서 생성된 데이터는 cartpole-v1 하위에 저장됨
script_dir = Path(__file__).parent
data_path = (script_dir / "offline_data" / "cheetah").as_posix()

print(f"Using offline data from: {data_path}")
print(f"Data exists: {os.path.exists(data_path)}")

# Setup the config for behavior cloning.
config = (
    BCConfig()
    .environment(env="HalfCheetah-v5")
    .learners(num_learners=0)
    .training(train_batch_size_per_learner=4096)
    .offline_data(
        input_=[data_path],
        input_read_episodes=True,
        input_read_batch_size=128,
        map_batches_kwargs={"concurrency": 2, "num_cpus": 1},
        iter_batches_kwargs={"prefetch_batches": 1},
        dataset_num_iters_per_learner=1,
    )
    .evaluation(
        evaluation_interval=3,
        evaluation_num_env_runners=1,
        evaluation_duration=3,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=True,
    )
)

# Set the stopping metric to be the evaluation episode return mean.
metric = f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"

tuner = tune.Tuner(
    "BC",
    param_space=config,
    run_config=tune.RunConfig(
        name="HalfCheetah_offline_bc",
        stop={metric: 3000},
        checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
        verbose=2,
    ),
)

analysis = tuner.fit()