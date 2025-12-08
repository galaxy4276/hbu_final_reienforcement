from ray import tune
from ray.rllib.algorithms.cql import CQLConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
)
from pathlib import Path
import os

script_dir = Path(__file__).parent
data_path = (script_dir / "offline_data" / "cheetah").as_posix()
save_dir = Path("/home/user/hbu_final_reinforcement/offline_results")

print(f"Using offline data from: {data_path}")
print(f"Data exists: {os.path.exists(data_path)}")

config = (
    MARWILConfig()
    .environment(env="HalfCheetah-v5")
    .framework("torch")
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )
    .training(
        train_batch_size_per_learner=4096,
        beta=1.0,
        lr=1e-4,
        num_epochs=200,
        vf_coeff=1.0,
        grad_clip=1.0,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[512, 512, 256], 
            fcnet_activation="tanh",
            vf_share_layers=False,
            free_log_std=True,
        )
    )
    .env_runners(
        num_env_runners=0,
        observation_filter=None,
    )
    .offline_data(
        input_=[data_path],
        input_read_episodes=True,
        input_read_batch_size=256,
        map_batches_kwargs={"concurrency": 2, "num_cpus": 1},
        iter_batches_kwargs={"prefetch_batches": 1},
        dataset_num_iters_per_learner=1,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=20,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=True,
    )
)

metric = f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"

tuner = tune.Tuner(
    "MARWIL",
    param_space=config,
    run_config=tune.RunConfig(
        name="HalfCheetah_offline_MARWIL",
        storage_path=str(save_dir),
        stop={metric: 3500},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True
        ),
        verbose=2,
    ),
)

print("🚀 Starting Offline Training...")
results = tuner.fit()

best_checkpoint = (
    results
    .get_best_result(
        metric=metric,
        mode="max"
    )
    .checkpoint.path
)

print(f"[OFFLINE] Best checkpoint path: {best_checkpoint}")