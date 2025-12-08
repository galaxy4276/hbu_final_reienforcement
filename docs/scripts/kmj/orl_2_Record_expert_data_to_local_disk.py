import os
import shutil
from pathlib import Path

os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.7"


from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

# 상황에 맞게 수정
best_checkpoint = "/home/user/hbu_final_reinforcement/results/HalfCheetah_expert/PPO_HalfCheetah-v5_72205_00000_0_2025-12-01_11-35-48/checkpoint_000633"

# Store recording data under the following path.
data_path = "offline_data/cheetah"

# Configure the algorithm for recording.
config = (
    PPOConfig()
    .environment(env="HalfCheetah-v5")
    .env_runners(
        batch_mode="complete_episodes",
        observation_filter="MeanStdFilter",
        )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_duration=50,
        evaluation_duration_unit="episodes",
        evaluation_sample_timeout_s=3600,
        evaluation_parallel_to_training=False,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[512, 512, 256], 
            fcnet_activation="tanh",
            vf_share_layers=False,
            free_log_std=True,
        ),
    )
    .offline_data(
        output=data_path,
        output_write_episodes=True,
        output_max_rows_per_file=100,
    )
)

algo = config.build()
algo.restore_from_path(best_checkpoint)

for i in range(10):
    print(f"[Record] Iter {i+1}")
    algo.evaluate()

algo.stop()