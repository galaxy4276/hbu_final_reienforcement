import os
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

best_checkpoint = "/home/pz1004/ray_results/docs_rllib_offline_pretrain_ppo/PPO_CartPole-v1_84fbc_00000_0_2025-11-17_23-48-46/checkpoint_000049"

# Store recording data under the following path.
data_path = "offline_data/cheetah"

# Configure the algorithm for recording.
config = (
    PPOConfig()
    .environment(env="HalfCheetah-v5")
    .env_runners(batch_mode="complete_episodes")
    .evaluation(
        evaluation_num_env_runners=2,
        evaluation_duration=20,
        evaluation_duration_unit="episodes",
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="tanh",
            vf_share_layers=False,
        ),
    )
    .offline_data(
        output=data_path,
        output_write_episodes=True,
        output_max_rows_per_file=25,
    )
)

algo = config.build()
algo.restore_from_path(best_checkpoint)

for i in range(5):
    print(f"[Record] Iter {i+1}")
    algo.evaluate()

algo.stop()