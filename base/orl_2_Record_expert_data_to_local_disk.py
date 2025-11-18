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
data_path = "docs_rllib_offline_pretrain_ppo"

# Configure the algorithm for recording.
config = (
    PPOConfig()
    # The environment needs to be specified.
    .environment(
        env="CartPole-v1",
    )
    # Make sure to sample complete episodes because
    # you want to record RLlib's episode objects.
    .env_runners(
        batch_mode="complete_episodes",
    )
    # Set up 5 evaluation `EnvRunners` for recording.
    # Sample 50 episodes in each evaluation rollout.
    .evaluation(
        evaluation_num_env_runners=2,
        evaluation_duration=50,
        evaluation_duration_unit="episodes",
    )
    # Use the checkpointed expert policy from the preceding PPO training.
    # Note, we have to use the same `model_config` as
    # the one with which the expert policy was trained, otherwise
    # the module state can't be loaded.
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[32],
            fcnet_activation="linear",
            # Share encoder layers between value network
            # and policy.
            vf_share_layers=True,
        ),
    )
    # Define the output path and format. In this example you
    # want to store data directly in RLlib's episode objects.
    # Each Parquet file should hold no more than 25 episodes.
    .offline_data(
        output=data_path,
        output_write_episodes=True,
        output_max_rows_per_file=10,
    )
)

# Build the algorithm.
algo = config.build()
# Load now the PPO-trained `RLModule` to use in recording.
algo.restore_from_path(
    best_checkpoint,
    # Load only the `RLModule` component here.
    # component=COMPONENT_RL_MODULE,
)

# Run 10 evaluation iterations and record the data.
for i in range(10):
    print(f"Iteration {i + 1}")
    eval_results = algo.evaluate()
    print(eval_results)

# Stop the algorithm. Note, this is important for when
# defining `output_max_rows_per_file`. Otherwise,
# remaining episodes in the `EnvRunner`s buffer isn't written to disk.
algo.stop()