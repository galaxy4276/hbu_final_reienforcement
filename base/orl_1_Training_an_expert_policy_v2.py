import ray
from ray import tune
from ray.train import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)

ray.init(include_dashboard=False, ignore_reinit_error=True)

# ============================================================
# Configure the PPO algorithm.
# Ray 공식 튜닝 예제 기반 + Google Brain 권장사항 적용
# 체크포인트: PPO_HalfCheetah-v5_0b810_00000_0_2025-12-07_02-21-47
# ============================================================
# [주요 변경 파라미터]
# 1. fcnet_hiddens: [256,256] → [512,256] (너비 튜닝)
# 2. post_fcnet_weights_initializer: 마지막 레이어 가중치 100배 작게 (gain=0.01)
# 3. vf_share_layers=False: Policy/Value 네트워크 분리
# ============================================================
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
    .env_runners(
        batch_mode="truncate_episodes",
        num_env_runners=8,
        num_envs_per_env_runner=8,
        observation_filter="MeanStdFilter",
    )
    .training(
        lr=3e-4,
        train_batch_size=65536,
        minibatch_size=4096,
        num_epochs=32,

        grad_clip=0.5,
        clip_param=0.2,
        vf_loss_coeff=0.5,

        entropy_coeff=0.0,

        use_kl_loss=True,
        kl_coeff=1.0,
        kl_target=0.01,

        gamma=0.99,
        lambda_=0.95,
        model={
            # [Google Brain 권장] 너비 튜닝: 기본값 [256,256]에서 변경
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "tanh",
            # [Google Brain 권장] Policy/Value 네트워크 분리
            "vf_share_layers": False,
            "free_log_std": True,
            # [Google Brain 권장] 마지막 레이어 가중치 100배 작게 초기화
            # → 초기 정책이 균등분포에 가까워 탐험 친화적
            "post_fcnet_weights_initializer": "orthogonal_",
            "post_fcnet_weights_initializer_config": {"gain": 0.01},
        },
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=True,
    )
)

# Define the metric to use for stopping.
metric = f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"

# Define the Tuner.
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=RunConfig(
        stop={
            metric: 10000.0,
        },
        name="halfcheetah_expert_ppo_v2",
        verbose=2,
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
        ),
    ),
)
results = tuner.fit()

# Store the best checkpoint to use it later for recording
# an expert policy.
best_checkpoint = results.get_best_result(metric=metric, mode="max").checkpoint.path

print(f"Best checkpoint path: {best_checkpoint}")
