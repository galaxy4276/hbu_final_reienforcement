import os
from pathlib import Path
import pickle

import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)

# ===== 경로 설정 =====
save_dir = "/home/user/hbu_final_reinforcement/finetuning_results"
expert_ckpt = (
    "/home/user/hbu_final_reinforcement/offline_results/HalfCheetah_offline_bc/"
    "BC_HalfCheetah-v5_a0a60_00000_0_2025-12-04_01-39-24/checkpoint_004067"
)
env_name = "HalfCheetah-v5"

os.makedirs(save_dir, exist_ok=True)

ray.init()

# ===== 1) BC RLModule state만 로드 =====
ckpt_path = Path(expert_ckpt)
bc_rlm_dir = ckpt_path / "learner_group" / "learner" / "rl_module" / "default_policy"
state_path = bc_rlm_dir / "module_state.pkl"

print("[LOAD] BC RLModule dir:", bc_rlm_dir)
print("[LOAD] module_state.pkl exists:", state_path.exists())

with open(state_path, "rb") as f:
    bc_state = pickle.load(f)

print("[LOAD] Loaded BC RLModule state dict from:", state_path)

# ===== 2) PPO 알고리즘 생성 (BC와 모델 구조 동일) =====
config = (
    PPOConfig()
    .environment(env_name)
    .framework("torch")
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[512, 512, 256],
            fcnet_activation="tanh",
            vf_share_layers=False,
            free_log_std=True,   # ✅ BC와 동일
        )
    )
    .training(
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        num_epochs=10,
        train_batch_size=32768 * 2,
        minibatch_size=4096,
        clip_param=0.2,
        vf_loss_coeff=1.0,
        entropy_coeff=0.0005,
        grad_clip=0.5,
    )
    .env_runners(
        batch_mode="complete_episodes",
        num_env_runners=5,
        num_envs_per_env_runner=4,
        observation_filter="MeanStdFilter",
    )
    # 🔴 evaluate()가 로컬 env_runner 말고
    #     별도의 평가용 env_runner_group을 쓰도록 명시
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_duration=10,              # 10 에피소드 평가
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=False,
    )
)

algo = config.build()

# BC weight를 PPO learner에 주입
algo.learner_group.set_weights({"default_policy": bc_state})
print("[INIT] Copied BC policy weights into PPO learner")

# ===== 3) 학습 전에 evaluate()로 사전 성능 확인 =====
print("\n========== PRE-TRAIN EVAL (algo.evaluate) ==========\n")

pre_eval_results = algo.evaluate()
er_pre = pre_eval_results.get(ENV_RUNNER_RESULTS, {})
pre_mean = er_pre.get(EPISODE_RETURN_MEAN, None)

print(f"[PRE-EVAL] raw eval results keys: {list(er_pre.keys())}")
print(f"[PRE-EVAL] episode_return_mean = {pre_mean}")

# ===== 4) PPO 온라인 파인튜닝 =====
print("\n========== PPO ONLINE FINE-TUNING ==========\n")

for i in range(500):
    result = algo.train()

    er = result.get(ENV_RUNNER_RESULTS, {})
    mean_ret = er.get(EPISODE_RETURN_MEAN, None)

    if mean_ret is None:
        print(f"[ONLINE] Iter {i+1:03d}, no episode_return_mean yet. keys={list(er.keys())}")
    else:
        print(f"[ONLINE] Iter {i+1:03d}, episode_return_mean = {mean_ret:.3f}")

    # 10 iter마다 평가 + 체크포인트 저장
    if (i + 1) % 10 == 0:
        print(f"\n========== EVAL at iter {i+1:03d} (algo.evaluate) ==========")
        eval_results = algo.evaluate()
        er_eval = eval_results.get(ENV_RUNNER_RESULTS, {})
        eval_mean = er_eval.get(EPISODE_RETURN_MEAN, None)
        print(f"[EVAL] Iter {i+1:03d}, eval_return_mean = {eval_mean}\n")

        ckpt_dir = algo.save_to_path(
            path=os.path.join(save_dir, f"iter_{i+1:03d}")
        )
        print(f"[SAVE] Saved checkpoint at: {ckpt_dir}")

algo.stop()
ray.shutdown()
