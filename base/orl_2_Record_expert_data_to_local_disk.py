"""
전문가 데이터 수집 스크립트 (Callback 방식, Pickle 저장)
기존 체크포인트를 활용하여 데이터를 수집하고 Pickle 포맷으로 저장합니다.
커스텀 PickleReader에서 읽기 쉽도록 리스트 오브 딕셔너리 형태로 저장합니다.

체크포인트: PPO_HalfCheetah-v5_f04ec_00000_0_2025-12-06_19-33-00
"""
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.7"

import pickle
import numpy as np
import uuid
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# 체크포인트 경로
checkpoint_base = "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_f04ec_00000_0_2025-12-06_19-33-00/checkpoint_000029"
rl_module_state_path = f"{checkpoint_base}/learner_group/learner/rl_module/default_policy/module_state.pt"

# 데이터 저장 경로
script_dir = Path(__file__).parent
data_path = script_dir / "halfcheetah_expert_ppo" / "HalfCheetah-v5"
data_path.mkdir(parents=True, exist_ok=True)

# 기존 데이터 파일 삭제
for f in data_path.glob("*.pkl"):
    f.unlink()
for f in data_path.glob("*.json"): # 이전 파일 청소
    f.unlink()
    
print(f"Loading checkpoint from: {rl_module_state_path}")
print(f"Data will be saved to: {data_path} (Pickle format)")

# 체크포인트 로드
with open(rl_module_state_path, 'rb') as f:
    module_state = pickle.load(f)

# 데이터 저장을 위한 전역 버퍼
captured_episodes = []
saved_count = 0

class DataCollectionCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, env_runner, **kwargs):
        global captured_episodes
        try:
            if hasattr(episode, "get_observations"):
                obs = episode.get_observations()
                actions = episode.get_actions()
                rewards = episode.get_rewards()
            elif hasattr(episode, "user_data"):
                obs = episode.user_data.get("obs", [])
                actions = episode.user_data.get("actions", [])
                rewards = episode.user_data.get("rewards", [])
            else:
                obs = getattr(episode, "observations", [])
                actions = getattr(episode, "actions", [])
                rewards = getattr(episode, "rewards", [])

            obs = np.array(obs, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            
            if len(obs) == 0:
                return

            episode_return = np.sum(rewards)
            
            # Pickle 저장을 위한 딕셔너리 (SampleBatch 호환)
            data = {
                'type': 'SampleBatch',
                'eps_id': str(uuid.uuid4()),
                'obs': obs, # numpy array 그대로 저장
                'actions': actions,
                'rewards': rewards,
                'terminated': bool(getattr(episode, "is_terminated", False)),
                'truncated': bool(getattr(episode, "is_truncated", False)),
                # 메타데이터
                'return': float(episode_return),
                'length': len(actions)
            }
            
            captured_episodes.append(data)
            print(f"> Captured Episode: Return={episode_return:.2f}, Length={len(actions)}")
            
        except Exception as e:
            print(f"Error in callback: {e}")

ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(env="HalfCheetah-v5")
    .evaluation(
        evaluation_num_env_runners=0,
        evaluation_interval=1,
        evaluation_duration=1,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "explore": False,
        }
    )
    .env_runners(num_env_runners=0, observation_filter="MeanStdFilter")
    .callbacks(DataCollectionCallback)
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="tanh",
            vf_share_layers=False,
        ),
    )
)

print("Building algorithm...")
algo = config.build()

print("Loading weights...")
if hasattr(algo, 'env_runner') and algo.env_runner:
    algo.env_runner.module.set_state(module_state)
if algo.eval_env_runner_group:
     if hasattr(algo.eval_env_runner_group, 'local_env_runner') and algo.eval_env_runner_group.local_env_runner:
         algo.eval_env_runner_group.local_env_runner.module.set_state(module_state)

print("Weights loaded. Starting data collection...")

def save_buffer_to_pickle(buffer, path, start_idx):
    if not buffer:
        return
    
    file_name = f"episodes_{start_idx:06d}.pkl"
    out_file = path / file_name
    
    with open(out_file, 'wb') as f:
        pickle.dump(buffer, f)
            
    print(f"Saved {len(buffer)} episodes to {out_file}")

TARGET_EPISODES = 50
patience = 0

while saved_count < TARGET_EPISODES:
    algo.evaluate()
    if captured_episodes:
        for _ in captured_episodes:
            if saved_count >= TARGET_EPISODES: break
            saved_count += 1
        
        if len(captured_episodes) >= 10 or saved_count >= TARGET_EPISODES:
            save_buffer_to_pickle(captured_episodes, data_path, saved_count - len(captured_episodes))
            captured_episodes = []
    else:
        patience += 1
        if patience > 10: break

algo.stop()
ray.shutdown()
print(f"Complete! Total saved: {saved_count}")