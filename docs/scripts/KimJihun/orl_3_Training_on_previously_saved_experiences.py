from ray import tune
from ray.train import RunConfig, CheckpointConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.policy.sample_batch import SampleBatch
from pathlib import Path
import os
import ray
import glob
import pickle
import numpy as np

# 커스텀 Pickle Reader 정의
class PickleReader(InputReader):
    def __init__(self, ioctx):
        self.ioctx = ioctx
        script_dir = Path(__file__).parent
        data_dir = script_dir / "halfcheetah_expert_ppo" / "HalfCheetah-v5"
        self.files = sorted(glob.glob(str(data_dir / "*.pkl")))
        self.cur_data = []
        print(f"PickleReader initialized. Found {len(self.files)} files.")

    def next(self):
        if not self.cur_data:
            if not self.files:
                # 무한 반복을 위해 파일 목록 리필
                script_dir = Path(__file__).parent
                data_dir = script_dir / "halfcheetah_expert_ppo" / "HalfCheetah-v5"
                self.files = sorted(glob.glob(str(data_dir / "*.pkl")))
                if not self.files:
                    raise EOFError
            f = self.files.pop(0)
            try:
                with open(f, 'rb') as fp:
                    self.cur_data = pickle.load(fp) # list of dicts
            except Exception as e:
                print(f"Error reading {f}: {e}")
                return self.next()
        
        if not self.cur_data:
             return self.next()
             
        batch_data = self.cur_data.pop(0)
        if "type" in batch_data:
            del batch_data["type"]
        
        # 필수 필드만 남기고 나머지 제거 (메타데이터 스칼라가 concat 에러 유발 가능)
        allowed_keys = {"obs", "actions", "rewards", "terminated", "truncated", "dones"}
        keys_to_remove = [k for k in batch_data.keys() if k not in allowed_keys and not k.startswith("state_in")]
        for k in keys_to_remove:
            del batch_data[k]
        
        # 스칼라 값을 배열로 변환 (Timestep 길이 T에 맞춤)
        # actions 길이를 기준으로 함
        if "actions" in batch_data:
            T = len(batch_data["actions"])
            
            # obs 길이 맞춤 (혹시 T+1이면 마지막 제거? 아니면 next_obs 처리?)
            # 여기서는 일단 그대로 두고, 길이 차이나면 concat에서 에러날 텐데 
            # 보통 episode.get_observations()는 actions와 길이가 같거나 +1임.
            # safe하게 actions 길이까지만 사용
            if "obs" in batch_data and len(batch_data["obs"]) > T:
                batch_data["obs"] = batch_data["obs"][:T]
            
            if "terminated" in batch_data and np.isscalar(batch_data["terminated"]):
                val = batch_data["terminated"]
                arr = np.zeros(T, dtype=bool)
                arr[-1] = val
                batch_data["terminated"] = arr
                batch_data["terminateds"] = arr
                
            if "truncated" in batch_data and np.isscalar(batch_data["truncated"]):
                val = batch_data["truncated"]
                arr = np.zeros(T, dtype=bool)
                arr[-1] = val
                batch_data["truncated"] = arr
                batch_data["truncateds"] = arr
                
            # dones 생성 - RLlib 최신 버전에서는 dones를 직접 설정하면 안 됨 (terminateds/truncateds로 자동 계산)
            # if "dones" not in batch_data:
            #     term = batch_data.get("terminated", np.zeros(T, bool))
            #     trunc = batch_data.get("truncated", np.zeros(T, bool))
            #     batch_data["dones"] = np.logical_or(term, trunc)

        return SampleBatch(batch_data)

def input_reader_creator(ioctx):
    return PickleReader(ioctx)

ray.init(ignore_reinit_error=True)

# 레거시 API 스택을 사용하는 BC (MARWIL) 설정
config = (
    MARWILConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(env="HalfCheetah-v5")
    .framework("torch")
    .resources(num_gpus=1) # GPU 1개 사용
    .env_runners(
        num_env_runners=7, # CPU 제한(16개)으로 인해 7로 조정
    )
    .offline_data(
        input_=input_reader_creator, # 커스텀 리더 함수 전달
        actions_in_input_normalized=True, 
    )
    .training(
        beta=0.0, # BC behavior
        lr=1e-3,
        train_batch_size=2048,
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
        }
    )
    .evaluation(
        evaluation_interval=20, # 자주 확인하기 위해 20으로 단축
        evaluation_num_env_runners=7,
        evaluation_duration=5,
        evaluation_parallel_to_training=False,
        evaluation_config={
            "input": "sampler",
        }
    )
)

# 메트릭 경로 (레거시)
metric = "evaluation/sampler_results/episode_reward_mean"

tuner = tune.Tuner(
    "MARWIL",
    param_space=config,
    run_config=RunConfig(
        name="halfcheetah_offline_bc_legacy_pickle",
        stop={metric: 4000.0},
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=0,
            checkpoint_at_end=True,
        ),
        verbose=2,
    )
)

print("Starting training...")
tuner.fit()