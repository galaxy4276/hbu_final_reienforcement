# Atari Breakout DQN Training Guide for SageMaker

## 프로젝트 목표

Atari Breakout 게임에서 **오프라인 강화학습(Offline RL)** 파이프라인을 구현합니다.

### 3단계 파이프라인

| 단계 | 스크립트 | 목표 |
|------|----------|------|
| Stage 1 | `train_expert.py` | DQN으로 전문가 정책 학습 (목표: 평균 300점) |
| Stage 2 | `record_data.py` | 전문가 에피소드 500개 녹화 |
| Stage 3 | `train_offline.py` | Behavior Cloning으로 오프라인 학습 |

---

## 현재 문제

RLlib 버전 호환성 문제로 `replay_buffer_config` 설정에서 오류 발생:

```
TypeError: argument of type 'ABCMeta' is not iterable
```

---

## 해결 방법

### 방법 1: Ray 버전 확인 후 다운그레이드

```bash
# 현재 버전 확인
python -c "import ray; print(ray.__version__)"

# 안정 버전으로 다운그레이드 (권장)
pip install "ray[rllib]==2.10.0" --force-reinstall
```

### 방법 2: 새로운 API 스택 사용 (Ray 2.50+)

`train_expert.py`의 `get_dqn_config()` 함수를 다음으로 교체:

```python
def get_dqn_config(num_gpus=1, num_workers=4):
    config = (
        DQNConfig()
        .environment(
            env="atari_breakout",
            env_config={"frameskip": 4, "repeat_action_probability": 0.25},
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .training(
            lr=2.5e-4,
            gamma=0.99,
            train_batch_size=32,
            target_network_update_freq=1000,
            double_q=True,
            dueling=False,
            num_steps_sampled_before_learning_starts=50000,
        )
        .env_runners(
            num_env_runners=num_workers,
            num_cpus_per_env_runner=1,
            rollout_fragment_length="auto",
            create_env_on_local_worker=True,
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_for_main_process=1,
        )
        .framework("torch")
        .evaluation(
            evaluation_interval=20,
            evaluation_duration=10,
            evaluation_num_env_runners=2,
            evaluation_config={"explore": False},
        )
    )
    return config
```

---

## 실행 순서

### 1. 환경 확인

```bash
# Ray 버전 확인
python -c "import ray; print(ray.__version__)"

# GPU 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### 2. Stage 1: 전문가 학습

```bash
cd ~/atari
python train_expert.py --num-gpus 1 --max-iterations 5000 --target-reward 300
```

**예상 시간**: 2-4시간
**출력 경로**: `~/ray_results/atari_dqn_expert/`

### 3. Stage 2: 데이터 수집

```bash
python record_data.py \
    --checkpoint-path ~/ray_results/atari_dqn_expert/model/best_checkpoint \
    --num-episodes 500
```

**예상 시간**: 30분-1시간
**출력 경로**: `~/ray_results/atari_record_data/`

### 4. Stage 3: 오프라인 학습

```bash
python train_offline.py --epochs 100
```

**예상 시간**: 30분-1시간
**출력 경로**: `~/ray_results/atari_bc_offline/`

---

## TensorBoard 모니터링

```bash
# 별도 터미널에서
tensorboard --logdir ~/ray_results/ --port 6006
```

브라우저에서 `http://localhost:6006` 접속

---

## 예상 성능

| 단계 | 알고리즘 | 목표 점수 |
|------|----------|----------|
| 기준선 | Random | ~1.2 |
| Stage 1 | DQN Expert | 300+ |
| Stage 3 | BC Offline | 180-210 (전문가의 60-70%) |

---

## 하이퍼파라미터

### DQN (Stage 1)

| 파라미터 | 값 |
|----------|-----|
| Learning Rate | 2.5e-4 |
| Gamma | 0.99 |
| Batch Size | 32 |
| Target Update Freq | 1000 |
| Epsilon (초기→최종) | 1.0 → 0.01 |
| Replay Buffer | 1,000,000 |
| Warmup Steps | 50,000 |

### BC (Stage 3)

| 파라미터 | 값 |
|----------|-----|
| Learning Rate | 1e-4 |
| Batch Size | 64 |
| Epochs | 100 |
| Dropout | 0.3 |

---

## 트러블슈팅

### 오류: `ABCMeta is not iterable`

Ray 2.50.x 버전의 버그입니다. 다음 중 하나를 시도:

1. Ray 다운그레이드: `pip install "ray[rllib]==2.10.0"`
2. 새로운 API 스택 사용 (위 코드 참조)

### 오류: `FrameStack not gymnasium.Env`

`FrameStack` 클래스가 `gym.Wrapper`를 상속하는지 확인.

### 메모리 부족

- `--num-workers` 값 줄이기 (4 → 2)
- Batch size 줄이기

---

## 파일 구조

```
~/atari/
├── train_expert.py      # Stage 1
├── record_data.py       # Stage 2
├── train_offline.py     # Stage 3
├── configs/
│   └── dqn_breakout_config.py
├── utils/
│   ├── preprocess.py
│   └── cnn_models.py
└── requirements.txt

~/ray_results/
├── atari_dqn_expert/    # Stage 1 출력
├── atari_record_data/   # Stage 2 출력
└── atari_bc_offline/    # Stage 3 출력
```

---

**최종 목표**: 환경 상호작용 없이 사전 수집된 데이터만으로 게임을 플레이할 수 있는 에이전트 학습
