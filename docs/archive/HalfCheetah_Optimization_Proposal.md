# HalfCheetah-v5 5000점 달성을 위한 심화 최적화 제안서

## 1. 목표 (Goal)
*   **Target Score**: **5000+** (기존 목표 3500~4000 대비 상향)
*   **전략**: 신경망 크기 확장(Capacity Increase) 및 정밀 파라미터 튜닝(+$\alpha$)

## 2. 핵심 전략: 신경망 크기 및 +$\alpha$ 튜닝

### 2.1. 신경망 크기 (Neural Network Size)
5000점 이상의 고득점은 에이전트가 매우 세밀하고 복잡한 제어 정책을 학습해야 함을 의미합니다.
*   **제안**: `[512, 512, 256]`
*   **근거**: 기존 `[256, 256]` 모델은 표현력의 한계로 인해 4000점 대에서 정체될 가능성이 높습니다. 더 깊고 넓은 네트워크는 미세한 관절 제어 패턴을 학습할 수 있는 용량(Capacity)을 제공합니다.

### 2.2. +$\alpha$ 파라미터 튜닝 (Fine-tuning)
단순히 모델만 키우는 것으로는 부족하며, 학습 안정성과 수렴 정밀도를 높이는 설정이 필요합니다.

| 파라미터 | 기존/일반 설정 | **5000점 목표 설정** | 효과 및 근거 |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | `3e-4` (고정) | **Linear Decay** (`3e-4` $\to$ `0.0`) | 학습 후반부에 보폭을 줄여 최적점(Global Optima)에 정밀하게 안착하도록 유도. |
| **Entropy Coeff** | `0.0` | **`0.001`** | 초기에는 약간의 무작위성을 부여하여 지역 최적해(Local Optima) 탈출을 돕고, 나중에는 0에 수렴. |
| **Grad Clip** | `0.5` | **`0.8`** | 더 큰 모델(Deep Network)에서는 그래디언트가 커질 수 있으므로, 클리핑 임계값을 약간 완화하여 학습 속도 저하 방지. |
| **VF Loss Coeff** | `0.5` | **`1.0`** | 가치 함수(Critic)의 정확도가 높아야 좋은 정책(Actor)이 나옴. Critic 학습 비중 강화. |
| **Batch Size** | `2048` | **`8192`** | 노이즈를 최소화하여 매우 안정적인 업데이트 보장. |

## 3. 최종 적용 설정 (Configuration to Apply)

### `orl_1_Training_an_expert_policy.py` 및 `orl_4_Online_finetuning.py` 적용 값

```python
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .framework("torch")
    .training(
        # [전략 1] 신경망 확장
        model={
            "fcnet_hiddens": [512, 512, 256],  # Deep & Wide
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
            "free_log_std": True,
        },
        # [전략 2] 정밀 튜닝 (+a)
        lr=[[0, 3e-4], [10_000_000, 0.0]],  # Decay Schedule (10M steps)
        train_batch_size=8192,
        minibatch_size=1024,
        num_epochs=20,
        clip_param=0.2,
        vf_loss_coeff=1.0,  # Critic 중요도 상향
        entropy_coeff=0.001, # 소량의 탐험 허용
        grad_clip=0.8,       # 클리핑 완화
        gamma=0.99,
        lambda_=0.95,
    )
    .env_runners(
        num_env_runners=5,
        num_envs_per_env_runner=4, # 총 20개 환경 병렬 처리
        observation_filter="MeanStdFilter"
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_duration=10,
    )
)
```

## 4. 실행 계획
1.  **설정 적용**: 위 설정을 두 파이썬 스크립트에 적용합니다.
2.  **학습 모니터링**: 초기 100-200 이터레이션에서 점수 상승 추이를 관찰합니다.
