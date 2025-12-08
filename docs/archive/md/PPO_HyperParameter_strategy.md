# [Report] HalfCheetah-v5 고성능 달성을 위한 PPO 최적화 전략

## 1\. 개요 및 목표

  * **목표:** MuJoCo `HalfCheetah-v5` 환경에서 **Expert Score (3500+ 점)** 달성.
  * **알고리즘:** Ray RLlib PPO (Proximal Policy Optimization).
  * **하드웨어:** NVIDIA RTX 3070 Ti (GPU), AMD Ryzen 3700X (CPU).
  * **핵심 전략:** 신경망 용량(Capacity) 증대, 배치 사이즈 대형화(Large Batch), 하드웨어 가속 최적화.

-----

## 2\. 최종 파라미터 설정 (Configuration)

```python   
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune

config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    # [Hardware: GPU Acceleration]
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,  # RTX 3070 Ti 활용
    )
    # [Hardware: CPU Parallelism]
    .env_runners(
        num_env_runners=12,      # Ryzen 3700X (16 Threads)의 약 75% 할당
        num_envs_per_env_runner=4, # Vectorization: 워커 당 4개 환경 동시 실행 (총 48개)
        observation_filter="MeanStdFilter", # 필수: MuJoCo 입력 정규화
    )
    .training(
        # [Optimization: Stability & Throughput]
        train_batch_size=8192,   # Large Batch for Low Variance & GPU Saturation
        minibatch_size=1024,     # High Throughput for RTX 3070 Ti
        num_epochs=10,           # Standard PPO Epochs
        
        # [Optimization: Learning Rate Schedule]
        # Linear Decay: 3e-4 -> 0.0 (1천만 스텝 기준)
        lr=[[0, 3e-4], [10_000_000, 0.0]],
        
        # [PPO Hyperparameters: Standard]
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.0,
        grad_clip=0.5,
        gamma=0.99,
        lambda_=0.95,
        
        # [Model Architecture: Wide Network]
        model={
            "fcnet_hiddens": [1024, 1024], # Google Brain 권장 (Width > Depth)
            "fcnet_activation": "tanh",    # Continuous Control Standard
            "vf_share_layers": False,      # Actor-Critic 분리
            "free_log_std": True,          # Independent Exploration Parameter
        },
    )
    .evaluation(
        evaluation_interval=20,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
    )
)
```

-----

## 3\. 설정 근거 및 참고 문헌 분석

각 설정값은 경험적 추측이 아닌, 해당 분야의 권위 있는 연구 결과에 기반합니다.

### A. 신경망 아키텍처: Wide Network (`[1024, 1024]`)

  * **설정:** Hidden Layer 크기를 `[256, 256]`에서 `[1024, 1024]`로 대폭 확장.
  * **근거 논문:** *Andrychowicz et al. (Google Brain), "What Matters In On-Policy Reinforcement Learning?" (2021)*
  * **분석:**
      * 해당 논문의 실험 결과, MuJoCo와 같은 연속 제어(Continuous Control) 환경에서는 신경망의 깊이(Depth)보다 \*\*너비(Width)\*\*가 성능에 더 긍정적인 영향을 미침이 증명되었습니다.
      * 초기 PPO 논문은 작은 네트워크를 썼으나, 최신 SOTA(State-of-the-Art) 모델들은 복잡한 관절 역학(Dynamics)을 표현하기 위해 더 큰 용량(Capacity)을 요구합니다. RTX 3070 Ti의 연산력은 이를 처리하기에 충분합니다.

### B. 배치 사이즈 및 최적화: Large Batch & Linear Decay

  * **설정:**
      * `train_batch_size`: 8192 (기본값 4000 대비 2배 이상)
      * `lr_schedule`: `3e-4`에서 `0.0`으로 선형 감소.
  * **근거 논문:**
    1.  *Schulman et al., "Proximal Policy Optimization Algorithms" (2017)*
    2.  *OpenAI Spinning Up Benchmarks*
  * **분석:**
      * **Large Batch:** 배치가 클수록 Policy Gradient의 분산(Variance)이 줄어들어 학습 방향이 정확해집니다(Stable Updates). 이는 학습 초기의 불안정성을 잡고 최종 점수를 높이는 핵심 요인입니다.
      * **Linear Decay:** 고정된 학습률은 최적점 근처에서 진동(Oscillation)을 유발합니다. 학습 종료 시점에 학습률을 `0`에 수렴시키는 것이 미세 조정(Fine-tuning)과 Expert Score 도달에 필수적입니다.

### C. 하드웨어 최적화: Throughput Maximization

  * **설정:**
      * CPU: `num_env_runners=12` (16 쓰레드 중 75% 사용)
      * GPU: `minibatch_size=1024`
  * **근거 논문:** *Liang et al., "RLlib: Abstractions for Distributed Reinforcement Learning" (2018)*
  * **분석:**
      * **CPU:** Ryzen 3700X의 16 쓰레드를 전부 사용하면 컨텍스트 스위칭 오버헤드가 발생합니다. 12개의 워커가 각각 4개의 환경을 벡터화(Vectorization)하여 처리함으로써, GPU가 쉴 새 없이 학습할 수 있도록 데이터를 공급합니다.
      * **GPU:** RTX 3070 Ti는 수천 개의 CUDA 코어를 가집니다. 작은 미니배치(64, 128)는 GPU 자원의 낭비입니다. 1024 크기의 미니배치는 GPU 연산 유닛을 포화(Saturation)시켜 학습 속도를 극대화합니다.

### D. 모델 세부 설정: Independent LogStd & Separate Networks

  * **설정:** `free_log_std=True`, `vf_share_layers=False`
  * **근거 논문:** *Engstrom et al., "Implementation Matters in Deep Policy Gradients" (2020)*
  * **분석:**
      * **Independent LogStd:** 행동의 표준편차(탐색 범위)를 신경망 입력(State)과 독립적인 변수로 분리합니다. 이는 초기 탐색이 너무 빨리 줄어드는 것을 방지하여 Local Minima(잘못된 걷기 자세)에 빠지는 것을 막습니다.
      * **Separate Networks:** Actor(행동)와 Critic(가치 평가) 네트워크를 분리하여 파라미터 간섭을 없애 성능을 향상시킵니다. 대형 네트워크(`1024x1024`)를 사용하므로 메모리 이슈 없이 적용 가능합니다.

-----

## 4\. 결론 (Conclusion)

이 보고서의 설정은 \*\*"학습의 안정성(Large Batch)"\*\*과 **"모델의 표현력(Wide Network)"**, 그리고 \*\*"하드웨어 효율성(High Throughput)"\*\*의 세 가지 축을 중심으로 튜닝되었습니다.

Google Brain과 OpenAI의 연구 결과에 따라, 이 설정은 기존 베이스라인 대비 더 빠르고 안정적으로 3500점 이상의 고득점에 도달할 것으로 예상됩니다.