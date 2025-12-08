# IQL (Implicit Q-Learning) 하이퍼파라미터 튜닝 리포트

> **실험 일시**: 2024년 12월 3일
> **데이터셋**: Minari `atari/breakout/expert-v0`
> **환경**: NVIDIA GPU, CUDA, AWS SageMaker

---

## 1. 개요

### 1.1 실험 목적

BC(Behavior Cloning) 모델의 한계를 극복하기 위해 IQL(Implicit Q-Learning) 알고리즘을 구현하고 하이퍼파라미터를 튜닝하였습니다.

### 1.2 IQL 알고리즘 특징

IQL은 오프라인 강화학습 알고리즘으로, 다음과 같은 특징을 가집니다:

- **Expectile Regression**: Q값의 상위 기대값을 학습하여 최적 행동 강조
- **Advantage-Weighted Regression (AWR)**: advantage 기반 가중치로 정책 학습
- **분포 외 행동 회피**: 데이터에 없는 행동에 대한 Q값 쿼리 없이 학습

### 1.3 BC 베이스라인 성능

| 지표 | BC (large_batch_cosine) |
|------|-------------------------|
| Mean Reward | **9.5** |
| Max Reward | **39.0** |
| 학습 시간 | 38.2분 |

---

## 2. 네트워크 아키텍처

### 2.1 공유 CNN + 3개 헤드 구조

```
                        ┌→ V Head: FC(512) → 1 (상태 가치)
Input (84,84,4) → CNN → ├→ Q Head: FC(512) → 4 (액션별 Q값)
                        └→ Policy Head: FC(512) → Dropout → 4 (액션 로짓)
```

### 2.2 CNN 인코더 (DeepMind Nature DQN 스타일)

```
Conv2d(4→32, 8×8, stride=4) + ReLU
Conv2d(32→64, 4×4, stride=2) + ReLU
Conv2d(64→64, 3×3, stride=1) + ReLU
Flatten → 3136 features
```

### 2.3 Target Network

- Q-head와 Encoder만 포함 (V-head, Policy-head 제외)
- Soft update (Polyak averaging): τ = 0.005

---

## 3. 실험 설정

### 3.1 1차 튜닝 (기본 탐색)

| 설정명 | tau (expectile) | beta (AWR temp) | 결과 |
|--------|-----------------|-----------------|------|
| baseline | 0.7 | 3.0 | 0.0 |
| high_tau | 0.9 | 3.0 | 0.0 |
| high_beta | 0.7 | 10.0 | **0.7** |
| combined | 0.9 | 5.0 | 0.0 |

**1차 결론**: beta 값이 높을수록 성능이 좋음 (beta=10이 유일하게 성능 발휘)

### 3.2 2차 튜닝 (TensorBoard 로깅 + 높은 beta 탐색)

| 설정명 | tau | beta | Best Reward | Max Reward | 학습 시간 |
|--------|-----|------|-------------|------------|-----------|
| **beta_10** | 0.7 | 10.0 | **1.8** | 7 | 130.3분 |
| beta_10_tau_0.8 | 0.8 | 10.0 | 1.7 | **9** | 129.2분 |
| beta_15 | 0.7 | 15.0 | 1.7 | 5 | 85.4분 |
| beta_20 | 0.7 | 20.0 | 0.3 | 4 | 146.9분 |

---

## 4. 상세 결과 분석

### 4.1 beta_10 (Best Configuration)

```
Epoch 10-50:  Reward = 0.0 (학습 초기)
Epoch 60:     Reward = 1.6 ± 0.8, Max = 2  ← 성능 시작
Epoch 70:     Reward = 1.8 ± 0.4, Max = 2  ← 최고점
Epoch 80:     Reward = 0.5 ± 0.9, Max = 2  ← 하락
Epoch 90:     Reward = 0.2 ± 0.5, Max = 2
Epoch 100:    Reward = 0.9 ± 1.9, Max = 7
```

**관찰**: epoch 60-70에서 피크, 이후 불안정

### 4.2 beta_15

```
Epoch 60:     Reward = 0.8 ± 1.3, Max = 4
Epoch 70:     Reward = 0.4 ± 0.5, Max = 1
Epoch 80:     Reward = 0.8 ± 0.7, Max = 2
Epoch 90:     Reward = 1.5 ± 1.3, Max = 5
Epoch 100:    Reward = 1.7 ± 0.6, Max = 3  ← 후반부 상승
```

**관찰**: 후반부(epoch 90-100)에서 점진적 개선

### 4.3 beta_20 (너무 높음)

```
Epoch 60:     Reward = 0.0
Epoch 70:     Reward = 0.0
Epoch 80:     Reward = 0.2 ± 0.4, Max = 1
Epoch 90:     Reward = 0.0
Epoch 100:    Reward = 0.3 ± 1.0, Max = 4
```

**관찰**: beta가 너무 높으면 학습 불안정

### 4.4 beta_10_tau_0.8

```
Epoch 50:     Reward = 1.7 ± 2.6, Max = 9  ← 가장 높은 Max
Epoch 60:     Reward = 0.0
Epoch 70:     Reward = 0.5 ± 1.0, Max = 3
Epoch 80-100: Reward < 0.5 (불안정)
```

**관찰**: 높은 Max를 달성하지만 불안정

---

## 5. 핵심 발견

### 5.1 하이퍼파라미터 영향

| 파라미터 | 최적 범위 | 영향 |
|----------|-----------|------|
| **beta** | 10-15 | 너무 낮으면(3) 학습 안됨, 너무 높으면(20) 불안정 |
| **tau** | 0.7-0.8 | tau=0.7이 더 안정적, tau=0.8은 높은 분산 |

### 5.2 학습 패턴

1. **지연된 성능 발현**: epoch 50-60까지 reward = 0, 이후 성능 시작
2. **불안정한 학습**: epoch 70 이후 성능 변동이 큼
3. **Early Stopping 필요**: epoch 70 근처에서 최적 성능

### 5.3 BC vs IQL 비교

| 알고리즘 | Best Config | Mean Reward | Max Reward | 특징 |
|----------|-------------|-------------|------------|------|
| **BC** | large_batch_cosine | **9.5** | **39.0** | 안정적, 단순 |
| **IQL** | beta_10 | 1.8 | 9 | 불안정, 복잡 |

**BC가 IQL보다 약 5배 더 좋은 성능**을 보입니다.

---

## 6. IQL 성능이 낮은 이유 분석

### 6.1 가능한 원인

1. **데이터셋 크기**: 23K 샘플은 IQL에게 부족할 수 있음
2. **Atari 환경 특성**: 이미지 기반 환경에서 IQL 학습이 어려움
3. **하이퍼파라미터 민감성**: IQL은 BC보다 튜닝에 민감
4. **학습 불안정성**: Q-learning 기반 알고리즘의 본질적 불안정성

### 6.2 개선 방향

1. **더 긴 학습**: 200-500 에폭 시도
2. **네트워크 크기 증가**: hidden_dim 1024
3. **Double Q-learning**: 두 개의 Q-network 사용
4. **데이터 증강**: 프레임 변환, 노이즈 추가

---

## 7. TensorBoard 시각화

### 7.1 로그 위치

```
~/ray_results/atari_iql_tuning_tb/tensorboard/
├── beta_10/
├── beta_15/
├── beta_20/
└── beta_10_tau_0.8/
```

### 7.2 시각화 실행 (Jupyter Notebook)

```python
%load_ext tensorboard
%tensorboard --logdir ~/ray_results/atari_iql_tuning_tb/tensorboard
```

### 7.3 기록된 메트릭

| 카테고리 | 메트릭 | 설명 |
|----------|--------|------|
| `loss/` | v_loss, q_loss, policy_loss | 각 네트워크 손실값 |
| `eval/` | mean_reward, max_reward | 환경 평가 결과 |
| `train/` | learning_rate_* | 학습률 스케줄링 |

---

## 8. 최적 하이퍼파라미터

### 8.1 IQL 추천 설정

```python
config = {
    'tau': 0.7,              # Expectile
    'beta': 10.0,            # AWR temperature
    'gamma': 0.99,           # Discount factor
    'lr': 3e-4,              # Learning rate
    'batch_size': 256,
    'epochs': 70,            # Early stopping at peak
    'target_update_rate': 0.005,  # Soft update tau
}
```

### 8.2 저장된 모델

```
~/ray_results/atari_iql_tuning_tb/
├── beta_10_model.pt          ← Best model
├── beta_15_model.pt
├── beta_20_model.pt
├── beta_10_tau_0.8_model.pt
├── tuning_summary.json
└── tuning_history.json
```

---

## 9. 결론

### 9.1 주요 결론

1. **BC > IQL**: 현재 데이터셋과 환경에서 BC가 IQL보다 훨씬 우수
2. **beta=10이 최적**: IQL에서 AWR temperature가 중요한 하이퍼파라미터
3. **학습 불안정**: IQL은 BC보다 학습이 불안정하고 튜닝이 어려움

### 9.2 권장 사항

| 목적 | 권장 알고리즘 |
|------|---------------|
| **빠른 성능 달성** | BC (large_batch_cosine) |
| **오프라인 RL 연구** | IQL (beta=10, tau=0.7) |
| **더 높은 성능 탐색** | CQL, BCQ, Decision Transformer |

### 9.3 다음 단계 옵션

1. **BC 모델 최종 사용**: Mean Reward 9.5로 현재 최고 성능
2. **CQL 구현**: Conservative Q-Learning으로 Q값 과대추정 방지
3. **앙상블**: BC + IQL 모델 결합
4. **더 큰 데이터셋**: 더 많은 expert 데이터 수집

---

## 부록: 파일 구조

```
atari/
├── train_iql_minari.py           # IQL 학습 스크립트
├── tune_iql_minari.py            # IQL 튜닝 (1차)
├── tune_iql_minari_tb.py         # IQL 튜닝 + TensorBoard
├── utils/
│   └── iql_networks.py           # IQL 네트워크 정의
└── docs/
    ├── BC_HYPERPARAMETER_TUNING_REPORT.md
    └── IQL_HYPERPARAMETER_TUNING_REPORT.md  # 본 문서
```

---

*이 리포트는 Claude Code를 사용하여 자동 생성되었습니다.*
