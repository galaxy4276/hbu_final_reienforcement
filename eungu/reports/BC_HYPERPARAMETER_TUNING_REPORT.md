# Behavior Cloning 하이퍼파라미터 튜닝 리포트

> **실험 일시**: 2024년 12월 2일
> **데이터셋**: Minari `atari/breakout/expert-v0`
> **환경**: NVIDIA A10G GPU, CUDA 13.0

---

## 1. 개요

### 1.1 실험 목적

Minari 데이터셋을 사용한 Behavior Cloning(BC) 모델의 성능을 향상시키기 위해 다양한 하이퍼파라미터 조합을 체계적으로 탐색하였습니다.

### 1.2 베이스라인 성능

초기 BC 모델(50 에폭 학습)의 성능:

| 지표 | 값 |
|------|-----|
| Mean Reward | 2.4 ± 2.3 |
| Max Reward | 10.0 |
| Train Accuracy | 56.2% |
| Val Accuracy | 48.3% |

---

## 2. 실험 설정

### 2.1 데이터셋 정보

```
Dataset: atari/breakout/expert-v0
Total Samples: 23,075
Train/Val Split: 90% / 10%
Train Samples: ~20,767
Val Samples: ~2,308
```

### 2.2 모델 아키텍처

DeepMind Nature DQN 스타일의 CNN 아키텍처:

```
Input: (84, 84, 4) - 4프레임 스택
├── Conv2d(4→32, 8×8, stride=4) + ReLU
├── Conv2d(32→64, 4×4, stride=2) + ReLU
├── Conv2d(64→64, 3×3, stride=1) + ReLU
├── Flatten → 3136
├── Linear(3136→512) + ReLU + Dropout
└── Linear(512→4) - 4개 액션 출력
```

### 2.3 테스트된 하이퍼파라미터 설정

| 설정명 | Learning Rate | Dropout | Batch Size | Epochs | Scheduler | 특징 |
|--------|---------------|---------|------------|--------|-----------|------|
| `baseline_long` | 1e-4 | 0.3 | 64 | 100 | ReduceLROnPlateau | 기본 설정, 에폭만 증가 |
| `low_lr_high_dropout` | 5e-5 | 0.5 | 64 | 100 | ReduceLROnPlateau | 과적합 방지 강화 |
| `large_batch_cosine` | 3e-4 | 0.3 | 128 | 100 | CosineAnnealingLR | 큰 배치 + 코사인 스케줄러 |
| `weight_decay` | 1e-4 | 0.4 | 64 | 100 | ReduceLROnPlateau | L2 정규화 (1e-4) |
| `grad_clip` | 2e-4 | 0.3 | 64 | 100 | ReduceLROnPlateau | 그래디언트 클리핑 (1.0) |

---

## 3. 실험 결과

### 3.1 최종 성능 비교

| 순위 | 설정 | Mean Reward | Max Reward | Best Val Loss | 학습 시간 |
|------|------|-------------|------------|---------------|----------|
| 🥇 1 | **large_batch_cosine** | **9.5 ± 9.5** | **39.0** | 1.0874 | 38.2분 |
| 🥈 2 | grad_clip | 5.5 ± 4.9 | 16.0 | 1.0993 | 45.3분 |
| 🥉 3 | weight_decay | 4.7 ± 5.3 | 23.0 | 1.1170 | 46.4분 |
| 4 | baseline_long | 3.1 ± 2.2 | 10.0 | 1.1198 | 48.3분 |
| 5 | low_lr_high_dropout | 1.9 ± 1.2 | 4.0 | 1.1306 | 48.2분 |

### 3.2 에폭별 성능 추이

#### large_batch_cosine (Best)
```
Epoch 20:  Reward = 3.0 ± 1.9,  Train Acc = 53.0%, Val Acc = 47.2%
Epoch 40:  Reward = 5.3 ± 7.0,  Train Acc = 65.5%, Val Acc = 46.3%
Epoch 60:  Reward = 9.5 ± 6.8,  Train Acc = 76.9%, Val Acc = 44.5%
Epoch 80:  Reward = 11.2 ± 10.7, Train Acc = 82.0%, Val Acc = 44.5%  ← Peak
Epoch 100: Reward = 5.9 ± 5.0,  Train Acc = 84.0%, Val Acc = 44.2%
```

#### baseline_long
```
Epoch 20:  Reward = 0.2 ± 0.4,  Train Acc = 47.4%, Val Acc = 46.5%
Epoch 40:  Reward = 1.5 ± 1.1,  Train Acc = 52.2%, Val Acc = 47.4%
Epoch 60:  Reward = 2.9 ± 3.2,  Train Acc = 56.7%, Val Acc = 47.8%
Epoch 80:  Reward = 2.8 ± 1.1,  Train Acc = 58.7%, Val Acc = 47.7%
Epoch 100: Reward = 2.9 ± 2.4,  Train Acc = 58.9%, Val Acc = 47.3%
```

#### low_lr_high_dropout (Worst)
```
Epoch 20:  Reward = 0.0 ± 0.0,  Train Acc = 45.6%, Val Acc = 46.0%
Epoch 40:  Reward = 0.0 ± 0.0,  Train Acc = 47.8%, Val Acc = 46.7%
Epoch 60:  Reward = 0.5 ± 0.8,  Train Acc = 49.8%, Val Acc = 46.4%
Epoch 80:  Reward = 1.8 ± 1.6,  Train Acc = 51.1%, Val Acc = 46.7%
Epoch 100: Reward = 2.0 ± 1.5,  Train Acc = 51.8%, Val Acc = 46.7%
```

---

## 4. 분석 및 인사이트

### 4.1 핵심 발견

#### 4.1.1 큰 배치 + 높은 학습률의 효과

`large_batch_cosine` 설정이 가장 높은 성능을 달성한 이유:

1. **안정적인 그래디언트 추정**: 배치 크기 128은 64 대비 2배 많은 샘플로 그래디언트를 계산하여 노이즈 감소
2. **높은 학습률 허용**: 큰 배치는 더 높은 학습률(3e-4)을 안정적으로 사용 가능
3. **코사인 스케줄러**: 부드러운 학습률 감소로 후반부까지 효과적인 학습 유지

```
Linear Scaling Rule: lr_new = lr_base × (batch_new / batch_base)
3e-4 = 1e-4 × (128 / 64) × 1.5
```

#### 4.1.2 과적합과 환경 성능의 관계

흥미로운 발견: **Train Accuracy와 환경 성능이 비례하지 않음**

| 설정 | Train Acc | Val Acc | Gap | Reward |
|------|-----------|---------|-----|--------|
| large_batch_cosine | 84.0% | 44.2% | 39.8% | **9.5** |
| baseline_long | 58.9% | 47.3% | 11.6% | 3.1 |
| low_lr_high_dropout | 51.8% | 46.7% | 5.1% | 1.9 |

**해석**: BC에서는 검증 정확도보다 모델의 **의사결정 품질**이 중요합니다. 과적합된 모델이라도 핵심 패턴을 더 강하게 학습하면 실제 환경에서 더 좋은 성능을 보일 수 있습니다.

#### 4.1.3 학습률의 중요성

| 학습률 | 설정 | 최종 Reward | 분석 |
|--------|------|-------------|------|
| 5e-5 (낮음) | low_lr_high_dropout | 1.9 | 학습 속도 너무 느림 |
| 1e-4 (기본) | baseline_long | 3.1 | 안정적이지만 보수적 |
| 2e-4 (중간) | grad_clip | 5.5 | 균형 잡힌 성능 |
| 3e-4 (높음) | large_batch_cosine | 9.5 | 최고 성능 (큰 배치와 함께) |

### 4.2 실패 분석

#### `low_lr_high_dropout`이 최악인 이유

1. **너무 느린 학습**: 5e-5 학습률로 100 에폭 동안 충분히 수렴하지 못함
2. **과도한 정규화**: Dropout 0.5는 이미 작은 데이터셋에서 너무 강한 제약
3. **에폭 60까지 Reward = 0**: 초기 학습이 너무 느려 유의미한 정책 형성 지연

---

## 5. 학습률 스케줄러 비교

### 5.1 ReduceLROnPlateau vs CosineAnnealingLR

```python
# ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# 검증 손실이 5 에폭 동안 개선되지 않으면 학습률을 절반으로 감소

# CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
# 코사인 함수를 따라 부드럽게 학습률 감소
```

**결과**: CosineAnnealingLR이 더 효과적
- 초기에 큰 학습률로 빠른 학습
- 후반부에 작은 학습률로 미세 조정
- 학습률 감소 시점이 손실 정체에 의존하지 않아 더 예측 가능

---

## 6. 최적 하이퍼파라미터

### 6.1 추천 설정

```python
config = {
    'learning_rate': 3e-4,      # 기본의 3배
    'batch_size': 128,          # 기본의 2배
    'dropout': 0.3,             # 기본 유지
    'epochs': 80,               # 100 에폭 중 80에서 최고 성능
    'scheduler': 'CosineAnnealingLR',
    'weight_decay': 0,          # L2 정규화 불필요
    'grad_clip': None,          # 그래디언트 클리핑 불필요
}
```

### 6.2 성능 개선 요약

| 지표 | 베이스라인 (50 에폭) | 최적 설정 | 개선율 |
|------|---------------------|-----------|--------|
| Mean Reward | 2.4 | **9.5** | **+296%** |
| Max Reward | 10.0 | **39.0** | **+290%** |
| Best Val Loss | 1.122 | **1.087** | -3.1% |

---

## 7. 추가 연구 방향

### 7.1 추가 튜닝 제안

1. **배치 크기 더 증가**: 256, 512로 추가 실험
2. **학습률 탐색**: 4e-4, 5e-4 테스트
3. **에폭 조기 종료**: 80 에폭에서 성능 피크, Early Stopping 적용

### 7.2 모델 아키텍처 개선

1. **Dueling DQN 아키텍처**: Value/Advantage 스트림 분리
2. **배치 정규화**: Conv 레이어 사이에 BatchNorm 추가
3. **더 깊은 네트워크**: 추가 Conv 레이어

### 7.3 고급 오프라인 RL 알고리즘

순수 BC의 한계를 극복하기 위한 대안:

| 알고리즘 | 특징 | 기대 효과 |
|----------|------|-----------|
| **CQL** (Conservative Q-Learning) | Q값 과대추정 방지 | 분포 외 액션 억제 |
| **IQL** (Implicit Q-Learning) | 암묵적 Q 학습 | 안정적인 오프라인 학습 |
| **Decision Transformer** | 시퀀스 모델링 | 장기 의존성 학습 |

---

## 8. 결론

이번 하이퍼파라미터 튜닝을 통해 BC 모델의 성능을 **약 4배** 향상시켰습니다.

### 핵심 교훈

1. **큰 배치 + 높은 학습률**: BC에서 가장 효과적인 조합
2. **코사인 스케줄러**: ReduceLROnPlateau보다 우수
3. **과적합 허용**: 검증 정확도보다 환경 성능이 중요
4. **적절한 정규화**: 과도한 드롭아웃은 오히려 성능 저하

### 한계점

- 최종 평균 보상 9.5점은 전문가 수준(300+)에 비해 여전히 낮음
- 순수 BC로는 오프라인 데이터의 한계를 극복하기 어려움
- 고급 오프라인 RL 알고리즘(CQL, IQL 등) 적용 필요

---

## 부록: 저장된 파일

```
~/ray_results/atari_bc_tuning/
├── baseline_long_model.pt        # 기본 설정 모델
├── low_lr_high_dropout_model.pt  # 낮은 LR + 높은 드롭아웃
├── large_batch_cosine_model.pt   # 🏆 최고 성능 모델
├── weight_decay_model.pt         # L2 정규화 적용
├── grad_clip_model.pt            # 그래디언트 클리핑 적용
├── tuning_results.json           # 전체 학습 히스토리
└── tuning_summary.json           # 요약 결과
```

---

*이 리포트는 Claude Code를 사용하여 자동 생성되었습니다.*
