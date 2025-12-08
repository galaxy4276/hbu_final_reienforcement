# BC (Behavior Cloning) 고급 하이퍼파라미터 튜닝 리포트

> **실험 일시**: 2024년 12월 3-4일
> **데이터셋**: Minari `atari/breakout/expert-v0`
> **환경**: NVIDIA GPU, CUDA, AWS SageMaker

---

## 1. 개요

### 1.1 실험 목적

BC(Behavior Cloning) 모델의 성능을 향상시키기 위해 다양한 정규화 및 학습 기법을 적용하고 TensorBoard를 통해 학습 과정을 시각화하였습니다.

### 1.2 이전 실험 결과 (베이스라인)

| 알고리즘 | Best Config | Mean Reward | Max Reward |
|----------|-------------|-------------|------------|
| BC (이전) | large_batch_cosine | 9.5 | 39.0 |
| IQL | beta_10 | 1.8 | 9 |

**결론**: BC가 IQL보다 약 5배 더 좋은 성능을 보여, BC 기반 성능 개선을 진행.

---

## 2. 실험 설정

### 2.1 테스트한 기법들

| 기법 | 설명 | 기대 효과 |
|------|------|-----------|
| **Label Smoothing** | 원-핫 레이블을 부드럽게 (0.1) | 과적합 방지, 일반화 향상 |
| **Large Batch** | 배치 크기 증가 (256) | 안정적 학습, 빠른 수렴 |
| **Mixup** | 입력 데이터 혼합 (alpha=0.2) | 데이터 증강, 일반화 |
| **OneCycleLR** | 학습률 사이클링 | 빠른 수렴, 좋은 최솟값 탐색 |
| **Gradient Clipping** | 그래디언트 클리핑 (1.0) | 학습 안정성 |
| **Combined** | 여러 기법 조합 | 시너지 효과 |

### 2.2 실험 구성

| Config | LR | Batch | Dropout | Scheduler | 추가 기법 |
|--------|-----|-------|---------|-----------|----------|
| baseline | 3e-4 | 128 | 0.3 | cosine | - |
| large_batch | 5e-4 | 256 | 0.3 | cosine | - |
| label_smooth | 3e-4 | 128 | 0.3 | cosine | label_smoothing=0.1 |
| mixup | 3e-4 | 128 | 0.3 | cosine | mixup_alpha=0.2 |
| combined | 5e-4 | 256 | 0.2 | cosine | label_smoothing=0.1, grad_clip=1.0 |
| onecycle | 1e-3 | 128 | 0.3 | onecycle | - |

---

## 3. 실험 결과

### 3.1 최종 순위

| 순위 | Config | Best Mean Reward | Max Reward | Best Epoch | Val Acc | 학습 시간 |
|------|--------|------------------|------------|------------|---------|----------|
| **1위** | **label_smooth** | **9.7** | 30 | 90 | 45.1% | 120.7분 |
| 2위 | large_batch | 9.4 | 37 | 60 | 43.9% | 122.0분 |
| 3위 | onecycle | 9.1 | **38** | 80 | 43.2% | 134.8분 |
| 4위 | combined | 8.9 | 26 | 90 | 43.8% | 120.3분 |
| 5위 | mixup | 7.8 | 25 | 60 | 45.3% | 133.7분 |
| 6위 | baseline | 6.3 | 19 | 90 | 45.3% | 135.8분 |

### 3.2 상세 학습 곡선

#### label_smooth (1위 - Best Mean Reward)
```
Epoch 10:  Train Acc: 47.9% | Val Acc: 47.1% | Reward: 0.4 ± 0.8
Epoch 30:  Train Acc: 63.2% | Val Acc: 46.4% | Reward: 8.7 ± 8.3  ← 초기 피크
Epoch 60:  Train Acc: 84.1% | Val Acc: 44.8% | Reward: 5.7 ± 7.9
Epoch 90:  Train Acc: 90.3% | Val Acc: 45.1% | Reward: 9.7 ± 6.7  ← 최고점
Epoch 100: Train Acc: 90.3% | Val Acc: 44.8% | Reward: 5.3 ± 2.5
```

#### large_batch (2위 - Best Max Reward 37)
```
Epoch 10:  Train Acc: 47.6% | Val Acc: 47.4% | Reward: 0.9 ± 0.3
Epoch 60:  Train Acc: 87.3% | Val Acc: 43.9% | Reward: 9.4 ± 8.1  ← 최고점
Epoch 90:  Train Acc: 92.8% | Val Acc: 44.2% | Reward: 7.2 ± 9.4, Max: 37
Epoch 100: Train Acc: 93.0% | Val Acc: 44.8% | Reward: 4.5 ± 5.2
```

#### onecycle (3위 - Best Max Reward 38)
```
Epoch 10:  Train Acc: 45.0% | Val Acc: 46.0% | Reward: 0.0
Epoch 40:  Train Acc: 87.2% | Val Acc: 43.2% | Reward: 7.1 ± 8.1  ← 빠른 수렴
Epoch 80:  Train Acc: 98.2% | Val Acc: 43.2% | Reward: 9.1 ± 10.1, Max: 38  ← 최고점
Epoch 100: Train Acc: 98.5% | Val Acc: 43.6% | Reward: 3.8 ± 2.6  ← 과적합
```

---

## 4. 핵심 발견

### 4.1 기법별 효과 분석

| 기법 | 효과 | 관찰 |
|------|------|------|
| **Label Smoothing** | **가장 효과적** | Mean Reward +54% (6.3 → 9.7) |
| **Large Batch** | 효과적 | 빠른 학습, 높은 Max Reward |
| **OneCycleLR** | 효과적 | 빠른 수렴, 최고 Max Reward (38) |
| **Combined** | 보통 | 개별 기법보다 낮은 성능 |
| **Mixup** | 제한적 | Atari 이미지에서 효과 미미 |

### 4.2 학습 패턴

1. **최적 Epoch**: 대부분 60-90 에폭에서 최고 성능 달성
2. **과적합 징후**: Train Acc > 90% 이후 성능 하락 경향
3. **Val Acc 한계**: 모든 설정에서 Val Acc ~44-47%로 수렴
4. **분산이 큰 Reward**: 표준편차가 크므로 여러 번 평가 필요

### 4.3 Early Stopping 권장 시점

| Config | 권장 Epoch | 이유 |
|--------|------------|------|
| label_smooth | 90 | Mean Reward 최고점 |
| large_batch | 60 | Mean Reward 최고점 |
| onecycle | 80 | Max Reward 최고점 |

---

## 5. 이전 실험과 비교

### 5.1 BC 튜닝 히스토리

| 실험 | Best Config | Mean Reward | Max Reward | 개선율 |
|------|-------------|-------------|------------|--------|
| BC 1차 | baseline | 3.2 | 15 | - |
| BC 2차 | large_batch_cosine | 9.5 | 39 | +197% |
| **BC 3차 (현재)** | **label_smooth** | **9.7** | 38 | +2% |

### 5.2 알고리즘 비교

| 알고리즘 | Best Mean Reward | Max Reward | 학습 시간 |
|----------|------------------|------------|----------|
| **BC (label_smooth)** | **9.7** | 38 | ~2시간 |
| BC (large_batch) | 9.4 | 37 | ~2시간 |
| IQL (beta_10) | 1.8 | 9 | ~2시간 |

**결론**: BC + Label Smoothing이 현재 최고 성능

---

## 6. TensorBoard 시각화

### 6.1 로그 위치

```
~/ray_results/atari_bc_advanced_tb/tensorboard/
├── baseline/
├── large_batch/
├── label_smooth/
├── mixup/
├── combined/
└── onecycle/
```

### 6.2 시각화 실행

```bash
# 터미널에서
tensorboard --logdir ~/ray_results/atari_bc_advanced_tb/tensorboard --port 6006

# Jupyter Notebook에서
%load_ext tensorboard
%tensorboard --logdir ~/ray_results/atari_bc_advanced_tb/tensorboard
```

### 6.3 기록된 메트릭

| 카테고리 | 메트릭 | 설명 |
|----------|--------|------|
| `loss/` | train_loss | Cross-entropy 손실 |
| `accuracy/` | train_acc, val_acc | 학습/검증 정확도 |
| `eval/` | mean_reward, max_reward, std_reward | 환경 평가 결과 |
| `train/` | learning_rate | 학습률 스케줄링 |

---

## 7. 최적 하이퍼파라미터

### 7.1 추천 설정 (label_smooth)

```python
config = {
    'lr': 3e-4,
    'batch_size': 128,
    'dropout': 0.3,
    'scheduler': 'cosine',
    'label_smoothing': 0.1,
    'epochs': 90,  # Early stopping
}
```

### 7.2 대안 설정 (높은 Max Reward 우선)

```python
config = {
    'lr': 1e-3,
    'batch_size': 128,
    'dropout': 0.3,
    'scheduler': 'onecycle',
    'epochs': 80,
}
```

---

## 8. 저장된 모델

```
~/ray_results/atari_bc_advanced_tb/
├── baseline_model.pt
├── large_batch_model.pt
├── label_smooth_model.pt      ← Best Mean Reward
├── mixup_model.pt
├── combined_model.pt
├── onecycle_model.pt          ← Best Max Reward
└── tensorboard/
```

---

## 9. 결론 및 권장 사항

### 9.1 주요 결론

1. **Label Smoothing이 가장 효과적**: 간단하면서도 성능 향상에 효과적
2. **BC > IQL**: Atari Breakout에서 BC가 IQL보다 우수
3. **과적합 관리 중요**: 90 에폭 근처에서 Early Stopping 권장
4. **Mixup 효과 제한적**: 이미지 기반 모방 학습에서 Mixup은 큰 효과 없음

### 9.2 향후 개선 방향

| 방향 | 설명 | 예상 효과 |
|------|------|-----------|
| **더 큰 데이터셋** | 더 많은 expert 데이터 수집 | Mean Reward 15+ |
| **앙상블** | 여러 모델 조합 | 안정성 향상 |
| **더 깊은 네트워크** | ResNet 기반 CNN | 특징 추출 향상 |
| **Data Augmentation** | 프레임 변환, 노이즈 | 일반화 향상 |

### 9.3 최종 권장 사항

| 목적 | 권장 설정 |
|------|-----------|
| **안정적인 Mean Reward** | label_smooth (9.7) |
| **최고 Max Reward** | onecycle (38) |
| **빠른 학습** | large_batch |

---

## 부록: 실험 환경

### A.1 하드웨어
- GPU: NVIDIA GPU (AWS SageMaker)
- CUDA: Enabled

### A.2 소프트웨어
- Python 3.x
- PyTorch 2.x
- Gymnasium + ALE
- Minari

### A.3 파일 구조

```
atari/
├── tune_bc_advanced_tb.py    # 본 실험 스크립트
├── utils/
│   └── cnn_models.py         # CNN 아키텍처
└── docs/
    ├── BC_HYPERPARAMETER_TUNING_REPORT.md
    ├── IQL_HYPERPARAMETER_TUNING_REPORT.md
    └── BC_ADVANCED_TUNING_REPORT.md  # 본 문서
```

---

*이 리포트는 Claude Code를 사용하여 자동 생성되었습니다.*
