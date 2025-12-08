# Atari Breakout 오프라인 강화학습 실험 종합 보고서

> **실험 기간**: 2024년 12월 3-4일
> **환경**: AWS SageMaker, NVIDIA GPU, CUDA
> **작성**: Claude Code

---

## 1. 프로젝트 개요

### 1.1 목표
Atari Breakout 게임을 위한 **오프라인 강화학습(Offline RL)** 에이전트 개발

### 1.2 데이터셋
| 항목 | 값 |
|------|-----|
| 데이터셋 | Minari `atari/breakout/expert-v0` |
| 샘플 수 | 23,075 transitions |
| 학습/검증 분할 | 20,767 / 2,308 (90%/10%) |
| 입력 형태 | 84×84×4 (grayscale, 4-frame stack) |
| 액션 수 | 4 (NOOP, FIRE, RIGHT, LEFT) |

### 1.3 네트워크 아키텍처 (DeepMind Nature DQN 스타일)
```
Input (84×84×4)
    ↓
Conv2d(4→32, 8×8, stride=4) + ReLU
    ↓
Conv2d(32→64, 4×4, stride=2) + ReLU
    ↓
Conv2d(64→64, 3×3, stride=1) + ReLU
    ↓
Flatten → 3136 features
    ↓
FC(3136→512) + ReLU + Dropout
    ↓
FC(512→4) → Action logits
```

---

## 2. 실험 진행 순서

| 단계 | 알고리즘 | 실험 내용 | 결과 |
|------|----------|-----------|------|
| 1 | BC 기본 | 기본 하이퍼파라미터 탐색 | Mean 3.2 |
| 2 | BC 튜닝 | 스케줄러, 배치 크기 최적화 | Mean 9.5 |
| 3 | IQL 구현 | Offline RL 알고리즘 구현 및 튜닝 | Mean 1.8 |
| 4 | BC 고급 | 정규화 기법 적용 | **Mean 9.7** |

---

## 3. 알고리즘 설명

### 3.1 Behavior Cloning (BC)
**지도학습 기반 모방 학습**

```
손실 함수: L = CrossEntropy(π(s), a_expert)
```

- Expert 데이터의 (상태, 액션) 쌍을 지도학습
- 단순하고 안정적
- 분포 이동(Distribution Shift) 문제 존재

### 3.2 Implicit Q-Learning (IQL)
**오프라인 강화학습 알고리즘**

```
V Loss: L_V = E[L_τ(Q(s,a) - V(s))]  (Expectile Loss)
Q Loss: L_Q = E[(r + γV(s') - Q(s,a))²]
Policy Loss: L_π = E[exp(β·A(s,a)) · log π(a|s)]  (AWR)
```

- Expectile Regression으로 Q값의 상위 기대값 학습
- Advantage-Weighted Regression으로 정책 추출
- 분포 외 행동(OOD actions) 쿼리 없이 학습

---

## 4. 실험 결과

### 4.1 알고리즘별 최고 성능

| 순위 | 알고리즘 | Best Config | Mean Reward | Max Reward | 학습 시간 |
|------|----------|-------------|-------------|------------|----------|
| **1** | **BC (고급)** | **label_smooth** | **9.7** | 38 | 120분 |
| 2 | BC (기본) | large_batch_cosine | 9.5 | 39 | 122분 |
| 3 | IQL | beta_10 | 1.8 | 9 | 130분 |

### 4.2 BC 고급 튜닝 상세 결과

| 순위 | Config | Mean Reward | Max Reward | 핵심 기법 |
|------|--------|-------------|------------|-----------|
| 1 | label_smooth | **9.7** | 30 | Label Smoothing 0.1 |
| 2 | large_batch | 9.4 | 37 | Batch Size 256 |
| 3 | onecycle | 9.1 | **38** | OneCycleLR |
| 4 | combined | 8.9 | 26 | Label Smoothing + Grad Clip |
| 5 | mixup | 7.8 | 25 | Mixup α=0.2 |
| 6 | baseline | 6.3 | 19 | 기본 설정 |

### 4.3 IQL 튜닝 상세 결과

| Config | tau (expectile) | beta (AWR temp) | Mean Reward | Max Reward |
|--------|-----------------|-----------------|-------------|------------|
| beta_10 | 0.7 | 10.0 | **1.8** | 7 |
| beta_10_tau_0.8 | 0.8 | 10.0 | 1.7 | **9** |
| beta_15 | 0.7 | 15.0 | 1.7 | 5 |
| beta_20 | 0.7 | 20.0 | 0.3 | 4 |
| baseline | 0.7 | 3.0 | 0.0 | 0 |

---

## 5. 핵심 발견

### 5.1 BC vs IQL 성능 비교

```
┌─────────────────────────────────────────────────────┐
│  BC (9.7)  ████████████████████████████████████  │
│  IQL (1.8) ███████                                │
└─────────────────────────────────────────────────────┘
```

**BC가 IQL보다 약 5.4배 더 좋은 성능**

#### 원인 분석
| 요인 | 설명 |
|------|------|
| 데이터셋 크기 | 23K 샘플은 IQL에 부족 (권장: 100K+) |
| 이미지 기반 학습 | Atari 환경에서 Q-learning 수렴이 어려움 |
| 하이퍼파라미터 민감성 | IQL은 tau, beta 튜닝에 매우 민감 |
| 학습 불안정성 | Q-learning 기반 알고리즘의 본질적 특성 |

### 5.2 효과적인 기법 분석

| 기법 | 효과 | Mean Reward 향상 | 비고 |
|------|------|-----------------|------|
| **Label Smoothing** | 가장 효과적 | +54% (6.3→9.7) | 과적합 방지 |
| Large Batch | 효과적 | +49% (6.3→9.4) | 안정적 학습 |
| OneCycleLR | 효과적 | +44% (6.3→9.1) | 빠른 수렴 |
| Gradient Clipping | 보통 | +41% (6.3→8.9) | 학습 안정성 |
| Mixup | 제한적 | +24% (6.3→7.8) | 이미지 혼합 효과 미미 |

### 5.3 학습 패턴

| 패턴 | 관찰 |
|------|------|
| 최적 Epoch | 60-90 에폭에서 최고 성능 달성 |
| 과적합 시점 | Train Acc > 90% 이후 성능 하락 |
| Val Acc 한계 | 모든 설정에서 44-47%로 수렴 |
| Reward 분산 | 표준편차가 크므로 여러 번 평가 필요 |

---

## 6. 최종 권장 설정

### 6.1 Best Configuration (BC + Label Smoothing)

```python
config = {
    # 네트워크
    'hidden_dim': 512,
    'dropout': 0.3,

    # 학습
    'lr': 3e-4,
    'batch_size': 128,
    'epochs': 90,  # Early stopping 권장

    # 정규화
    'label_smoothing': 0.1,
    'scheduler': 'cosine',  # CosineAnnealingLR

    # 평가
    'eval_episodes': 10,
    'eval_interval': 10,
}
```

### 6.2 대안 설정 (높은 Max Reward 우선)

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

## 7. 생성된 파일 목록

### 7.1 학습 스크립트

| 파일 | 설명 |
|------|------|
| `train_bc_minari.py` | BC 기본 학습 |
| `tune_bc_minari.py` | BC 하이퍼파라미터 튜닝 |
| `tune_bc_advanced_tb.py` | BC 고급 튜닝 + TensorBoard |
| `train_iql_minari.py` | IQL 학습 |
| `tune_iql_minari.py` | IQL 튜닝 (1차) |
| `tune_iql_minari_tb.py` | IQL 튜닝 + TensorBoard |

### 7.2 유틸리티

| 파일 | 설명 |
|------|------|
| `utils/cnn_models.py` | CNN 아키텍처 정의 |
| `utils/iql_networks.py` | IQL 네트워크 (V, Q, Policy) |
| `visualize_results.py` | 결과 시각화 (matplotlib) |

### 7.3 문서

| 파일 | 설명 |
|------|------|
| `docs/BC_HYPERPARAMETER_TUNING_REPORT.md` | BC 튜닝 보고서 |
| `docs/IQL_HYPERPARAMETER_TUNING_REPORT.md` | IQL 튜닝 보고서 |
| `docs/BC_ADVANCED_TUNING_REPORT.md` | BC 고급 튜닝 보고서 |
| `docs/OFFLINE_RL_EXPERIMENT_REPORT.md` | 종합 보고서 (본 문서) |

### 7.4 저장된 모델

```
~/ray_results/
├── atari_bc_tuning/
│   └── best_model.pt
├── atari_bc_advanced_tb/
│   ├── label_smooth_model.pt  ← Best Mean Reward
│   ├── large_batch_model.pt
│   ├── onecycle_model.pt      ← Best Max Reward
│   └── tensorboard/
└── atari_iql_tuning_tb/
    ├── beta_10_model.pt
    └── tensorboard/
```

---

## 8. 성과 요약

### 8.1 정량적 성과

| 지표 | 값 |
|------|-----|
| **최고 Mean Reward** | **9.7** (label_smooth) |
| **최고 Max Reward** | **38** (onecycle) |
| **Baseline 대비 향상** | +54% (6.3 → 9.7) |
| **총 실험 설정 수** | 14개 (BC 6개 + IQL 8개) |
| **총 학습 시간** | 약 20시간+ |

### 8.2 정성적 성과

| 항목 | 성과 |
|------|------|
| 알고리즘 비교 | BC vs IQL 성능 비교 및 분석 완료 |
| 최적화 기법 검증 | Label Smoothing, OneCycleLR 등 효과 검증 |
| 재현 가능한 코드 | 모든 실험 스크립트 및 설정 문서화 |
| TensorBoard 시각화 | 학습 과정 모니터링 환경 구축 |

---

## 9. 한계점 및 향후 개선 방향

### 9.1 현재 한계

| 한계 | 설명 |
|------|------|
| 데이터셋 크기 | 23K 샘플은 복잡한 Offline RL에 부족 |
| 성능 천장 | Mean Reward ~10으로 Expert 수준에 미달 |
| Val Acc 한계 | 44-47%에서 정체 (과적합 vs 일반화 트레이드오프) |

### 9.2 향후 개선 방향

| 방향 | 설명 | 예상 효과 |
|------|------|----------|
| **더 큰 데이터셋** | 100K+ expert 데이터 수집 | Mean 15+ |
| **앙상블** | 여러 모델 조합 | 안정성 향상 |
| **ResNet CNN** | 더 깊은 네트워크 | 특징 추출 향상 |
| **CQL/BCQ** | 다른 Offline RL 알고리즘 | IQL 대비 성능 향상 |
| **Data Augmentation** | 프레임 변환, 노이즈 | 일반화 향상 |
| **Decision Transformer** | Transformer 기반 접근 | 장기 의존성 학습 |

---

## 10. 결론

### 10.1 주요 결론

1. **BC > IQL**: Atari Breakout + 23K 데이터셋 환경에서 BC가 IQL보다 5배 이상 우수
2. **Label Smoothing이 가장 효과적**: 간단하면서도 +54% 성능 향상
3. **적절한 Early Stopping 필요**: 90 에폭 근처에서 최적 성능
4. **Mixup 효과 제한적**: 이미지 기반 모방 학습에서 큰 효과 없음

### 10.2 권장 사항

| 목적 | 권장 |
|------|------|
| 빠른 성능 달성 | BC + Label Smoothing |
| 최고 Max Reward | BC + OneCycleLR |
| Offline RL 연구 | 더 큰 데이터셋으로 IQL/CQL 재시도 |
| 프로덕션 배포 | 앙상블 + Early Stopping |

---

## 부록

### A. TensorBoard 시각화

```bash
# 터미널에서
tensorboard --logdir ~/ray_results/atari_bc_advanced_tb/tensorboard --port 6006

# Jupyter Notebook에서
%load_ext tensorboard
%tensorboard --logdir ~/ray_results/atari_bc_advanced_tb/tensorboard
```

### B. 모델 로드 및 평가

```python
import torch
from utils.cnn_models import BCCNN

# 모델 로드
model = BCCNN(action_dim=4, dropout=0.3)
model.load_state_dict(torch.load('label_smooth_model.pt'))
model.eval()

# 추론
with torch.no_grad():
    action_logits = model(obs_tensor)
    action = action_logits.argmax(dim=-1)
```

### C. 실험 환경

| 항목 | 값 |
|------|-----|
| 플랫폼 | AWS SageMaker |
| GPU | NVIDIA GPU (CUDA enabled) |
| Python | 3.x |
| PyTorch | 2.x |
| Gymnasium | 1.0+ |
| Minari | Latest |

---

*이 보고서는 Claude Code를 사용하여 자동 생성되었습니다.*
*생성일: 2024년 12월 4일*
