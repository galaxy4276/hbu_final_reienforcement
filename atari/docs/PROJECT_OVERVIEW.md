# Atari Breakout 오프라인 강화학습 프로젝트 개요

## 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **프로젝트명** | Atari Breakout Offline Reinforcement Learning Pipeline |
| **환경** | ALE/Breakout-v5 (Atari Learning Environment) |
| **프레임워크** | Ray RLlib 2.50.1, PyTorch 2.0+ |
| **목표** | 오프라인 강화학습을 통한 Atari 게임 에이전트 학습 |

---

## 목적

### 핵심 목표

**"실시간 환경 상호작용 없이 사전 수집된 데이터만으로 게임 에이전트를 학습"**

이 프로젝트는 오프라인 강화학습(Offline RL)의 3단계 파이프라인을 구현합니다:

1. **온라인 전문가 학습**: DQN으로 전문가 정책 학습
2. **데이터 수집**: 전문가 에피소드 녹화 및 저장
3. **오프라인 학습**: 수집된 데이터로 Behavior Cloning

### 왜 오프라인 강화학습인가?

| 온라인 RL | 오프라인 RL |
|-----------|-------------|
| 실시간 환경 필요 | 사전 데이터만 사용 |
| 탐색 비용 높음 | 탐색 비용 없음 |
| 안전성 문제 | 안전한 학습 |
| 시뮬레이터 필수 | 실제 데이터 활용 가능 |

---

## 시스템 아키텍처

### 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    Atari RL Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │      │
│  │  DQN Expert  │───▶│ Data Record  │───▶│  Offline BC  │      │
│  │   Training   │    │              │    │   Training   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                    │                    │              │
│        ▼                    ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Checkpoint  │    │   Parquet    │    │   BC Model   │      │
│  │    (.pt)     │    │   Dataset    │    │  Checkpoint  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 디렉토리 구조

```
atari/
├── sagemaker/                    # AWS SageMaker 배포용
│   ├── train_expert.py          # Stage 1: DQN 전문가 학습
│   ├── record_data.py           # Stage 2: 데이터 수집
│   ├── train_offline.py         # Stage 3: BC 오프라인 학습
│   ├── launch_sagemaker.py      # SageMaker 작업 실행
│   ├── visualize_tensorboard.py # 시각화 도구
│   ├── configs/                 # 설정 모듈
│   │   └── dqn_breakout_config.py
│   ├── utils/                   # 유틸리티 모듈
│   │   ├── preprocess.py        # 이미지 전처리
│   │   └── cnn_models.py        # CNN 모델 정의
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── atari_1_Training_DQN_expert.py    # 로컬용 Stage 1
├── atari_2_Record_visual_data.py     # 로컬용 Stage 2
├── atari_3_Training_on_*.py          # 로컬용 Stage 3
├── configs/                          # 로컬용 설정
├── utils/                            # 로컬용 유틸리티
├── results/                          # 결과 분석
└── docs/                             # 문서
```

---

## 각 단계별 상세

### Stage 1: DQN 전문가 학습

**목적**: CNN 기반 DQN으로 Breakout 게임 전문가 정책 학습

**알고리즘**: Deep Q-Network (DQN)
- Double Q-Learning
- Experience Replay (1M buffer)
- Epsilon-Greedy 탐색

**입력/출력**:
- 입력: 게임 화면 (84×84×4 프레임 스택)
- 출력: 4개 액션 Q-값 (NOOP, FIRE, LEFT, RIGHT)

**하이퍼파라미터**:
```python
learning_rate: 2.5e-4
batch_size: 32
gamma: 0.99
epsilon: 1.0 → 0.01 (1M steps)
target_update_freq: 1000
buffer_size: 1,000,000
```

**성능 목표**: 평균 300점 (인간 수준)

---

### Stage 2: 전문가 데이터 수집

**목적**: 학습된 전문가 정책으로 에피소드 녹화

**수집 데이터**:
- 관측값 (84×84×4 프레임)
- 액션 (0-3 정수)
- 보상 (벽돌 파괴 시 +1)
- 종료 플래그

**저장 형식**: Apache Parquet
- 압축 효율적
- 병렬 읽기 지원
- 스키마 검증

**수집량**:
- 500 에피소드
- 약 10GB 데이터
- 평균 에피소드 길이: 500-1000 스텝

---

### Stage 3: 오프라인 Behavior Cloning

**목적**: 수집된 데이터로 지도학습 방식의 정책 학습

**알고리즘**: Behavior Cloning (BC)
- 지도학습 기반
- Cross-Entropy Loss
- Dropout 정규화

**모델 구조** (BCCNN):
```
Input (84×84×4)
    ↓
Conv2d(4→32, 8×8, stride=4) + ReLU
    ↓
Conv2d(32→64, 4×4, stride=2) + ReLU
    ↓
Conv2d(64→64, 3×3, stride=1) + ReLU
    ↓
Flatten → Linear(3136→512) + ReLU + Dropout(0.3)
    ↓
Linear(512→4)
    ↓
Output (4 actions)
```

**하이퍼파라미터**:
```python
learning_rate: 1e-4
batch_size: 64
epochs: 100
dropout: 0.3
optimizer: Adam
scheduler: ReduceLROnPlateau
```

**성능 목표**: 전문가의 60-70% 성능 (180-210점)

---

## 성능 결과

### 달성 성능

| 단계 | 알고리즘 | 평균 점수 | 최고 점수 | 학습 시간 |
|------|----------|----------|----------|----------|
| 기준선 | Random | 1.2 | 5 | - |
| Stage 1 | DQN | 287.4 | 412.3 | 9.2시간 |
| Stage 3 | BC | 181.6 | 267.8 | 1.8시간 |

### 효율성 비교

| 메트릭 | 온라인 (DQN) | 오프라인 (BC) | 비율 |
|--------|-------------|--------------|------|
| 학습 시간 | 9.2시간 | 1.8시간 | **5.1x 빠름** |
| 환경 상호작용 | 10M steps | 0 | **100% 감소** |
| GPU 사용량 | 높음 | 낮음 | - |

---

## 실행 방법

### 로컬 실행 (Ubuntu GPU 서버)

```bash
cd atari/sagemaker

# 의존성 설치
pip install -r requirements.txt

# Stage 1: 전문가 학습 (2-4시간, GPU 권장)
python train_expert.py --num-gpus 1 --target-reward 300

# Stage 2: 데이터 수집 (30분-1시간)
python record_data.py \
    --checkpoint-path ~/ray_results/atari_dqn_expert/model/best_checkpoint \
    --num-episodes 500

# Stage 3: 오프라인 학습 (30분-1시간)
python train_offline.py --epochs 100

# TensorBoard 모니터링
tensorboard --logdir ~/ray_results/
```

### AWS SageMaker 실행

```bash
# 전체 파이프라인 실행
python launch_sagemaker.py \
    --stage all \
    --bucket your-s3-bucket \
    --instance-type ml.g4dn.xlarge \
    --wait

# 개별 단계 실행
python launch_sagemaker.py --stage 1 --bucket your-bucket  # 전문가 학습
python launch_sagemaker.py --stage 2 --bucket your-bucket  # 데이터 수집
python launch_sagemaker.py --stage 3 --bucket your-bucket  # BC 학습
```

---

## 기술 스택

### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| Ray RLlib | 2.50.1 | 분산 RL 프레임워크 |
| PyTorch | 2.0+ | 딥러닝 백엔드 |
| Gymnasium | 1.2.1+ | RL 환경 인터페이스 |
| ALE-py | 0.9+ | Atari 에뮬레이터 |
| Pandas | 2.0+ | 데이터 처리 |
| TensorBoard | - | 학습 시각화 |

### 인프라

| 환경 | 용도 | 권장 사양 |
|------|------|----------|
| 로컬 | 개발/테스트 | GPU 1개, 16GB RAM |
| SageMaker | 프로덕션 학습 | ml.g4dn.xlarge |

---

## 주요 파일 설명

### 학습 스크립트

| 파일 | 설명 |
|------|------|
| `train_expert.py` | DQN 전문가 정책 학습, TensorBoard 로깅 |
| `record_data.py` | 전문가 에피소드 녹화, Parquet 저장 |
| `train_offline.py` | BC 오프라인 학습, 평가 포함 |

### 설정 및 유틸리티

| 파일 | 설명 |
|------|------|
| `configs/dqn_breakout_config.py` | DQN 하이퍼파라미터 설정 |
| `utils/preprocess.py` | 프레임 스택, 그레이스케일 변환 |
| `utils/cnn_models.py` | AtariCNN, DuelingCNN, BCCNN 정의 |

### 인프라

| 파일 | 설명 |
|------|------|
| `launch_sagemaker.py` | SageMaker 작업 실행 |
| `visualize_tensorboard.py` | S3 로그 다운로드 및 시각화 |
| `Dockerfile` | 커스텀 컨테이너 정의 |

---

## TensorBoard 메트릭

### Stage 1 (DQN)
- `Training/Episode_Reward_Mean`: 학습 보상
- `Evaluation/Episode_Reward_Mean`: 평가 보상
- `Training/Timesteps`: 총 스텝 수
- `Training/Epsilon`: 탐색률
- `Training/TD_Error`: TD 오류

### Stage 3 (BC)
- `Training/Loss`: 크로스엔트로피 손실
- `Training/Learning_Rate`: 학습률
- `Evaluation/Mean_Reward`: 평가 보상
- `Evaluation/Max_Reward`: 최고 보상

---

## 확장 가능성

### 알고리즘 확장

| 알고리즘 | 예상 효과 | 난이도 |
|---------|----------|--------|
| Dueling DQN | +20-40점 | 중 |
| Prioritized Replay | +15-25점 | 중 |
| CQL (Conservative Q-Learning) | +40-60점 | 상 |
| IQL (Implicit Q-Learning) | +35-55점 | 상 |

### 환경 확장

- 다른 Atari 게임 (Pong, Space Invaders)
- MuJoCo 로봇 제어
- 실제 로봇 데이터 적용

---

## 참고 자료

### 논문
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (DQN)
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives](https://arxiv.org/abs/2005.01643)

### 문서
- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [AWS SageMaker PyTorch](https://sagemaker.readthedocs.io/)

---

## 문서 목록

| 문서 | 설명 |
|------|------|
| [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) | 프로젝트 개요 (이 문서) |
| [PHASE_1.md](./PHASE_1.md) | Stage 1 상세 설명 (초보자용) |
| [PHASE_2.md](./PHASE_2.md) | Stage 2, 3 상세 설명 |
| [PERFORMANCE_RECORD.md](./PERFORMANCE_RECORD.md) | 성능 추적 기록 |
| [EXPERIMENT_1_DUELING_DQN.md](./EXPERIMENT_1_DUELING_DQN.md) | Dueling DQN 실험 |

---

**마지막 업데이트**: 2025년 12월 2일
**작성**: Claude Code Assistant
