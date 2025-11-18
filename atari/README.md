# Atari Breakout-v5 Offline Reinforcement Learning

이 프로젝트는 Atari ALE/Breakout-v5 환경에서 **오프라인 강화학습** 파이프라인을 구현합니다. 전문가 에이전트의 데이터를 수집하고, 이를 활용하여 새로운 에이전트를 모방 학습(behavior cloning)하는 과정을 포함합니다.

## 🎮 프로젝트 개요

**환경**: Atari ALE/Breakout-v5 (210x160x3 시각적 관측)
**알고리즘**: DQN (전문가 훈련) → BC (오프라인 학습)
**아키텍처**: CNN 기반 시각적 신경망
**프레임워크**: Ray RLlib + PyTorch

## 📁 프로젝트 구조

```
atari/
├── atari_1_Training_DQN_expert.py              # Stage 1: DQN 전문가 훈련
├── atari_2_Record_visual_data.py               # Stage 2: 전문가 데이터 수집
├── atari_3_Training_on_previously_saved_experiences.py  # Stage 3: 오프라인 BC 학습
├── utils/
│   ├── preprocess.py                           # 이미지 전처리 유틸리티
│   └── cnn_models.py                           # CNN 모델 정의
├── configs/
│   └── dqn_breakout_config.py                  # DQN 설정 관리
└── README.md                                   # 이 파일
```

## 🚀 실행 순서

### 사전 설정

```bash
# Conda 환경 활성화
conda activate rl_offline

# Atari ROM 경로 설정
export ALE_ROM_DIR=~/atari_roms
```

### Stage 1: 전문가 DQN 훈련

```bash
# CNN 모델로 DQN 전문가 훈련
python atari/atari_1_Training_DQN_expert.py \
    --experiment-name breakout_expert \
    --model-type cnn \
    --num-gpus 0

# 또는 Dueling DQN 사용
python atari/atari_1_Training_DQN_expert.py \
    --experiment-name breakout_dueling \
    --model-type dueling \
    --num-gpus 1
```

**목표**: 평가 점수 300+ (인간 수준)

**출력**: `~/ray_results/breakout_expert/`에 체크포인트 저장

**모니터링**:
```bash
tensorboard --logdir=~/ray_results/breakout_expert/
```

### Stage 2: 전문가 데이터 수집

```bash
# Stage 1에서 얻은 체크포인트 경로로 업데이트 필요!
# atari_2_Record_visual_data.py의 best_checkpoint 변수 수정

python atari/atari_2_Record_visual_data.py \
    --checkpoint-path ~/ray_results/breakout_expert/DQN_.../checkpoint_XXXX \
    --output-dir docs_rllib_offline_pretrain_ppo/ale-breakout-v5 \
    --num-episodes 500 \
    --num-workers 2
```

**목표**: 500개 전문가 에피소드 수집

**출력**: `docs_rllib_offline_pretrain_ppo/ale-breakout-v5/`에 Parquet 파일 저장

### Stage 3: 오프라인 모방 학습

```bash
python atari/atari_3_Training_on_previously_saved_experiences.py \
    --input-data-dir docs_rllib_offline_pretrain_ppo/ale-breakout-v5 \
    --experiment-name bc_breakout_offline \
    --num-iterations 200 \
    --num-gpus 0
```

**목표**: 전문가 성능의 50%+ 달성 (150+ 점수)

**출력**: `~/ray_results/bc_breakout_offline/`에 BC 체크포인트 저장

**모델 평가**:
```bash
python atari/atari_3_Training_on_previously_saved_experiences.py \
    --evaluate-only \
    --checkpoint-path ~/ray_results/bc_breakout_offline/BC_.../checkpoint_XXXX \
    --num-eval-episodes 50
```

## 🧠 핵심 컴포넌트

### 이미지 전처리 (`preprocess.py`)

- **그레이스케일 변환**: (210, 160, 3) → (210, 160)
- **리사이징**: (210, 160) → (84, 84)
- **정규화**: 픽셀값 [0, 255] → [0, 1]
- **프레임 스태킹**: 4 프레임 → (84, 84, 4)

### CNN 아키텍처 (`cnn_models.py`)

- **AtariCNN**: DeepMind Nature paper 기반
  - Conv2d(32, 8x8) + ReLU
  - Conv2d(64, 4x4) + ReLU
  - Conv2d(64, 3x3) + ReLU
  - FC(512) + ReLU
  - FC(4) 출력

- **DuelingAtariCNN**: Value-Advantage 분리
- **BCCNN**: Behavior Cloning专用 (Dropout 추가)

### DQN 설정 (`dqn_breakout_config.py`)

- **학습률**: 2.5e-4 (Atari 표준)
- **배치크기**: 32
- **리플레이 버퍼**: 1M 스텝
- **네트워크 업데이트**: 1000 스텝마다
- **탐색전략**: Epsilon-Greedy (1.0 → 0.01)

## 📊 예상 성능

| 단계 | 알고리즘 | 점수 목표 | 학습 시간 |
|------|----------|----------|----------|
| 랜덤 | Random Agent | 1-2점 | - |
| 전문가 | DQN | 300-400점 | 8-12시간 |
| BC | Behavior Cloning | 150-200점 | 1-2시간 |

## 🔧 트러블슈팅

### 환경 오류
```
Namespace ALE not found
```
**해결**: Atari ROMs 설치 확인
```bash
export ALE_ROM_DIR=~/atari_roms
autorom --install-dir ~/atari_roms
```

### 메모리 부족
**증상**: `MemoryError` 또는 시스템 느려짐
**해결**:
- 배치크기 감소 (`--episodes-per-worker 20`)
- 파일크기 감소 (`--max-file-size 3`)

### GPU 오류
**증상**: CUDA 오류
**해결**: `--num-gpus 0`으로 CPU 사용

### 학습 수렴 문제
**증상**: 점수가 오르지 않음
**해결**:
- 학습률 조정 (1e-5 ~ 1e-3)
- 전문가 데이터 품질 확인
- 더 많은 에피소드 수집

## 📈 성능 최적화

### 전문가 데이터 개선
1. **더 높은 성능**: DQN 훈련을 400+ 점수까지
2. **다양성**: 다른 초기 조건에서 데이터 수집
3. **노이즈 제거**: 안정적인 정책만 사용

### BC 모델 개선
1. **더 복잡한 모델**: ResNet, EfficientNet 등
2. **데이터 증강**: 이미지 회전, 노이즈 추가
3. **정규화**: Dropout, Weight Decay

## 🧪 실험 아이디어

### 알고리즘 비교
- DQN vs PPO vs A3C 전문가
- BC vs DAgger vs VIB 오프라인 학습

### 아키텍처 실험
- Different CNN backbones
- LSTM/Transformer for temporal modeling
- Multi-task learning (value + policy)

### 데이터 효율성
- Minimum required expert episodes
- Active learning for data selection
- Curriculum learning approaches

## 📚 참고 자료

### 논문
- **Human-level control through deep reinforcement learning** (Mnih et al., 2015)
- **Dueling Network Architectures for Deep Reinforcement Learning** (Wang et al., 2016)
- **Deep Learning for Atari Games** (Nair et al., 2015)

### 라이브러리
- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [Gymnasium Atari](https://gymnasium.farama.org/environments/atari/)
- [OpenCV Documentation](https://opencv.org/)

### 튜토리얼
- [DeepMind RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-2020)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

## 📝 개발 노트

### v1.0 (2025-11-18)
- ✅ 3단계 파이프라인 완성
- ✅ CNN 기반 시각적 처리
- ✅ 오프라인 데이터 수집 및 학습
- ✅ 완전 자동화된 워크플로우

### 다음 버전 계획
- 🔄 더 복잡한 Atari 게임 (Montezuma's Revenge)
- 🔄 Multi-environment 학습
- 🔄 실제 로봇 제어로 확장
- 🔄 Minari 데이터셋 통합

---

**연락처**: Claude Code Assistant
**최종 업데이트**: 2025년 11월 18일