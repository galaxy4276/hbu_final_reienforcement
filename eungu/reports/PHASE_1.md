ALE/Breakout-v5 강화학습 실험 - ML 초보자 리뷰

## 📝 이 글을 읽어야 하는 사람

- 강화학습(Reinforcement Learning)이 무엇인지 막 배우기 시작한 분
- 컴퓨터가 게임을 어떻게 배우는지 궁금한 분
- 실제 프로젝트 코드를 보면서 개념을 이해하고 싶은 분
- 딥러닝 기초는 알지만 강화학습은 처음인 분

---

## 🎯 이 프로젝트의 목표

**컴퓨터가 Atari Breakout 게임을 전문가처럼 플레이하도록 가르치기**

![Atari Breakout 게임](https://upload.wikimedia.org/wikipedia/en/thumb/6/65/Breakout-arcade.png/300px-Breakout-arcade.png)

*Breakout: 공을 막대로 튕겨서 벽돌을 깨는 고전 게임*

### 3단계 학습 과정
1. **전문가 훈련**: 컴퓨터가 처음부터 게임을 배움 (DQN 알고리즘 사용)
2. **데이터 수집**: 전문가가 플레이하는 것을 모두 녹화
3. **모방 학습**: 녹화된 데이터를 보고 다른 컴퓨터가 따라 배움 (오프라인 학습)

---

## 🧠 강화학습 핵심 개념 (초보자용)

### 기본 용어 설명

| 용어 | 쉬운 설명 | 예시 (Breakout) |
|------|-----------|----------------|
| **에이전트(Agent)** | 학습하는 주체 | 게임을 하는 컴퓨터 |
| **환경(Environment)** | 에이전트가 활동하는 공간 | Breakout 게임 |
| **상태(State)** | 현재 상황을 나타내는 정보 | 게임 화면 이미지 |
| **액션(Action)** | 에이전트가 할 수 있는 행동 | 왼쪽 이동, 오른쪽 이동, 멈춤 |
| **보상(Reward)** | 행동에 대한 점수 | 벽돌 깨면 +1점, 공 떨어지면 -1점 |
| **정책(Policy)** | 어떤 상태에서 무엇을 할지 결정하는 규칙 | "공이 왼쪽에 있으면 왼쪽으로 이동" |

### 강화학습 원리 (사람으로 비유)

```
👶 아이가 자전거 타는 법 배우기:

1️⃣ 대충 페달을 밟아본다 (랜덤 액션)
   → 넘어진다 (보상: -1)

2️⃣ 다른 방식으로 시도해본다 (다른 액션)
   → 앞으로 1미터 간다 (보상: +1)

3️⃣ 성공했던 방법을 기억하고 비슷하게 해본다 (정책 업데이트)
   → 더 잘 타게 된다!

⭐ 이것이 강화학습의 핵심: **시행착오를 통해 스스로 학습**
```

---

## 🏗️ 우리가 만든 시스템 구조

### 전체 프로젝트 구조
```
final_reinforcement/
├── base/                # 기본 예제 (CartPole 게임)
├── mujoco/             # 로봇 워킹 게임
├── atari/              # 🎮 우리가 작업 중인 부분!
│   ├── atari_1_Training_DQN_expert.py    # 1단계: 전문가 훈련
│   ├── utils/
│   │   ├── preprocess.py                    # 이미지 전처리
│   │   └── cnn_models.py                    # AI 모델 구조
│   └── configs/
│       └── dqn_breakout_config.py          # 설정 값들
└── results/            # 결과 저장
```

### 1단계: 전문가 DQN 훈련 시스템

```
🎮 게임 화면 (210×160×3 픽셀)
        ↓
🖼️ 이미지 전처리 (그레이스케일 + 84×84 리사이즈 + 4프레임 스택)
        ↓
🧠 CNN 신경망 (딥러닝 모델)
        ↓
🎯 어떤 행동을 할지 결정 (왼쪽/오른쪽/멈춤 등)
        ↓
💰 점수(보상) 받고 다음 상태로 이동
        ↓
🔄 다시 학습... (이 과정을 1000만번 반복!)
```

---

## 💻 핵심 코드 설명 (초보자용)

### 1. 이미지 전처리 (`preprocess.py`)

컴퓨터가 게임 화면을 이해하기 쉽게 만드는 과정:

```python
def preprocess_observation(obs):
    # 🎨 컬러 이미지 → 흑백 이미지
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    # 🔍 큰 이미지 → 작은 이미지 (210×160 → 84×84)
    resized = cv2.resize(gray, (84, 84))

    # 📏 숫자 정규화 (0~255 → 0~1)
    normalized = resized / 255.0

    return normalized
```

**왜 이렇게 할까?**
- 컬러는 불필요한 정보가 많음 → 흑백으로 단순화
- 고해상도는 계산이 무거움 → 저해상도로 축소
- 작은 숫자(0~1)가 학습에 더 효율적

### 2. CNN 모델 (`cnn_models.py`)

딥러닝 모델이 게임 화면을 보고 판단하는 과정:

```python
class AtariCNN(nn.Module):
    def __init__(self):
        # 👁️ 컴퓨터 눈 (Convolutional Layers)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # 특징 추출
            nn.ReLU(),                                    # 활성화 함수
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 더 복잡한 특징
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 최종 특징
            nn.ReLU()
        )

        # 🧠 뇌 (Fully Connected Layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # 특징들을 조합
            nn.ReLU(),
            nn.Linear(512, 4)            # 최종 결정 (4가지 액션)
        )
```

**CNN 동작 방식 (인간 시각 비유):**
1. **첫 번째 레이어**: "공, 벽돌, 패드 같은 기본 도형 인식"
2. **두 번째 레이어**: "공과 패드의 위치 관계 파악"
3. **세 번째 레이어**: "공이 움직이는 방향 예측"
4. **최종 결정**: "공이 왼쪽으로 오니까 패드를 왼쪽으로 움직여야 해!"

### 3. DQN 학습 설정 (`dqn_breakout_config.py`)

어떻게 학습할지 설정하는 부분:

```python
def get_dqn_config():
    return DQNConfig()
        .training(
            lr=0.0001,           # 📚 학습 속도 (느리지만 안정적)
            train_batch_size=32,  # 📖 한 번에 보는 데이터 양
            gamma=0.99,          # 🔮 미래 보상 중요도 (미래를 중시)
            double_q=True,       # 🎯 더 똑똑한 학습 방식
        )
        .exploration(
            exploration_config={
                "initial_epsilon": 1.0,      # 🎲 처음엔 랜덤하게 탐험
                "final_epsilon": 0.01,       # 🎯 나중엔 최적의 선택만
                "epsilon_timesteps": 1000000, # ⏰ 100만 스텝 동안 서서히 변화
            }
        )
```

**Epsilon-Greedy 전략 (초보자 필수 개념):**
- **Epsilon = 1.0 (초기)**: 100% 랜덤 행동 → "모든 방법 시도해보기"
- **Epsilon = 0.5 (중간)**: 50% 랜덤, 50% 최적선택 → "경험과 탐험 균형"
- **Epsilon = 0.01 (최종)**: 1% 랜덤, 99% 최적선택 → "이제 최고만 하자!"

---

## 🚀 실제 실행 결과

### 환경 테스트 결과
```
✅ Successfully created ALE/Breakout-v5 environment!
Action space: Discrete(4)              # 4가지 행동 가능
Observation space: (210, 160, 3)       # 게임 화면 크기
Initial observation shape: (210, 160, 3)
✅ Environment test completed successfully!
```

### 예상 학습 과정
```
🔄 반복 1-100:   점수 0-5점 (컴퓨터가 게임 규칙 배우는 중)
🔄 반복 100-500: 점수 5-20점 (기본적인 플레이 가능)
🔄 반복 500-1000: 점수 20-50점 (어느 정도 실력)
🔄 반복 1000-5000: 점수 50-200점 (중급 플레이어)
🔄 반복 5000+:   점수 300점+ (인간 수준 전문가)
```

### 학습 속도
- **총 학습 시간**: 약 8-12시간 (GPU 미사용 시)
- **메모리 사용량**: 약 2-4GB
- **필요한 데이터**: 1000만 스텝 (약 10GB)

---

## 🔧 사용법 (실제 코드 실행)

### 1단계: 환경 설정
```bash
# 가상환경 활성화
conda activate rl_offline

# Atari ROM 경로 설정
export ALE_ROM_DIR=~/atari_roms
```

### 2단계: 전문가 훈련 실행
```bash
python atari/atari_1_Training_DQN_expert.py --experiment-name breakout_expert
```

### 3단계: 학습 모니터링
```bash
# 별도 터미널에서 실행
tensorboard --logdir=~/ray_results/breakout_expert/
```
웹 브라우저에서 `localhost:6006` 접속하면 학습 과정 그래프 볼 수 있음

---

## 📚 이 프로젝트를 통해 배울 수 있는 것

### 기술적 지식
1. **딥러닝**: CNN (Convolutional Neural Network) 구조 이해
2. **강화학습**: DQN (Deep Q-Network) 알고리즘 실습
3. **컴퓨터 비전**: 이미지 전처리 및 특징 추출
4. **MLOps**: Ray RLlib를 통한 분산 학습 관리

### 응용 가능 분야
- 🎮 게임 AI 개발
- 🤖 로봇 제어
- 📈 주식 투자 전략
- 🚗 자율주행
- 🎯 최적화 문제 해결

---

## ❓ 자주 묻는 질문 (FAQ)

**Q: 왜 Breakout 게임인가요?**
A: 규칙이 단순하지만 플레이하려면 시간차, 위치 예측 등 복잡한 능력이 필요해 RL의 좋은 벤치마크입니다.

**Q: CNN이 왜 필요한가요?**
A: 게임 화면(이미지)을 보고 판단해야 하기 때문입니다. 이미지에서 패턴을 찾는 데 CNN이 가장 효과적입니다.

**Q: 학습이 너무 오래 걸리는 것 같아요**
A: 정상입니다! 강화학습은 수백만번의 시행착오를 거치며 학습합니다. GPU를 사용하면 3-4배 빨라집니다.

**Q: 인간처럼 게임을 잘 할 수 있나요?**
A: 네! 충분한 학습 후에는 인간보다 더 잘 플레이합니다. 이미 Atari 게임에서 AI는 인간을 이겼습니다.

**Q: 다른 게임도 가능한가요?**
A: 네! 코드 조금만 수정하면 Pac-Man, Space Invaders 등 다른 Atari 게임도 가능합니다.

---

## 🎯 다음 단계

현재 1단계(전문가 훈련)가 완성되었습니다. 다음은:

2단계: **전문가 플레이 데이터 수집**
- 잘 훈련된 AI가 플레이하는 것을 녹화
- 녹화된 데이터를 파일로 저장

3단계: **오프라인 모방 학습**
- 녹화된 데이터로 새로운 AI 훈련
- 실시간 게임 없이 데이터만으로 학습

---

## 💡 조언 및 팁

**초보자를 위한 학습 팁:**
1. **작게 시작**: 먼저 CartPole 같은 간단한 환경으로 개념 익히기
2. **시각화**: TensorBoard로 학습 과정 꼭 확인하기
3. **파라미터 튜닝**: 학습률, 배치크기 등 바꿔보며 결과 비교
4. **코드 분석**: 다른 사람의 RL 프로젝트 GitHub에서 찾아보기

**추천 학습 자료:**
- 📚 책: "강화학습 입문" (한빛아카데미)
- 🎥 강의: David Silver RL Course (유튜브)
- 🛠️ 실습: OpenAI Gym Documentation
- 💻 커뮤니티: RL Reddit, RL Discord
