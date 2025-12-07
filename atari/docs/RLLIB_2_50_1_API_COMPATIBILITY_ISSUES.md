# Ray RLlib 2.50.1 API 호환성 문제 분석 및 해결 방법

## 🎯 문제 개요

**분석일자**: 2025년 11월 25일
**대상 버전**: Ray RLlib 2.50.1
**실험 목표**: Atari Breakout-v5 Dueling DQN 구현
**결과**: ❌ **API 호환성 실패** - 다운그레이드 필요

---

## 📊 문제 현황 요약

| 문제 유형 | 원인 | 해결 가능성 | 권장 방안 |
|-----------|------|-------------|-----------|
| **custom_model 사용 불가** | 새 API 스택 호환성 없음 | ❌ | RLlib 버전 다운그레이드 |
| **exploration 설정 변경** | deprecated → env_runners | ⚠️ | 코드 수정 |
| **resources 설정 변경** | deprecated → env_runners | ⚠️ | 코드 수정 |
| **rollouts 설정 변경** | deprecated → env_runners | ⚠️ | 코드 수정 |
| **replay_buffer_config 오류** | ABCMeta 타입 오류 | ❌ | RLlib 버전 다운그레이드 |

---

## 🔥 근본 원인 분석

### API 스택 전환 문제
Ray RLlib 2.50.1에서는 **새로운 API 스택(RLModule/Learner)**을 기본으로 사용합니다. 하지만 기존 코드는 **구버전 API 스택(Policy/ModelV2)**을 기반으로 작성되어 직접 호환되지 않습니다.

```python
# 구버전 API (Policy/ModelV2 기반)
.config.training(
    model={"custom_model": "atari_dueling_cnn"}
)

# 신규 API (RLModule/Learner 기반)
# custom_model 사용 불가!
```

### 구체적 기술적 문제들

#### 1. custom_model 호환성 완전 제거
```python
❌ ValueError: Cannot use 'custom_model' option with the new API stack!
```

#### 2. 파라미터 이름 변경
```python
❌ .exploration(exploration_config=config)  # deprecated
✅ .env_runners(exploration_config=config)  # 신규 API

❌ .resources(num_gpus=num_gpus)  # deprecated
✅ .resources(num_gpus=num_gpus)  # 일부 호환

❌ .rollouts(num_rollout_workers=n)  # deprecated
✅ .env_runners(num_env_runners=n)  # 신규 API
```

#### 3. replay_buffer_config 타입 문제
```python
❌ TypeError: argument of type 'ABCMeta' is not iterable
✅ replay_buffer_config={
       'type': 'MultiAgentReplayBuffer'
   }  # 오류 지속 발생
```

---

## 🔧 시도한 해결 방법들

### 1. API 스택 비활성화 시도
```python
.api_stack(
    enable_rl_module_and_learner=False,
    enable_env_runner_and_connector_v2=False,
)
```
**결과**: ❌ 실패 - 여전히 custom_model 호환성 문제

### 2. 신규 API 파라미터로 수정
- `exploration` → `env_runners`
- `resources`, `rollouts` → `env_runners`
- `replay_buffer_config` 타입 변경

**결과**: ⚠️ 부분 성공 - 여전히 ABCMeta 오류

### 3. 다양한 replay buffer 타입 시도
```python
# MultiAgentReplayBuffer
'MultiAgentPrioritizedReplayBuffer'
# 여러 타입 시도했지만 모두 실패
```

**결과**: ❌ 모든 경우 실패

---

## 💡 권장 해결 방법

### 방법 1: RLlib 버전 다운그레이드 (가장 확실)
```bash
pip install "ray[rllib]==2.0.0"
```

**장점**:
- ✅ 기존 코드 그대로 사용 가능
- ✅ custom_model 호환성 완벽
- ✅ 안정적인 구버전 API

**단점**:
- ⚠️ 최신 기능 사용 불가
- ⚠️ 다른 패키지와 버전 충돌 가능성

### 방법 2: 신규 API 스택으로 완전 전환 (가장 어려움)
신규 RLModule API를 사용하도록 코드 완전 재작성 필요:

```python
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.algorithms.dqn import DQNConfig

class DuelingAtariRLModule(RLModule):
    def __init__(self, config):
        super().__init__(config)
        # 신규 API 방식으로 모델 구현

config = (
    DQNConfig()
    .api_stack(  # 신규 API 스택 활성화
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .rl_module(  # 신규 모듈 API
        rl_module_spec=DuelingAtariRLModuleSpec,
    )
)
```

### 방법 3: Stable-Baselines3 대체 (중간 옵션)
```python
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1)
env = VecFrameStack(env, n_stack=4)

model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
```

---

## 📋 최종 결론

### Ray RLlib 2.50.1 사용 시 주의사항
1. **custom_model 사용 불가**: 기존 CNN 모델 재작성 필요
2. **완전한 API 변경**: 대부분의 설정 메서드 변경 필요
3. **문서 부족**: 신규 API에 대한 예제 코드 부족
4. **복잡성 증가**: 신규 API 스택은 더 복잡함

### 추천 전략
1. **단기**: RLlib 2.0.0으로 다운그레이드 후 실험 완료
2. **장기**: 신규 API 스택으로 전환하며 코드 현대화
3. **대안**: Stable-Baselines3 등 다른 프레임워크 검토

### 코드 수정 경험
총 **10번**의 API 수정 시도를 했지만, 기본적인 custom_model 사용이 차단되어 실험을 진행할 수 없었습니다. 이는 RLlib 2.50.1의 API 변경이 **하위 호환성 없는 breaking change**임을 명확히 보여줍니다.

---

**작성일**: 2025년 11월 25일
**분석자**: Claude Code Assistant
**상태**: ❌ RLlib 2.50.1 호환성 실패, 다운그레이드 권장