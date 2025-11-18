# 강화학습 기말 프로젝트: 오프라인 강화학습 기반 에이전트 학습 및 파인튜닝

## 프로젝트 개요

본 프로젝트는 Ray RLlib을 활용하여 오프라인 강화학습 기반 에이전트를 학습한 후, 온라인 강화학습으로 파인튜닝하는 완전한 파이프라인을 구현합니다. 3단계 실습을 통해 오프라인 강화학습의 핵심 개념과 실제 적용 방법을 체계적으로 학습합니다.

## 목표

- ✅ **오프라인 강화학습 예제 따라하기**: RLlib을 활용한 기본 오프라인 RL 파이프라인 이해
- ✅ **MuJoCo HalfCheetah-v5 적용**: 복잡한 물리 시뮬레이션 환경에서의 오프라인 RL 구현
- ✅ **온라인 파인튜닝**: 오프라인으로 사전학습된 에이전트의 온라인 RL 기반 성능 향상
- ✅ **Minari 데이터셋 활용**: 전문가 데이터 변환 및 RLlib 호환 포맷 적용
- ✅ **Atari Breakout 구현**: 고차원 시각 환경에서의 오프라인 RL 성능 검증

## 기술 스택

- **강화학습 프레임워크**: Ray RLlib (v2.50.1)
- **환경**: Gymnasium 1.2.1
- **물리 시뮬레이션**: MuJoCo 3.3.7
- **딥러닝 백엔드**: PyTorch
- **데이터 처리**: Parquet, PyArrow

## 프로젝트 구조

```
final_reinforcement/
├── base/                           # 기준 구현 (오프라인 RL 기본 파이프라인)
│   ├── orl_1_Training_an_expert_policy.py
│   ├── orl_2_Record_expert_data_to_local_disk.py
│   └── orl_3_Training_on_previously_saved_experiences.py
├── mujoco/                         # MuJoCo HalfCheetah-v5 구현
├── minari/                         # Minari 데이터셋 변환 파이프라인
├── atari/                          # ALE/Breakout-v5 구현
├── docs/                           # 프로젝트 문서
│   └── 강화학습 기말 프로젝트.pdf
└── results/                        # 실험 결과 및 분석
```

## Base 디렉터리: 기준 구현

### 목적

`base/` 디렉터리는 오프라인 강화학습의 **표준 구현 패턴**을 제공합니다. CartPole-v1 환경을 통해 오프라인 RL의 3단계 파이프라인을 단계별로 구현하며, 각 환경별 구현(MuJoCo, Atari)의 **참조 코드**로 활용됩니다.

### 3단계 파이프라인

#### 1단계: 전문가 정책 학습 (`orl_1_Training_an_expert_policy.py`)
- **알고리즘**: PPO (Proximal Policy Optimization)
- **환경**: CartPole-v1
- **목표**: 평균 에피소드 리턴 450.0 달성
- **출력**: 전문가 정책 체크포인트 (`~/ray_results/docs_rllib_offline_pretrain_ppo/`)

#### 2단계: 전문가 데이터 수집 (`orl_2_Record_expert_data_to_local_disk.py`)
- **목적**: 학습된 전문가 정책으로 에피소드 데이터 기록
- **데이터 형식**: RLlib Episode 객체 → Parquet 파일
- **수집량**: 500 에피소드 (10회 반복 × 50 에피소드)
- **저장 위치**: `docs_rllib_offline_pretrain_ppo/cartpole-v1/`

#### 3단계: 오프라인 학습 (`orl_3_Training_on_previously_saved_experiences.py`)
- **알고리즘**: BC (Behavior Cloning)
- **데이터 소스**: 2단계에서 수집된 전문가 데이터
- **목표**: 수집된 데이터로부터 정책 복제 학습
- **평가**: 3회 학습마다 5 에피소드 평가

### 기술적 특징

- **체크포인트 기반**: 단계별 모델 상태 저장 및 복원
- **병렬 처리**: 데이터 수집 및 학습에서 다중 워커 활용
- **에피소드 기반**: 완전한 에피소드 단위 데이터 처리
- **실용적 접근**: 학습 가능한 간단한 환경으로 개념 검증

## 실행 방법

### 사전 요구사항

```bash
pip install "ray[rllib]" torch gymnasium
```

### 기준 파이프라인 실행

```bash
# 1단계: 전문가 정책 학습
python base/orl_1_Training_an_expert_policy.py

# 2단계: 체크포인트 경로 업데이트 후 데이터 수집
# base/orl_2_*.py 파일 내 best_checkpoint 변수 경로 수정 필요
python base/orl_2_Record_expert_data_to_local_disk.py

# 3단계: 오프라인 학습
python base/orl_3_Training_on_previously_saved_experiences.py
```

## 학습 모니터링

```bash
# TensorBoard로 학습 과정 시각화
tensorboard --logdir=~/ray_results/

# Ray 대시보드 확인
ray start --head
```

## 프로젝트 확장 가이드

### MuJoCo HalfCheetah-v5 구현
- `base/` 코드를 참조하여 CartPole → HalfCheetah 환경 변경
- 연속 행동 공간에 맞는 네트워크 아키텍처 수정
- 물리 시뮬레이션 특성을 고려한 보상 함수 최적화

### Minari 데이터셋 통합
- Atari/Breakout 전문가 데이터 다운로드
- Minari 포맷 → RLlib Episode 객체 변환
- 대용량 데이터 처리 파이프라인 구현

### ALE/Breakout-v5 구현
- 고차원 시각 입력 처리 (CNN 기반)
- 프레임 스킵 및 이미지 전처리
- Atari 환경 특화된 네트워크 구조

## 팀 개발 가이드

### 역할 분장
- **오프라인 RL 전문가**: MuJoCo 환경 및 고급 알고리즘 구현
- **데이터 엔지니어링**: Minari 데이터셋 변환 및 처리 파이프라인
- **Atari 환경 전문가**: Breakout 환경 및 시각적 RL 구현

### 개발 순서
1. **1주차**: 각자 환경별 기반 코드 구현 (병렬 개발)
2. **2주차**: 오프라인 → 온라인 파인튜닝 구현 및 최적화
3. **3주차**: 통합 테스트 및 성능 비교 분석

## 성공 기준

- **MuJoCo**: 오프라인 학습된 에이전트의 파인튜닝 후 성능 향상 확인
- **Minari**: 전문가 데이터 성공적 변환 및 오프라인 학습 동작
- **Atari**: 시각 환경에서의 오프라인 RL 유효성 검증
- **비교 분석**: 3환경 간 오프라인 RL 성능 특성 분석

## 참고 자료

- [Ray RLlib 공식 문서](https://docs.ray.io/en/latest/rllib/)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [오프라인 강화학습 튜토리얼](https://docs.ray.io/en/latest/rllib/rllib-offline.html)

---

이 프로젝트는 오프라인 강화학습의 이론적 개념과 실제 구현을 연결하며, 다양한 환경에서의 적용 가능성을 탐구하는 종합적인 학습 경험을 제공합니다.