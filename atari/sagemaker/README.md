# Atari RL Training on AWS SageMaker

AWS SageMaker에서 Atari Breakout 강화학습 파이프라인을 실행하기 위한 가이드입니다.

## 디렉토리 구조

```
sagemaker/
├── train_expert.py      # Stage 1: DQN 전문가 학습
├── record_data.py       # Stage 2: 전문가 데이터 수집
├── train_offline.py     # Stage 3: 오프라인 BC 학습
├── launch_sagemaker.py  # SageMaker 작업 실행 스크립트
├── requirements.txt     # Python 의존성
├── Dockerfile          # 커스텀 컨테이너 (선택사항)
└── README.md           # 이 파일
```

## 사전 요구사항

### 1. AWS 설정

```bash
# AWS CLI 설치 및 설정
pip install awscli
aws configure
```

### 2. SageMaker IAM Role 생성

SageMaker 실행 역할에 다음 권한 필요:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (또는 특정 버킷 접근 권한)

### 3. S3 버킷 생성

```bash
aws s3 mb s3://your-atari-rl-bucket
```

### 4. 로컬 환경 설정

```bash
pip install sagemaker boto3
export SAGEMAKER_ROLE="arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE"
```

## 빠른 시작

### 전체 파이프라인 실행 (권장)

```bash
cd atari/sagemaker

python launch_sagemaker.py \
    --stage all \
    --bucket your-atari-rl-bucket \
    --instance-type ml.g4dn.xlarge \
    --wait
```

### 개별 단계 실행

#### Stage 1: DQN 전문가 학습

```bash
python launch_sagemaker.py \
    --stage 1 \
    --bucket your-atari-rl-bucket \
    --instance-type ml.g4dn.xlarge \
    --max-iterations 5000 \
    --target-reward 300 \
    --wait
```

#### Stage 2: 전문가 데이터 수집

```bash
python launch_sagemaker.py \
    --stage 2 \
    --bucket your-atari-rl-bucket \
    --checkpoint-s3-uri s3://your-bucket/atari-rl/stage1-expert/.../model.tar.gz \
    --num-episodes 500 \
    --wait
```

#### Stage 3: 오프라인 BC 학습

```bash
python launch_sagemaker.py \
    --stage 3 \
    --bucket your-atari-rl-bucket \
    --data-s3-uri s3://your-bucket/atari-rl/stage2-data/... \
    --epochs 100 \
    --batch-size 64 \
    --wait
```

## 인스턴스 유형 가이드

| 인스턴스 | GPU | 메모리 | 가격/시간 | 권장 용도 |
|---------|-----|--------|----------|----------|
| `ml.g4dn.xlarge` | T4 | 16GB | ~$0.7 | Stage 1, 2, 3 (기본) |
| `ml.g5.xlarge` | A10G | 24GB | ~$1.0 | 빠른 학습 |
| `ml.p3.2xlarge` | V100 | 16GB | ~$3.0 | 대규모 실험 |

**권장**: `ml.g4dn.xlarge` - Atari DQN에 충분하고 비용 효율적

## 예상 시간 및 비용

| 단계 | 예상 시간 | 예상 비용 (g4dn.xlarge) |
|------|----------|------------------------|
| Stage 1 (Expert) | 2-4시간 | $1.4 - $2.8 |
| Stage 2 (Record) | 30분-1시간 | $0.35 - $0.7 |
| Stage 3 (BC) | 30분-1시간 | $0.35 - $0.7 |
| **총합** | **3-6시간** | **$2.1 - $4.2** |

## Spot 인스턴스로 비용 절감

`launch_sagemaker.py`에 다음 옵션 추가 (70% 비용 절감):

```python
estimator = PyTorch(
    ...
    use_spot_instances=True,
    max_wait=86400,  # 24시간 대기
)
```

## 로컬 테스트

SageMaker 실행 전 로컬에서 테스트:

```bash
# GPU 환경에서
python train_expert.py --num-gpus 1 --max-iterations 100

# CPU 환경에서
python train_expert.py --num-gpus 0 --max-iterations 10
```

## 모니터링

### SageMaker 콘솔

1. [SageMaker 콘솔](https://console.aws.amazon.com/sagemaker/) 접속
2. Training Jobs 메뉴에서 진행 상황 확인
3. CloudWatch Logs에서 상세 로그 확인

### AWS CLI

```bash
# 학습 작업 상태 확인
aws sagemaker describe-training-job --training-job-name YOUR_JOB_NAME

# 로그 확인
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

## 결과 다운로드

```bash
# 모델 다운로드
aws s3 cp s3://your-bucket/atari-rl/stage1-expert/YOUR_JOB/output/model.tar.gz ./

# 압축 해제
tar -xzf model.tar.gz
```

## 트러블슈팅

### ROM 설치 오류

Atari ROM이 자동 설치되지 않을 경우:

```python
# train_expert.py 상단에 추가
import subprocess
subprocess.run(['pip', 'install', 'autorom[accept-rom-license]'])
subprocess.run(['AutoROM', '--accept-license'])
```

### 메모리 부족

`ml.g4dn.xlarge` 대신 `ml.g4dn.2xlarge` (32GB 메모리) 사용

### 학습이 너무 느림

- `num-workers` 증가 (기본 4 → 8)
- 더 강력한 인스턴스 사용 (`ml.g5.xlarge`)

## 파이프라인 확장

### 하이퍼파라미터 튜닝

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(1e-5, 1e-3),
}

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='reward',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=3,
)
```

### 분산 학습

여러 인스턴스에서 병렬 학습:

```python
estimator = PyTorch(
    ...
    instance_count=2,  # 다중 인스턴스
    distribution={'smdistributed': {'dataparallel': {'enabled': True}}}
)
```

## 참고 자료

- [SageMaker PyTorch SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)
- [Ray RLlib on SageMaker](https://docs.ray.io/en/latest/cluster/running-on-aws.html)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
