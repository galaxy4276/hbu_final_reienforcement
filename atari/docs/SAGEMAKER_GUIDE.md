# AWS SageMaker 배포 가이드

## 개요

이 문서는 Atari RL 파이프라인을 AWS SageMaker에서 실행하기 위한 상세 가이드입니다.

---

## 사전 요구사항

### AWS 설정

1. **AWS 계정 및 CLI**
```bash
pip install awscli boto3 sagemaker
aws configure
```

2. **IAM Role 생성**
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`

3. **S3 버킷**
```bash
aws s3 mb s3://your-atari-rl-bucket
```

4. **환경 변수**
```bash
export SAGEMAKER_ROLE="arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE"
```

---

## 디렉토리 구조

```
atari/sagemaker/
├── train_expert.py          # Stage 1: DQN 학습
├── record_data.py           # Stage 2: 데이터 수집
├── train_offline.py         # Stage 3: BC 학습
├── launch_sagemaker.py      # 실행 스크립트
├── visualize_tensorboard.py # 시각화
├── configs/
│   ├── __init__.py
│   └── dqn_breakout_config.py
├── utils/
│   ├── __init__.py
│   ├── preprocess.py
│   └── cnn_models.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 실행 방법

### 전체 파이프라인

```bash
cd atari/sagemaker

python launch_sagemaker.py \
    --stage all \
    --bucket your-bucket \
    --instance-type ml.g4dn.xlarge \
    --wait
```

### 개별 단계 실행

**Stage 1: 전문가 학습**
```bash
python launch_sagemaker.py \
    --stage 1 \
    --bucket your-bucket \
    --instance-type ml.g4dn.xlarge \
    --max-iterations 5000 \
    --target-reward 300 \
    --wait
```

**Stage 2: 데이터 수집**
```bash
python launch_sagemaker.py \
    --stage 2 \
    --bucket your-bucket \
    --checkpoint-s3-uri s3://your-bucket/atari-rl/stage1-expert/.../model.tar.gz \
    --num-episodes 500 \
    --wait
```

**Stage 3: BC 학습**
```bash
python launch_sagemaker.py \
    --stage 3 \
    --bucket your-bucket \
    --data-s3-uri s3://your-bucket/atari-rl/stage2-data/... \
    --epochs 100 \
    --batch-size 64 \
    --wait
```

---

## 인스턴스 유형

| 인스턴스 | GPU | 메모리 | 가격/시간 | 권장 용도 |
|---------|-----|--------|----------|----------|
| `ml.g4dn.xlarge` | T4 (16GB) | 16GB | ~$0.7 | 기본 (권장) |
| `ml.g5.xlarge` | A10G (24GB) | 24GB | ~$1.0 | 빠른 학습 |
| `ml.p3.2xlarge` | V100 (16GB) | 61GB | ~$3.0 | 대규모 실험 |

### Spot 인스턴스 (비용 70% 절감)

```python
# launch_sagemaker.py 수정
estimator = PyTorch(
    ...
    use_spot_instances=True,
    max_wait=86400,  # 24시간 대기
)
```

---

## 비용 예상

| 단계 | 예상 시간 | 비용 (g4dn.xlarge) |
|------|----------|-------------------|
| Stage 1 | 2-4시간 | $1.4 - $2.8 |
| Stage 2 | 30분-1시간 | $0.35 - $0.7 |
| Stage 3 | 30분-1시간 | $0.35 - $0.7 |
| **총합** | **3-6시간** | **$2.1 - $4.2** |

---

## 출력 경로

### S3 구조
```
s3://your-bucket/atari-rl/
├── stage1-expert/
│   └── {job-name}/
│       ├── model.tar.gz          # 모델 체크포인트
│       └── output/
│           ├── tensorboard/      # TensorBoard 로그
│           ├── training_history.json
│           └── training_summary.json
├── stage2-data/
│   └── {job-name}/
│       └── output/
│           ├── expert_data/      # Parquet 파일들
│           └── recording_summary.json
└── stage3-bc/
    └── {job-name}/
        ├── model.tar.gz          # BC 모델
        └── output/
            ├── tensorboard/
            ├── training_history.json
            └── training_summary.json
```

### 로컬 경로 (로컬 실행 시)
```
~/ray_results/
├── atari_dqn_expert/
│   ├── model/
│   ├── checkpoints/
│   ├── output/
│   └── tensorboard/
├── atari_record_data/
│   └── expert_data/
└── atari_bc_offline/
    ├── model/
    ├── output/
    └── tensorboard/
```

---

## TensorBoard 시각화

### S3에서 다운로드 후 시각화

```bash
python visualize_tensorboard.py \
    --s3-uri s3://your-bucket/atari-rl/stage1-expert/output/ \
    --launch
```

### 로컬 로그 시각화

```bash
python visualize_tensorboard.py \
    --local-dir ~/ray_results/atari_dqn_expert/tensorboard \
    --launch
```

### JSON에서 정적 그래프 생성

```bash
python visualize_tensorboard.py \
    --json-file training_history.json \
    --output-dir ./plots
```

---

## 모니터링

### SageMaker 콘솔

1. [SageMaker 콘솔](https://console.aws.amazon.com/sagemaker/) 접속
2. Training Jobs 메뉴
3. 진행 상황 및 로그 확인

### AWS CLI

```bash
# 작업 상태 확인
aws sagemaker describe-training-job --training-job-name YOUR_JOB_NAME

# 로그 확인
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### 결과 다운로드

```bash
# 모델 다운로드
aws s3 cp s3://your-bucket/atari-rl/stage1-expert/YOUR_JOB/output/model.tar.gz ./

# 압축 해제
tar -xzf model.tar.gz
```

---

## 트러블슈팅

### ROM 설치 오류

**증상**: `Namespace ALE not found`

**해결**:
```python
# 스크립트에 자동 설치 포함됨
import subprocess
subprocess.run(['pip', 'install', 'autorom[accept-rom-license]'])
subprocess.run(['AutoROM', '--accept-license'])
```

### 메모리 부족

**증상**: OOM 오류

**해결**:
- `ml.g4dn.2xlarge` (32GB) 사용
- `batch_size` 감소
- `num_workers` 감소

### 학습 느림

**해결**:
- 더 강력한 인스턴스 사용
- `num_workers` 증가

---

## 로컬 테스트

SageMaker 실행 전 로컬에서 테스트:

```bash
# GPU 환경
python train_expert.py --num-gpus 1 --max-iterations 100

# CPU 환경
python train_expert.py --num-gpus 0 --max-iterations 10
```

---

## 환경 변수 (SageMaker)

SageMaker 환경에서 자동 설정되는 변수:

| 변수 | 경로 | 용도 |
|------|------|------|
| `SM_MODEL_DIR` | `/opt/ml/model` | 모델 저장 |
| `SM_OUTPUT_DATA_DIR` | `/opt/ml/output/data` | 출력 데이터 |
| `SM_CHANNEL_TRAINING` | `/opt/ml/input/data/training` | 입력 데이터 |
| `SM_CHECKPOINT_DIR` | `/opt/ml/checkpoints` | 체크포인트 |

---

**마지막 업데이트**: 2025년 12월 2일
