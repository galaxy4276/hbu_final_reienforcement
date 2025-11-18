#!/bin/bash

# 강화학습 Anaconda 환경 설정 스크립트
# Author: Claude Code
# Date: $(date +"%Y-%m-%d")

set -e  # 에러 발생 시 종료

echo "🚀 강화학습 Anaconda 환경 설정 시작..."

# 환경 설정 변수
ENV_NAME="rl_offline"
PYTHON_VERSION="3.12"
RAY_VERSION="2.50.1"
GYMNASIUM_VERSION="1.2.1"
MUJOCO_VERSION="3.3.7"

# conda 경로 확인 및 설정
CONDA_PATH="/home/user/anaconda3"
if [ ! -d "$CONDA_PATH" ]; then
    echo "❌ Anaconda가 $CONDA_PATH에 설치되어 있지 않습니다."
    echo "Anaconda 설치 경로를 확인해주세요."
    exit 1
fi

# 1. 기존 환경 삭제 (존재하는 경우)
echo "📦 기존 환경 삭제 중..."
$CONDA_PATH/bin/conda remove -n $ENV_NAME --all -y || true

# 2. 새 환경 생성
echo "🐍 Python $PYTHON_VERSION 환경 생성 중..."
$CONDA_PATH/bin/conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 3. Ray RLlib 설치
echo "☀️ Ray RLlib $RAY_VERSION 설치 중..."
$CONDA_PATH/envs/$ENV_NAME/bin/pip install "ray[rllib]==$RAY_VERSION"

# 4. PyTorch 설치
echo "🔥 PyTorch 설치 중..."
$CONDA_PATH/envs/$ENV_NAME/bin/pip install torch

# 5. Gymnasium 설치 (요구 버전 정확히 설치)
echo "🏋️ Gymnasium $GYMNASIUM_VERSION 설치 중..."
$CONDA_PATH/envs/$ENV_NAME/bin/pip uninstall gymnasium -y || true
$CONDA_PATH/envs/$ENV_NAME/bin/pip install "gymnasium[mujoco]==$GYMNASIUM_VERSION"

# 6. 추가 유틸리티 설치
echo "🛠️ 추가 유틸리티 설치 중..."
$CONDA_PATH/envs/$ENV_NAME/bin/pip install matplotlib seaborn jupyter

# 7. 설치 확인
echo "✅ 설치 버전 확인 중..."
$CONDA_PATH/envs/$ENV_NAME/bin/python -c "
import ray
import torch
import gymnasium
import mujoco
print('✅ 설치 완료된 버전:')
print(f'   Ray: {ray.__version__}')
print(f'   PyTorch: {torch.__version__}')
print(f'   Gymnasium: {gymnasium.__version__}')
print(f'   MuJoCo: {mujoco.__version__}')
"

# 8. 활성화 안내
echo ""
echo "🎉 환경 설정 완료!"
echo ""
echo "📋 환경 정보:"
echo "   - 환경명: $ENV_NAME"
echo "   - Python: $PYTHON_VERSION"
echo "   - Ray RLlib: $RAY_VERSION"
echo "   - Gymnasium: $GYMNASIUM_VERSION"
echo "   - MuJoCo: $MUJOCO_VERSION"
echo ""
echo "🔧 사용 방법:"
echo "   활성화: conda activate $ENV_NAME"
echo "   비활성화: conda deactivate"
echo ""
echo "🧪 테스트 명령어:"
echo "   python -c \"import ray; print(f'Ray: {ray.__version__}')\""
echo ""

echo "✅ 스크립트 실행 완료!"