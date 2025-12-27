#!/bin/bash
# 가상환경 활성화 및 실행 스크립트

# 가상환경이 없으면 생성
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치 확인
if ! python -c "import taichi" 2>/dev/null; then
    echo "패키지 설치 중..."
    pip install -e . || pip install -r requirements.txt
fi

# 시뮬레이션 실행
python -m sim.main "$@"
