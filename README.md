# Robotics Simulation Engine

Portable robotics simulation engine (rigid + FEM + contact) in Python using Taichi.

## Target Platform

- Primary: macOS (Apple Silicon)
- Portable to: CPU / Metal / Vulkan backends

## Requirements

- Python 3.11+
- Taichi >= 1.7.0
- Conda (Miniconda 또는 Anaconda)

## Installation

### Conda 환경 생성 및 설정

```bash
# 1. 콘다 환경 생성 (Python 3.11)
conda create -n robotics-sim python=3.11

# 2. 환경 활성화
conda activate robotics-sim

# 3. 프로젝트 디렉토리로 이동
cd /Users/jinu/workspace/robotics1

# 4. 패키지 설치
pip install -e .
```

### 환경 비활성화

```bash
conda deactivate
```

## Running the Simulation

### 기본 실행 방법

```bash
# 1. 콘다 환경 활성화 (아직 활성화하지 않았다면)
conda activate robotics-sim

# 2. 프로젝트 디렉토리로 이동
cd /Users/jinu/workspace/robotics1

# 3. 시뮬레이션 실행
python -m sim.main [options]
```

### 실행 옵션

- `--arch`: Taichi backend architecture (cpu, metal, vulkan, cuda). Default: metal
- `--dt`: Fixed timestep. Default: 0.01
- `--substeps`: Number of substeps per frame. Default: 1

### 실행 예제

```bash
# 기본 설정으로 실행 (Metal 백엔드)
python -m sim.main

# CPU 백엔드로 실행
python -m sim.main --arch cpu

# 커스텀 timestep과 substeps로 실행
python -m sim.main --dt 0.005 --substeps 2

# 모든 옵션 지정
python -m sim.main --arch metal --dt 0.01 --substeps 1
```

### 문제 해결

만약 Metal 백엔드 초기화에 실패하면 자동으로 CPU 백엔드로 폴백됩니다.

## Controls

- **SPACE**: Pause/Resume simulation
- **N**: Single step (advance one frame)
- **R**: Reset simulation
- **Q** or **ESC**: Quit

## HUD Display

The simulation window displays:
- Architecture (arch)
- Timestep (dt)
- Substeps
- FPS (frames per second)
- Pause status
- Frame number
- Simulation time

## Running Tests

```bash
pytest sim/tests/
```

## Project Structure

```
sim/
├── core/           # Core simulation modules
│   ├── config.py   # Configuration
│   ├── app.py      # Taichi initialization
│   ├── time.py     # Time stepping
│   ├── input.py    # Input handling
│   └── stats.py    # Runtime statistics
├── render/         # Rendering modules
│   ├── ti_viewer.py # Taichi UI wrapper
│   └── hud.py      # HUD rendering
├── tests/          # Tests
└── main.py         # Main entry point
```

## Development

This project follows these principles:
- Vendor-neutral (no NVIDIA-only dependencies)
- Modular and clean code structure
- Deterministic and reproducible behavior
- Fixed timestep with substeps
- Real-time visualization from day one

## License

[Add license information]

