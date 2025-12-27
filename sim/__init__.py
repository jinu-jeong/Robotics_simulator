"""
Taichi 기반 실시간 3D 시뮬레이션 패키지
"""

__version__ = "0.1.0"
__author__ = "Robotics Simulator Team"
__description__ = "Taichi 기반 실시간 3D 로보틱스 시뮬레이션 프레임워크"

# 핵심 모듈 임포트
from .core.config import SimConfig
from .core.app import TaichiApp, init_taichi
from .core.time import FixedStepper
from .core.input import InputState
from .core.stats import RuntimeStats

# 렌더링 모듈 임포트
from .render.ti_viewer import TaichiViewer3D, Camera3D
from .render.hud import HUD
from .render.debug_draw import DebugDraw

__all__ = [
    # Core
    "SimConfig",
    "TaichiApp", 
    "init_taichi",
    "FixedStepper",
    "InputState",
    "RuntimeStats",
    
    # Rendering
    "TaichiViewer3D",
    "Camera3D", 
    "HUD",
    "DebugDraw",
]