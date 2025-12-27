"""
시뮬레이션 설정 관리 모듈
"""
from dataclasses import dataclass
from typing import Optional
import platform


@dataclass
class SimConfig:
    """시뮬레이션 설정을 관리하는 데이터클래스"""
    
    # 시뮬레이션 파라미터
    dt: float = 1.0 / 60.0  # 타임스텝 (초)
    substeps: int = 1       # 서브스텝 수
    gravity: float = -9.81  # 중력 가속도
    
    # 윈도우 설정
    window_width: int = 1024
    window_height: int = 768
    vsync: bool = True
    
    # 시스템 설정
    arch: str = "auto"      # Taichi 아키텍처 {cpu, metal, vulkan, cuda, auto}
    seed: Optional[int] = None  # 랜덤 시드
    
    def __post_init__(self):
        """설정 검증 및 자동 설정"""
        self._validate_config()
        if self.arch == "auto":
            self.arch = self._get_default_arch()
    
    def _validate_config(self):
        """설정값 검증"""
        if self.dt <= 0:
            raise ValueError(f"dt는 양수여야 합니다: {self.dt}")
        
        if self.substeps < 1:
            raise ValueError(f"substeps는 1 이상이어야 합니다: {self.substeps}")
        
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError(f"윈도우 크기는 양수여야 합니다: {self.window_width}x{self.window_height}")
        
        valid_archs = {"cpu", "metal", "vulkan", "cuda", "auto"}
        if self.arch not in valid_archs:
            raise ValueError(f"지원하지 않는 아키텍처: {self.arch}. 지원 아키텍처: {valid_archs}")
    
    def _get_default_arch(self) -> str:
        """플랫폼에 따른 기본 아키텍처 반환"""
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            return "metal"
        elif system == "linux":
            return "vulkan"
        elif system == "windows":
            return "vulkan"
        else:
            return "cpu"
    
    @property
    def effective_dt(self) -> float:
        """서브스텝을 고려한 실제 타임스텝"""
        return self.dt / self.substeps
    
    @property
    def target_fps(self) -> float:
        """목표 FPS"""
        return 1.0 / self.dt
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "dt": self.dt,
            "substeps": self.substeps,
            "gravity": self.gravity,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "vsync": self.vsync,
            "arch": self.arch,
            "seed": self.seed,
        }
    
    def __str__(self) -> str:
        """문자열 표현"""
        return (f"SimConfig(dt={self.dt:.4f}, substeps={self.substeps}, "
                f"arch={self.arch}, window={self.window_width}x{self.window_height})")