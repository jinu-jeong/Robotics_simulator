"""Simulation configuration."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SimConfig:
    """Simulation configuration parameters."""
    
    dt: float = 0.01  # Fixed timestep
    substeps: int = 1  # Number of substeps per frame
    gravity: float = -9.81  # Gravity acceleration
    window_size: Tuple[int, int] = (1024, 768)  # (width, height)
    arch: str = "metal"  # Taichi backend: cpu, metal, vulkan, cuda
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        if self.window_size[0] <= 0 or self.window_size[1] <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        valid_archs = {"cpu", "metal", "vulkan", "cuda"}
        if self.arch not in valid_archs:
            raise ValueError(f"arch must be one of {valid_archs}, got {self.arch}")

