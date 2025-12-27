"""Time stepping and simulation control."""
from dataclasses import dataclass
from typing import Callable, Optional
from .config import SimConfig


@dataclass
class FixedStepper:
    """Fixed timestep stepper with pause/resume/single-step/reset support."""
    
    config: SimConfig
    paused: bool = False
    frame: int = 0
    sim_time: float = 0.0
    
    def __post_init__(self):
        """Initialize stepper state."""
        self._step_callback: Optional[Callable[[float], None]] = None
    
    def set_step_callback(self, callback: Callable[[float], None]) -> None:
        """Set the callback function to call on each step."""
        self._step_callback = callback
    
    def step(self) -> None:
        """Perform a single simulation step."""
        if self.paused:
            return
        
        dt_sub = self.config.dt / self.config.substeps
        
        for _ in range(self.config.substeps):
            if self._step_callback:
                self._step_callback(dt_sub)
            self.sim_time += dt_sub
        
        self.frame += 1
    
    def single_step(self) -> None:
        """Perform a single step even if paused."""
        was_paused = self.paused
        self.paused = False
        self.step()
        self.paused = was_paused
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.paused = True
    
    def resume(self) -> None:
        """Resume the simulation."""
        self.paused = False
    
    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
    
    def reset(self) -> None:
        """Reset simulation state."""
        self.frame = 0
        self.sim_time = 0.0
        self.paused = False

