"""Taichi UI viewer wrapper."""
import taichi as ti
from typing import Tuple
from ..core.config import SimConfig


class TiViewer:
    """Wrapper for Taichi UI window."""
    
    def __init__(self, config: SimConfig):
        """
        Initialize viewer.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.window = ti.ui.Window(
            "Robotics Simulation",
            res=config.window_size,
            vsync=True
        )
        self.canvas = self.window.get_canvas()
    
    def should_close(self) -> bool:
        """
        Check if window should close.
        
        Returns:
            True if window should close
        """
        return not self.window.running
    
    def get_window(self) -> ti.ui.Window:
        """
        Get the underlying Taichi window.
        
        Returns:
            Taichi UI window instance
        """
        return self.window
    
    def get_canvas(self) -> ti.ui.Canvas:
        """
        Get the canvas for drawing.
        
        Returns:
            Taichi UI canvas instance
        """
        return self.canvas
    
    def show(self) -> None:
        """Show the window (call at end of frame)."""
        self.window.show()

