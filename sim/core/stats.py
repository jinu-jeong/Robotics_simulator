"""Runtime statistics tracking."""
import time
from collections import deque
from typing import Deque


class RuntimeStats:
    """Tracks FPS and other runtime statistics."""
    
    def __init__(self, window_size: int = 60):
        """
        Initialize stats tracker.
        
        Args:
            window_size: Number of frames to average for FPS calculation
        """
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.last_time: float = time.time()
        self.fps: float = 0.0
    
    def update(self) -> None:
        """Update statistics (call once per frame)."""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current frames per second
        """
        return self.fps

