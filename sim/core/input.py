"""Input polling and key handling."""
from typing import Dict, Callable, Optional
import taichi as ti


class InputHandler:
    """Handles keyboard input polling."""
    
    def __init__(self, window: ti.ui.Window):
        """
        Initialize input handler.
        
        Args:
            window: Taichi UI window instance
        """
        self.window = window
        self.key_handlers: Dict[str, Callable[[], None]] = {}
    
    def register_key(self, key: str, handler: Callable[[], None]) -> None:
        """
        Register a key handler.
        
        Args:
            key: Key name (e.g., 'space', 'n', 'r', 'q', 'esc')
            handler: Callback function to call when key is pressed
        """
        self.key_handlers[key] = handler
    
    def poll(self) -> None:
        """Poll for input and trigger registered handlers."""
        for key, handler in self.key_handlers.items():
            if self._is_key_pressed(key):
                handler()
    
    def _is_key_pressed(self, key: str) -> bool:
        """
        Check if a key is currently pressed.
        
        Args:
            key: Key name
            
        Returns:
            True if key is pressed
        """
        # Map key names to Taichi UI key strings
        key_map = {
            "space": " ",
            "esc": "ESCAPE",
            "n": "n",
            "r": "r",
            "q": "q",
        }
        
        if key not in key_map:
            return False
        
        return self.window.is_pressed(key_map[key])

