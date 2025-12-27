"""HUD (Heads-Up Display) rendering."""
import taichi as ti
from typing import Tuple
from ..core.config import SimConfig
from ..core.time import FixedStepper
from ..core.stats import RuntimeStats


class HUD:
    """Renders simulation HUD overlay."""
    
    def __init__(self, window: ti.ui.Window):
        """
        Initialize HUD.
        
        Args:
            window: Taichi UI window (needed for GUI access)
        """
        self.window = window
        self.gui = window.get_gui()
    
    def draw(
        self,
        config: SimConfig,
        stepper: FixedStepper,
        stats: RuntimeStats
    ) -> None:
        """
        Draw HUD overlay.
        
        Args:
            config: Simulation configuration
            stepper: Time stepper
            stats: Runtime statistics
        """
        # Begin GUI frame
        self.gui.begin("Simulation HUD", 0.05, 0.05, 0.3, 0.4)
        
        # Build HUD text
        self.gui.text("=== Simulation HUD ===")
        self.gui.text(f"Arch: {config.arch}")
        self.gui.text(f"dt: {config.dt:.4f}")
        self.gui.text(f"Substeps: {config.substeps}")
        self.gui.text(f"FPS: {stats.get_fps():.1f}")
        self.gui.text(f"Paused: {stepper.paused}")
        self.gui.text(f"Frame: {stepper.frame}")
        self.gui.text(f"Sim Time: {stepper.sim_time:.4f}")
        
        # End GUI frame
        self.gui.end()

