"""Main simulation entry point."""
import argparse
import sys
from .core.config import SimConfig
from .core.app import init_taichi
from .core.time import FixedStepper
from .core.input import InputHandler
from .core.stats import RuntimeStats
from .render.ti_viewer import TiViewer
from .render.hud import HUD


def main():
    """Main simulation loop."""
    parser = argparse.ArgumentParser(description="Robotics Simulation")
    parser.add_argument(
        "--arch",
        type=str,
        default="metal",
        choices=["cpu", "metal", "vulkan", "cuda"],
        help="Taichi backend architecture"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Fixed timestep"
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="Number of substeps per frame"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimConfig(
        dt=args.dt,
        substeps=args.substeps,
        arch=args.arch
    )
    
    # Initialize Taichi (may fallback to cpu if metal fails)
    actual_arch = init_taichi(config.arch)
    config.arch = actual_arch  # Update config with actual arch used
    
    # Create components
    viewer = TiViewer(config)
    stepper = FixedStepper(config)
    stats = RuntimeStats()
    hud = HUD(viewer.get_window())
    
    # Setup input handler
    input_handler = InputHandler(viewer.get_window())
    input_handler.register_key("space", stepper.toggle_pause)
    input_handler.register_key("n", stepper.single_step)
    input_handler.register_key("r", stepper.reset)
    input_handler.register_key("q", lambda: sys.exit(0))
    input_handler.register_key("esc", lambda: sys.exit(0))
    
    # Simulation step callback (placeholder for now)
    def simulation_step(dt: float):
        """Placeholder simulation step."""
        # This will be implemented in later phases
        pass
    
    stepper.set_step_callback(simulation_step)
    
    # Main loop
    print("Simulation started. Controls: SPACE=pause, N=step, R=reset, Q/ESC=quit")
    
    while not viewer.should_close():
        # Poll input
        input_handler.poll()
        
        # Update simulation
        stepper.step()
        
        # Update stats
        stats.update()
        
        # Set canvas background color
        viewer.get_canvas().set_background_color((0.1, 0.1, 0.1))  # Dark gray background
        
        # Draw HUD
        hud.draw(config, stepper, stats)
        
        # Show window
        viewer.show()
    
    print("Simulation ended.")


if __name__ == "__main__":
    main()

