"""
HUD (Head-Up Display) 렌더링 모듈
"""
import taichi as ti
from typing import Dict, Any, Optional, List, Tuple
from ..core.config import SimConfig
from ..core.time import FixedStepper
from ..core.stats import RuntimeStats


class HUD:
    """HUD 렌더링을 관리하는 클래스"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.font_size = 18
        self.line_height = 22
        self.margin = 10
        
        # 색상 정의
        self.colors = {
            "text": (0.9, 0.9, 0.9),
            "text_dim": (0.7, 0.7, 0.7),
            "good": (0.2, 0.8, 0.2),
            "warning": (0.8, 0.8, 0.2),
            "critical": (0.8, 0.2, 0.2),
            "paused": (0.8, 0.4, 0.2),
        }
        
        # 표시 옵션
        self.show_system_info = True
        self.show_simulation_info = True
        self.show_camera_info = False  # 기본적으로 숨김
        self.show_controls = True
        self.show_performance = True
    
    def render(self, window, stepper: FixedStepper, stats: RuntimeStats, 
               camera_info: Optional[Dict] = None):
        """HUD를 렌더링합니다"""
        if not window:
            return
        
        # ti.ui.Window에서 GUI 사용
        gui = window.get_gui()
        
        # HUD 윈도우 시작
        gui.begin("HUD", 0.02, 0.02, 0.3, 0.4)
        
        # 시스템 정보
        if self.show_system_info:
            self._render_system_info_gui(gui)
        
        # 시뮬레이션 정보
        if self.show_simulation_info:
            self._render_simulation_info_gui(gui, stepper)
        
        # 성능 정보
        if self.show_performance:
            self._render_performance_info_gui(gui, stats)
        
        # 카메라 정보 (선택적)
        if self.show_camera_info and camera_info:
            self._render_camera_info_gui(gui, camera_info)
        
        gui.end()
        
        # 컨트롤 정보 (별도 윈도우)
        if self.show_controls:
            self._render_controls_gui(gui)
    
    def _render_system_info_gui(self, gui):
        """시스템 정보를 GUI로 렌더링합니다"""
        gui.text(f"Architecture: {self.config.arch}")
        gui.text(f"Resolution: {self.config.window_width}x{self.config.window_height}")
        gui.text("")  # 빈 줄
    
    def _render_simulation_info_gui(self, gui, stepper: FixedStepper):
        """시뮬레이션 정보를 GUI로 렌더링합니다"""
        status = stepper.get_status()
        status_text = "PAUSED" if status["paused"] else "RUNNING"
        
        gui.text(f"Status: {status_text}")
        gui.text(f"Frame: {status['frame_idx']}")
        gui.text(f"Time: {status['sim_time']:.3f}s")
        gui.text(f"dt: {status['dt']:.4f}s ({status['substeps']} substeps)")
        gui.text("")  # 빈 줄
    
    def _render_performance_info_gui(self, gui, stats: RuntimeStats):
        """성능 정보를 GUI로 렌더링합니다"""
        current_fps = stats.get_fps()
        avg_fps = stats.get_average_fps()
        frame_time = stats.get_frame_time_ms()
        
        gui.text(f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})")
        gui.text(f"Frame Time: {frame_time:.2f}ms")
        gui.text(f"Total Frames: {stats.total_frames}")
        gui.text("")  # 빈 줄
    
    def _render_camera_info_gui(self, gui, camera_info: Dict):
        """카메라 정보를 GUI로 렌더링합니다"""
        pos = camera_info.get("position", [0, 0, 0])
        target = camera_info.get("target", [0, 0, 0])
        
        gui.text("Camera:")
        gui.text(f"  Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        gui.text(f"  Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        gui.text(f"  Distance: {camera_info.get('distance', 0):.2f}")
        gui.text(f"  Yaw: {camera_info.get('yaw', 0):.1f}°")
        gui.text(f"  Pitch: {camera_info.get('pitch', 0):.1f}°")
        gui.text("")  # 빈 줄
    
    def _render_controls_gui(self, gui):
        """컨트롤 정보를 GUI로 렌더링합니다"""
        gui.begin("Controls", 0.02, 0.5, 0.3, 0.45)
        
        gui.text("Controls:")
        gui.text("  LMB drag: Orbit camera")
        gui.text("  RMB/MMB drag: Pan camera")
        gui.text("  Mouse wheel: Zoom")
        gui.text("  SPACE: Pause/Resume")
        gui.text("  N: Single step")
        gui.text("  R: Reset simulation")
        gui.text("  Q/ESC: Quit")
        
        gui.end()
    
    
    def _get_fps_color(self, fps: float) -> Tuple[float, float, float]:
        """FPS에 따른 색상을 반환합니다"""
        target_fps = 1.0 / self.config.dt
        
        if fps >= target_fps * 0.9:  # 90% 이상
            return self.colors["good"]
        elif fps >= target_fps * 0.6:  # 60% 이상
            return self.colors["warning"]
        else:
            return self.colors["critical"]
    
    def toggle_system_info(self):
        """시스템 정보 표시를 토글합니다"""
        self.show_system_info = not self.show_system_info
    
    def toggle_camera_info(self):
        """카메라 정보 표시를 토글합니다"""
        self.show_camera_info = not self.show_camera_info
        print(f"카메라 정보 표시: {'ON' if self.show_camera_info else 'OFF'}")
    
    def toggle_performance_info(self):
        """성능 정보 표시를 토글합니다"""
        self.show_performance = not self.show_performance
    
    def toggle_controls(self):
        """컨트롤 정보 표시를 토글합니다"""
        self.show_controls = not self.show_controls
    
    def get_status(self) -> Dict[str, bool]:
        """HUD 상태를 반환합니다"""
        return {
            "system_info": self.show_system_info,
            "simulation_info": self.show_simulation_info,
            "camera_info": self.show_camera_info,
            "controls": self.show_controls,
            "performance": self.show_performance,
        }