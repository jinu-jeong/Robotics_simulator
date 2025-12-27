"""
Taichi 3D 뷰어 및 카메라 컨트롤 모듈
"""
import taichi as ti
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
from ..core.config import SimConfig
from ..core.input import InputState, get_camera_control_input


class Camera3D:
    """3D 카메라 클래스"""
    
    def __init__(self, 
                 target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 distance: float = 5.0,
                 yaw: float = 0.0,
                 pitch: float = 30.0):
        """
        Args:
            target: 카메라가 바라보는 목표점
            distance: 목표점으로부터의 거리
            yaw: 수평 회전각 (도)
            pitch: 수직 회전각 (도)
        """
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        
        # 제한값
        self.min_distance = 0.1
        self.max_distance = 100.0
        self.min_pitch = -89.0
        self.max_pitch = 89.0
        
        # 감도 설정
        self.orbit_sensitivity = 100.0
        self.pan_sensitivity = 2.0
        self.zoom_sensitivity = 0.1
        
        # 계산된 위치와 방향
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self._update_position()
    
    def _update_position(self):
        """카메라 위치를 업데이트합니다"""
        # 구면 좌표계에서 직교 좌표계로 변환
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        # 카메라 위치 계산
        x = self.distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        y = self.distance * math.sin(pitch_rad)
        z = self.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        
        self.position = self.target + np.array([x, y, z], dtype=np.float32)
    
    def orbit(self, dx: float, dy: float):
        """궤도 회전을 수행합니다"""
        self.yaw += dx * self.orbit_sensitivity
        self.pitch -= dy * self.orbit_sensitivity  # Y축 반전
        
        # 각도 제한
        self.pitch = max(self.min_pitch, min(self.max_pitch, self.pitch))
        
        # Yaw는 360도 순환
        self.yaw = self.yaw % 360.0
        
        self._update_position()
    
    def pan(self, dx: float, dy: float):
        """팬 이동을 수행합니다"""
        # 카메라의 로컬 축 계산
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        camera_up = np.cross(right, forward)
        camera_up = camera_up / np.linalg.norm(camera_up)
        
        # 팬 이동 계산 (거리에 비례)
        pan_scale = self.distance * self.pan_sensitivity * 0.001
        
        offset = (-dx * right + dy * camera_up) * pan_scale
        self.target += offset
        
        self._update_position()
    
    def zoom(self, delta: float):
        """줌 인/아웃을 수행합니다"""
        zoom_factor = 1.0 + delta * self.zoom_sensitivity
        self.distance *= zoom_factor
        
        # 거리 제한
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
        
        self._update_position()
    
    def reset(self):
        """카메라를 초기 위치로 리셋합니다"""
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 5.0
        self.yaw = 0.0
        self.pitch = 30.0
        self._update_position()
    
    def get_view_matrix(self) -> np.ndarray:
        """뷰 매트릭스를 반환합니다"""
        # Taichi는 내부적으로 뷰 매트릭스를 처리하므로
        # 여기서는 위치와 타겟만 반환
        return {
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist()
        }
    
    def get_info(self) -> Dict[str, Any]:
        """카메라 정보를 반환합니다"""
        return {
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "distance": self.distance,
            "yaw": self.yaw,
            "pitch": self.pitch,
        }


class TaichiViewer3D:
    """Taichi 3D 뷰어 클래스"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.window = None
        self.canvas = None
        self.scene = None
        self.camera_ti = None
        self.camera = Camera3D()
        
        # 렌더링 설정
        self.background_color = (0.1, 0.1, 0.1)
        self.show_wireframe = False
        
        # 성능 최적화: 미리 생성된 필드들
        self.grid_field = None
        self.grid_color_field = None
        self.axes_field = None
        self.axes_color_field = None
        
    def initialize(self) -> bool:
        """뷰어를 초기화합니다"""
        try:
            # 3D UI 윈도우 생성 (ti.ui.Window 사용)
            self.window = ti.ui.Window(
                "Robotics Simulator",
                res=(self.config.window_width, self.config.window_height),
                vsync=self.config.vsync
            )
            
            # 3D 캔버스 생성
            self.canvas = self.window.get_canvas()
            
            # 3D 씬 생성
            self.scene = self.window.get_scene()
            
            # 카메라 생성
            self.camera_ti = ti.ui.Camera()
            
            # 성능 최적화: 정적 지오메트리 미리 생성
            self._create_static_geometry()
            
            print(f"3D 뷰어 초기화 완료: {self.config.window_width}x{self.config.window_height}")
            return True
            
        except Exception as e:
            print(f"3D 뷰어 초기화 실패: {e}")
            return False
    
    def _create_static_geometry(self):
        """정적 지오메트리를 미리 생성합니다 (성능 최적화)"""
        # 그리드 생성
        self._create_grid_geometry(size=10.0, divisions=20)
        
        # 축 생성
        self._create_axes_geometry(length=1.0)
    
    def _create_grid_geometry(self, size: float = 10.0, divisions: int = 20):
        """그리드 지오메트리를 미리 생성합니다"""
        grid_points = []
        grid_colors = []
        
        step = size / divisions
        half_size = size / 2.0
        
        # X축 방향 라인들
        for i in range(divisions + 1):
            x = -half_size + i * step
            grid_points.extend([
                [x, 0.0, -half_size],
                [x, 0.0, half_size]
            ])
            
            # 중앙선은 더 밝게
            if i == divisions // 2:
                grid_colors.extend([[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]])
            else:
                grid_colors.extend([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
        
        # Z축 방향 라인들
        for i in range(divisions + 1):
            z = -half_size + i * step
            grid_points.extend([
                [-half_size, 0.0, z],
                [half_size, 0.0, z]
            ])
            
            # 중앙선은 더 밝게
            if i == divisions // 2:
                grid_colors.extend([[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]])
            else:
                grid_colors.extend([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
        
        # Taichi 필드 생성 (한 번만)
        if grid_points:
            self.grid_field = ti.Vector.field(3, dtype=ti.f32, shape=len(grid_points))
            self.grid_color_field = ti.Vector.field(3, dtype=ti.f32, shape=len(grid_colors))
            
            self.grid_field.from_numpy(np.array(grid_points, dtype=np.float32))
            self.grid_color_field.from_numpy(np.array(grid_colors, dtype=np.float32))
    
    def _create_axes_geometry(self, length: float = 1.0):
        """축 지오메트리를 미리 생성합니다"""
        axes_points = [
            # X축 (빨간색)
            [0.0, 0.0, 0.0], [length, 0.0, 0.0],
            # Y축 (초록색)
            [0.0, 0.0, 0.0], [0.0, length, 0.0],
            # Z축 (파란색)
            [0.0, 0.0, 0.0], [0.0, 0.0, length],
        ]
        
        axes_colors = [
            # X축 (빨간색)
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            # Y축 (초록색)
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            # Z축 (파란색)
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        ]
        
        # Taichi 필드 생성 (한 번만)
        self.axes_field = ti.Vector.field(3, dtype=ti.f32, shape=len(axes_points))
        self.axes_color_field = ti.Vector.field(3, dtype=ti.f32, shape=len(axes_colors))
        
        self.axes_field.from_numpy(np.array(axes_points, dtype=np.float32))
        self.axes_color_field.from_numpy(np.array(axes_colors, dtype=np.float32))
    
    def update_camera_from_input(self, input_state: InputState):
        """입력에 따라 카메라를 업데이트합니다"""
        camera_input = get_camera_control_input(input_state)
        
        # 궤도 회전
        if abs(camera_input["orbit_dx"]) > 0 or abs(camera_input["orbit_dy"]) > 0:
            self.camera.orbit(camera_input["orbit_dx"], camera_input["orbit_dy"])
        
        # 팬 이동
        if abs(camera_input["pan_dx"]) > 0 or abs(camera_input["pan_dy"]) > 0:
            self.camera.pan(camera_input["pan_dx"], camera_input["pan_dy"])
        
        # 줌
        if abs(camera_input["zoom_delta"]) > 0:
            self.camera.zoom(camera_input["zoom_delta"])
    
    def begin_frame(self):
        """프레임 렌더링을 시작합니다"""
        if not self.scene or not self.camera_ti:
            return
        
        # 카메라 설정 업데이트
        view_info = self.camera.get_view_matrix()
        
        self.camera_ti.position(*view_info["position"])
        self.camera_ti.lookat(*view_info["target"])
        self.camera_ti.up(*view_info["up"])
        
        # 씬 설정
        self.scene.set_camera(self.camera_ti)
        
        # 조명 설정
        self.scene.point_light(pos=(2, 4, -1), color=(0.8, 0.8, 0.8))
        self.scene.ambient_light((0.4, 0.4, 0.4))
    
    def end_frame(self):
        """프레임 렌더링을 완료합니다"""
        if not self.canvas or not self.scene:
            return
        
        # 씬을 캔버스에 렌더링
        self.canvas.scene(self.scene)
        
        # 윈도우 표시
        self.window.show()
    
    def draw_ground_grid(self, size: float = 10.0, divisions: int = 20):
        """바닥 그리드를 그립니다 (최적화된 버전)"""
        if not self.scene or not self.grid_field:
            return
        
        # 미리 생성된 필드 사용 (매우 빠름)
        self.scene.lines(self.grid_field, width=1.0, per_vertex_color=self.grid_color_field)
    
    def draw_world_axes(self, length: float = 1.0):
        """월드 축을 그립니다 (최적화된 버전)"""
        if not self.scene or not self.axes_field:
            return
        
        # 미리 생성된 필드 사용 (매우 빠름)
        self.scene.lines(self.axes_field, width=3.0, per_vertex_color=self.axes_color_field)
    
    def is_running(self) -> bool:
        """윈도우가 실행 중인지 확인합니다"""
        return self.window is not None and self.window.running
    
    def get_window(self):
        """GUI 윈도우를 반환합니다"""
        return self.window
    
    def get_camera_info(self) -> Dict[str, Any]:
        """카메라 정보를 반환합니다"""
        return self.camera.get_info()
    
    def reset_camera(self):
        """카메라를 초기 위치로 리셋합니다"""
        self.camera.reset()
        print("카메라 리셋")
    
    def cleanup(self):
        """리소스를 정리합니다"""
        if self.window:
            # ti.ui.Window은 destroy 메서드가 있음
            self.window.destroy()
            self.window = None
        
        self.canvas = None
        self.scene = None
        self.camera_ti = None
        print("3D 뷰어 정리 완료")