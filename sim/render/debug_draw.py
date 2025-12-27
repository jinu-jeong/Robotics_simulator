"""
디버그 그리기 헬퍼 모듈
"""
import taichi as ti
import numpy as np
import math
from typing import List, Tuple, Optional, Union


class DebugDraw:
    """디버그 그리기를 위한 헬퍼 클래스"""
    
    def __init__(self):
        # 기본 색상 정의
        self.colors = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5),
            "light_gray": (0.7, 0.7, 0.7),
            "dark_gray": (0.3, 0.3, 0.3),
        }
    
    def draw_grid(self, scene, size: float = 10.0, divisions: int = 20, 
                  color: Union[str, Tuple[float, float, float]] = "gray",
                  center_color: Union[str, Tuple[float, float, float]] = "light_gray",
                  y_level: float = 0.0) -> None:
        """
        XZ 평면에 그리드를 그립니다.
        
        Args:
            scene: Taichi 씬 객체
            size: 그리드 전체 크기
            divisions: 분할 수
            color: 일반 그리드 라인 색상
            center_color: 중앙 라인 색상
            y_level: 그리드의 Y 좌표
        """
        if not scene:
            return
        
        # 색상 처리
        grid_color = self._resolve_color(color)
        center_grid_color = self._resolve_color(center_color)
        
        # 그리드 라인을 위한 점들 생성
        grid_points = []
        grid_colors = []
        
        step = size / divisions
        half_size = size / 2.0
        
        # X축 방향 라인들
        for i in range(divisions + 1):
            x = -half_size + i * step
            grid_points.extend([
                [x, y_level, -half_size],
                [x, y_level, half_size]
            ])
            
            # 중앙선은 다른 색상
            if i == divisions // 2:
                grid_colors.extend([center_grid_color, center_grid_color])
            else:
                grid_colors.extend([grid_color, grid_color])
        
        # Z축 방향 라인들
        for i in range(divisions + 1):
            z = -half_size + i * step
            grid_points.extend([
                [-half_size, y_level, z],
                [half_size, y_level, z]
            ])
            
            # 중앙선은 다른 색상
            if i == divisions // 2:
                grid_colors.extend([center_grid_color, center_grid_color])
            else:
                grid_colors.extend([grid_color, grid_color])
        
        # Taichi 필드로 변환하여 렌더링
        if grid_points:
            self._render_lines(scene, grid_points, grid_colors, width=1.0)
    
    def draw_axes(self, scene, length: float = 1.0, width: float = 3.0,
                  origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """
        월드 좌표축을 그립니다.
        
        Args:
            scene: Taichi 씬 객체
            length: 축의 길이
            width: 라인 두께
            origin: 축의 원점
        """
        if not scene:
            return
        
        ox, oy, oz = origin
        
        # 축 라인 정의
        axes_points = [
            # X축 (빨간색)
            [ox, oy, oz], [ox + length, oy, oz],
            # Y축 (초록색)
            [ox, oy, oz], [ox, oy + length, oz],
            # Z축 (파란색)
            [ox, oy, oz], [ox, oy, oz + length],
        ]
        
        axes_colors = [
            # X축 (빨간색)
            self.colors["red"], self.colors["red"],
            # Y축 (초록색)
            self.colors["green"], self.colors["green"],
            # Z축 (파란색)
            self.colors["blue"], self.colors["blue"],
        ]
        
        self._render_lines(scene, axes_points, axes_colors, width=width)
    
    def draw_line(self, scene, start: Tuple[float, float, float], 
                  end: Tuple[float, float, float],
                  color: Union[str, Tuple[float, float, float]] = "white",
                  width: float = 2.0) -> None:
        """
        단일 라인을 그립니다.
        
        Args:
            scene: Taichi 씬 객체
            start: 시작점
            end: 끝점
            color: 라인 색상
            width: 라인 두께
        """
        if not scene:
            return
        
        line_color = self._resolve_color(color)
        points = [list(start), list(end)]
        colors = [line_color, line_color]
        
        self._render_lines(scene, points, colors, width=width)
    
    def draw_box_wireframe(self, scene, center: Tuple[float, float, float],
                          size: Tuple[float, float, float],
                          color: Union[str, Tuple[float, float, float]] = "white",
                          width: float = 2.0) -> None:
        """
        박스의 와이어프레임을 그립니다.
        
        Args:
            scene: Taichi 씬 객체
            center: 박스 중심점
            size: 박스 크기 (width, height, depth)
            color: 라인 색상
            width: 라인 두께
        """
        if not scene:
            return
        
        cx, cy, cz = center
        w, h, d = size
        hw, hh, hd = w/2, h/2, d/2
        
        # 박스의 8개 꼭짓점
        vertices = [
            [cx-hw, cy-hh, cz-hd],  # 0: 왼쪽 아래 앞
            [cx+hw, cy-hh, cz-hd],  # 1: 오른쪽 아래 앞
            [cx+hw, cy+hh, cz-hd],  # 2: 오른쪽 위 앞
            [cx-hw, cy+hh, cz-hd],  # 3: 왼쪽 위 앞
            [cx-hw, cy-hh, cz+hd],  # 4: 왼쪽 아래 뒤
            [cx+hw, cy-hh, cz+hd],  # 5: 오른쪽 아래 뒤
            [cx+hw, cy+hh, cz+hd],  # 6: 오른쪽 위 뒤
            [cx-hw, cy+hh, cz+hd],  # 7: 왼쪽 위 뒤
        ]
        
        # 박스의 12개 모서리
        edges = [
            # 앞면
            (0, 1), (1, 2), (2, 3), (3, 0),
            # 뒷면
            (4, 5), (5, 6), (6, 7), (7, 4),
            # 연결선
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        line_color = self._resolve_color(color)
        points = []
        colors = []
        
        for start_idx, end_idx in edges:
            points.extend([vertices[start_idx], vertices[end_idx]])
            colors.extend([line_color, line_color])
        
        self._render_lines(scene, points, colors, width=width)
    
    def draw_sphere_wireframe(self, scene, center: Tuple[float, float, float],
                             radius: float, segments: int = 16,
                             color: Union[str, Tuple[float, float, float]] = "white",
                             width: float = 2.0) -> None:
        """
        구의 와이어프레임을 그립니다.
        
        Args:
            scene: Taichi 씬 객체
            center: 구 중심점
            radius: 구 반지름
            segments: 분할 수
            color: 라인 색상
            width: 라인 두께
        """
        if not scene:
            return
        
        cx, cy, cz = center
        line_color = self._resolve_color(color)
        points = []
        colors = []
        
        # XY 평면 원
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            
            x1 = cx + radius * math.cos(angle1)
            y1 = cy + radius * math.sin(angle1)
            x2 = cx + radius * math.cos(angle2)
            y2 = cy + radius * math.sin(angle2)
            
            points.extend([[x1, y1, cz], [x2, y2, cz]])
            colors.extend([line_color, line_color])
        
        # XZ 평면 원
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            
            x1 = cx + radius * math.cos(angle1)
            z1 = cz + radius * math.sin(angle1)
            x2 = cx + radius * math.cos(angle2)
            z2 = cz + radius * math.sin(angle2)
            
            points.extend([[x1, cy, z1], [x2, cy, z2]])
            colors.extend([line_color, line_color])
        
        # YZ 평면 원
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            
            y1 = cy + radius * math.cos(angle1)
            z1 = cz + radius * math.sin(angle1)
            y2 = cy + radius * math.cos(angle2)
            z2 = cz + radius * math.sin(angle2)
            
            points.extend([[cx, y1, z1], [cx, y2, z2]])
            colors.extend([line_color, line_color])
        
        self._render_lines(scene, points, colors, width=width)
    
    def draw_arrow(self, scene, start: Tuple[float, float, float],
                   direction: Tuple[float, float, float], length: float = 1.0,
                   color: Union[str, Tuple[float, float, float]] = "yellow",
                   width: float = 2.0, head_size: float = 0.1) -> None:
        """
        화살표를 그립니다.
        
        Args:
            scene: Taichi 씬 객체
            start: 시작점
            direction: 방향 벡터 (정규화됨)
            length: 화살표 길이
            color: 화살표 색상
            width: 라인 두께
            head_size: 화살표 머리 크기
        """
        if not scene:
            return
        
        # 방향 벡터 정규화
        dir_vec = np.array(direction)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        
        start_pos = np.array(start)
        end_pos = start_pos + dir_vec * length
        
        # 화살표 몸체
        self.draw_line(scene, start_pos.tolist(), end_pos.tolist(), color, width)
        
        # 화살표 머리 (간단한 V 모양)
        # 수직 벡터 찾기
        if abs(dir_vec[1]) < 0.9:
            up = np.array([0, 1, 0])
        else:
            up = np.array([1, 0, 0])
        
        right = np.cross(dir_vec, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, dir_vec)
        
        head_length = length * head_size
        head_width = head_length * 0.5
        
        # 화살표 머리의 두 점
        head1 = end_pos - dir_vec * head_length + right * head_width
        head2 = end_pos - dir_vec * head_length - right * head_width
        
        self.draw_line(scene, end_pos.tolist(), head1.tolist(), color, width)
        self.draw_line(scene, end_pos.tolist(), head2.tolist(), color, width)
    
    def _resolve_color(self, color: Union[str, Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """색상을 RGB 튜플로 변환합니다"""
        if isinstance(color, str):
            return self.colors.get(color, self.colors["white"])
        return color
    
    def _render_lines(self, scene, points: List[List[float]], 
                     colors: List[Tuple[float, float, float]], width: float = 2.0):
        """라인들을 렌더링합니다"""
        if not points or not colors:
            return
        
        # Taichi 필드 생성
        num_points = len(points)
        points_field = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
        colors_field = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
        
        # 데이터 복사
        points_field.from_numpy(np.array(points, dtype=np.float32))
        colors_field.from_numpy(np.array(colors, dtype=np.float32))
        
        # 라인 렌더링
        scene.lines(points_field, width=width, per_vertex_color=colors_field)