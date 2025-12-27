"""충돌 프리미티브 정의"""
import taichi as ti
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Plane:
    """평면 프리미티브"""
    point: Tuple[float, float, float]  # 평면 위의 한 점
    normal: Tuple[float, float, float]  # 평면 법선 벡터 (정규화됨)
    
    def __post_init__(self):
        """초기화 후 법선 벡터 정규화"""
        import math
        nx, ny, nz = self.normal
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            self.normal = (nx/length, ny/length, nz/length)
        else:
            raise ValueError("법선 벡터의 길이가 0입니다")
    
    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """점에서 평면까지의 거리 (부호 있음)"""
        px, py, pz = point
        plane_x, plane_y, plane_z = self.point
        nx, ny, nz = self.normal
        
        # 평면 방정식: n·(p - p0) = 0
        # 거리 = n·(p - p0)
        return nx * (px - plane_x) + ny * (py - plane_y) + nz * (pz - plane_z)
    
    def closest_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """점에서 평면으로의 최근접점"""
        distance = self.distance_to_point(point)
        px, py, pz = point
        nx, ny, nz = self.normal
        
        # 최근접점 = 원점 - 거리 * 법선
        closest_x = px - distance * nx
        closest_y = py - distance * ny
        closest_z = pz - distance * nz
        
        return (closest_x, closest_y, closest_z)


@dataclass
class Sphere:
    """구 프리미티브"""
    center: Tuple[float, float, float]
    radius: float
    
    def __post_init__(self):
        """초기화 후 반지름 검증"""
        if self.radius <= 0:
            raise ValueError("반지름은 0보다 커야 합니다")
    
    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """점에서 구 표면까지의 거리 (부호 있음, 내부는 음수)"""
        import math
        px, py, pz = point
        cx, cy, cz = self.center
        
        # 중심에서 점까지의 거리
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        distance_to_center = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 표면까지의 거리
        return distance_to_center - self.radius
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """점이 구 내부에 있는지 확인"""
        return self.distance_to_point(point) <= 0


@dataclass
class Box:
    """박스 프리미티브 (AABB)"""
    min_point: Tuple[float, float, float]
    max_point: Tuple[float, float, float]
    
    def __post_init__(self):
        """초기화 후 경계 검증"""
        min_x, min_y, min_z = self.min_point
        max_x, max_y, max_z = self.max_point
        
        if min_x >= max_x or min_y >= max_y or min_z >= max_z:
            raise ValueError("최소점이 최대점보다 작아야 합니다")
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """점이 박스 내부에 있는지 확인"""
        px, py, pz = point
        min_x, min_y, min_z = self.min_point
        max_x, max_y, max_z = self.max_point
        
        return (min_x <= px <= max_x and 
                min_y <= py <= max_y and 
                min_z <= pz <= max_z)
    
    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """점에서 박스까지의 거리"""
        px, py, pz = point
        min_x, min_y, min_z = self.min_point
        max_x, max_y, max_z = self.max_point
        
        # 각 축에서 박스 외부 거리 계산
        dx = max(0, max(min_x - px, px - max_x))
        dy = max(0, max(min_y - py, py - max_y))
        dz = max(0, max(min_z - pz, pz - max_z))
        
        import math
        return math.sqrt(dx*dx + dy*dy + dz*dz)


# 편의 함수들
def create_ground_plane(y: float = 0.0) -> Plane:
    """지면 평면 생성 (XZ 평면, Y=y)"""
    return Plane(
        point=(0.0, y, 0.0),
        normal=(0.0, 1.0, 0.0)
    )


def create_wall_planes(bounds: Tuple[float, float, float, float, float, float]) -> list[Plane]:
    """경계 벽 평면들 생성"""
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    
    walls = [
        # X 방향 벽들
        Plane(point=(min_x, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),   # 왼쪽 벽
        Plane(point=(max_x, 0.0, 0.0), normal=(-1.0, 0.0, 0.0)),  # 오른쪽 벽
        
        # Y 방향 벽들
        Plane(point=(0.0, min_y, 0.0), normal=(0.0, 1.0, 0.0)),   # 바닥
        Plane(point=(0.0, max_y, 0.0), normal=(0.0, -1.0, 0.0)),  # 천장
        
        # Z 방향 벽들
        Plane(point=(0.0, 0.0, min_z), normal=(0.0, 0.0, 1.0)),   # 앞쪽 벽
        Plane(point=(0.0, 0.0, max_z), normal=(0.0, 0.0, -1.0)),  # 뒤쪽 벽
    ]
    
    return walls
