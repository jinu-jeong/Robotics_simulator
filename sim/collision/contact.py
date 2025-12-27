"""접촉 처리 및 충돌 감지"""
import taichi as ti
import numpy as np
from typing import Tuple, List, Optional
from .primitives import Plane, Sphere, Box


@ti.data_oriented
class ContactResolver:
    """접촉 해결 클래스"""
    
    def __init__(self):
        """초기화"""
        # 접촉 파라미터
        self.restitution = 0.3  # 반발 계수
        self.friction = 0.5     # 마찰 계수
        
        # 안정성 파라미터
        self.position_correction = 0.8  # 위치 보정 강도
        self.velocity_threshold = 0.01  # 속도 임계값
    
    def resolve_particle_plane_contact(self, 
                                     position: ti.Vector,
                                     velocity: ti.Vector,
                                     radius: float,
                                     mass: float,
                                     plane_point: ti.Vector,
                                     plane_normal: ti.Vector,
                                     dt: float) -> Tuple[ti.Vector, ti.Vector, bool]:
        """
        파티클-평면 접촉 해결
        
        Returns:
            (new_position, new_velocity, in_contact)
        """
        # 평면까지의 거리 계산
        to_point = position - plane_point
        distance = to_point.dot(plane_normal)
        penetration = radius - distance
        
        in_contact = False
        new_pos = position
        new_vel = velocity
        
        if penetration > 0.0:  # 접촉 중
            in_contact = True
            
            # 위치 보정 (침투 해결)
            correction = penetration * self.position_correction
            new_pos = position + plane_normal * correction
            
            # 속도 보정
            normal_velocity = velocity.dot(plane_normal)
            
            if normal_velocity < 0.0:  # 평면으로 향하는 속도
                # 법선 방향 속도 성분
                vel_normal = normal_velocity * plane_normal
                # 접선 방향 속도 성분
                vel_tangent = velocity - vel_normal
                
                # 반발 적용
                new_vel_normal = -self.restitution * vel_normal
                
                # 마찰 적용
                tangent_speed = vel_tangent.norm()
                if tangent_speed > self.velocity_threshold:
                    friction_impulse = min(self.friction * abs(normal_velocity), tangent_speed)
                    friction_dir = vel_tangent.normalized()
                    new_vel_tangent = vel_tangent - friction_impulse * friction_dir
                else:
                    new_vel_tangent = ti.Vector([0.0, 0.0, 0.0])
                
                new_vel = new_vel_normal + new_vel_tangent
        
        return new_pos, new_vel, in_contact


class ParticlePlaneCollision:
    """파티클-평면 충돌 처리"""
    
    def __init__(self, plane: Plane):
        """
        Args:
            plane: 평면 프리미티브
        """
        self.plane = plane
        self.resolver = ContactResolver()
    
    def check_collision(self, position: Tuple[float, float, float], radius: float) -> Tuple[bool, float]:
        """
        충돌 검사
        
        Returns:
            (is_colliding, penetration_depth)
        """
        distance = self.plane.distance_to_point(position)
        penetration = radius - distance
        
        return penetration > 0.0, max(0.0, penetration)
    
    def resolve_collision(self, 
                         position: Tuple[float, float, float],
                         velocity: Tuple[float, float, float],
                         radius: float,
                         mass: float,
                         dt: float) -> Tuple[Tuple[float, float, float], 
                                           Tuple[float, float, float], 
                                           bool]:
        """
        충돌 해결
        
        Returns:
            (new_position, new_velocity, in_contact)
        """
        # Taichi 벡터로 변환
        pos_vec = ti.Vector(list(position))
        vel_vec = ti.Vector(list(velocity))
        plane_point_vec = ti.Vector(list(self.plane.point))
        plane_normal_vec = ti.Vector(list(self.plane.normal))
        
        # 접촉 해결
        new_pos, new_vel, in_contact = self.resolver.resolve_particle_plane_contact(
            pos_vec, vel_vec, radius, mass, plane_point_vec, plane_normal_vec, dt
        )
        
        # 튜플로 변환하여 반환
        return (tuple(new_pos), tuple(new_vel), in_contact)


class CollisionWorld:
    """충돌 월드 관리"""
    
    def __init__(self):
        """초기화"""
        self.planes: List[Plane] = []
        self.spheres: List[Sphere] = []
        self.boxes: List[Box] = []
        
        # 기본 지면 추가
        self.add_ground_plane()
    
    def add_ground_plane(self, y: float = 0.0) -> None:
        """지면 평면 추가"""
        from .primitives import create_ground_plane
        ground = create_ground_plane(y)
        self.planes.append(ground)
    
    def add_plane(self, plane: Plane) -> None:
        """평면 추가"""
        self.planes.append(plane)
    
    def add_sphere(self, sphere: Sphere) -> None:
        """구 추가"""
        self.spheres.append(sphere)
    
    def add_box(self, box: Box) -> None:
        """박스 추가"""
        self.boxes.append(box)
    
    def check_particle_collisions(self, 
                                position: Tuple[float, float, float],
                                radius: float) -> List[Tuple[str, int, float]]:
        """
        파티클의 모든 충돌 검사
        
        Returns:
            List of (primitive_type, index, penetration_depth)
        """
        collisions = []
        
        # 평면과의 충돌 검사
        for i, plane in enumerate(self.planes):
            distance = plane.distance_to_point(position)
            penetration = radius - distance
            if penetration > 0.0:
                collisions.append(("plane", i, penetration))
        
        # 구와의 충돌 검사
        for i, sphere in enumerate(self.spheres):
            distance = sphere.distance_to_point(position)
            penetration = radius - distance
            if penetration > 0.0:
                collisions.append(("sphere", i, penetration))
        
        # 박스와의 충돌 검사
        for i, box in enumerate(self.boxes):
            if box.contains_point(position):
                distance = box.distance_to_point(position)
                penetration = radius - distance
                if penetration > 0.0:
                    collisions.append(("box", i, penetration))
        
        return collisions
    
    def resolve_particle_collisions(self,
                                  position: Tuple[float, float, float],
                                  velocity: Tuple[float, float, float],
                                  radius: float,
                                  mass: float,
                                  dt: float) -> Tuple[Tuple[float, float, float], 
                                                    Tuple[float, float, float], 
                                                    bool]:
        """
        파티클의 모든 충돌 해결
        
        Returns:
            (new_position, new_velocity, any_contact)
        """
        current_pos = position
        current_vel = velocity
        any_contact = False
        
        # 평면과의 충돌 해결 (가장 깊은 침투부터)
        plane_collisions = []
        for i, plane in enumerate(self.planes):
            distance = plane.distance_to_point(current_pos)
            penetration = radius - distance
            if penetration > 0.0:
                plane_collisions.append((penetration, i, plane))
        
        # 침투 깊이 순으로 정렬
        plane_collisions.sort(key=lambda x: x[0], reverse=True)
        
        for penetration, i, plane in plane_collisions:
            collision_handler = ParticlePlaneCollision(plane)
            current_pos, current_vel, in_contact = collision_handler.resolve_collision(
                current_pos, current_vel, radius, mass, dt
            )
            if in_contact:
                any_contact = True
        
        return current_pos, current_vel, any_contact
    
    def get_ground_plane(self) -> Optional[Plane]:
        """지면 평면 반환 (첫 번째 평면)"""
        return self.planes[0] if self.planes else None
