"""파티클 시스템 구현"""
import taichi as ti
import numpy as np
from typing import Tuple, Optional
import random


@ti.data_oriented
class ParticleSystem:
    """파티클 시스템 클래스"""
    
    def __init__(self, max_particles: int = 1000):
        """
        Args:
            max_particles: 최대 파티클 수
        """
        self.max_particles = max_particles
        self.num_particles = 0
        
        # 파티클 속성 필드
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.mass = ti.field(dtype=ti.f32, shape=max_particles)
        self.radius = ti.field(dtype=ti.f32, shape=max_particles)
        
        # 접촉 상태 (지면과의 접촉 여부)
        self.in_contact = ti.field(dtype=ti.i32, shape=max_particles)
        
        # 물리 파라미터
        self.gravity = ti.Vector([0.0, -9.81, 0.0])
        self.ground_y = 0.0
        
        # 접촉 파라미터 (스프링-댐퍼)
        self.contact_stiffness = 1000.0
        self.contact_damping = 50.0
        
        # 기본 파티클 속성
        self.default_mass = 1.0
        self.default_radius = 0.05
    
    def reset(self, seed: Optional[int] = None) -> None:
        """파티클 시스템 리셋 및 초기화"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 3D 파일 형태로 파티클 배치
        self._initialize_particle_pile()
        
        print(f"파티클 시스템 리셋: {self.num_particles}개 파티클")
    
    def _initialize_particle_pile(self) -> None:
        """3D 파일 형태로 파티클 초기화"""
        # 그리드 설정
        grid_size = 8  # 8x8x8 그리드
        spacing = self.default_radius * 2.2  # 겹치지 않도록 간격 설정
        jitter_amount = spacing * 0.1  # 작은 지터
        
        self.num_particles = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if self.num_particles >= self.max_particles:
                        break
                    
                    # 그리드 위치 계산 (중앙 정렬)
                    x = (i - grid_size/2) * spacing
                    y = (j + 1) * spacing + 1.0  # 지면 위에 배치
                    z = (k - grid_size/2) * spacing
                    
                    # 작은 지터 추가
                    x += (random.random() - 0.5) * jitter_amount
                    y += (random.random() - 0.5) * jitter_amount
                    z += (random.random() - 0.5) * jitter_amount
                    
                    # 파티클 속성 설정
                    idx = self.num_particles
                    self.position[idx] = ti.Vector([x, y, z])
                    self.velocity[idx] = ti.Vector([0.0, 0.0, 0.0])
                    self.force[idx] = ti.Vector([0.0, 0.0, 0.0])
                    self.mass[idx] = self.default_mass
                    self.radius[idx] = self.default_radius
                    self.in_contact[idx] = 0
                    
                    self.num_particles += 1
    
    @ti.kernel
    def clear_forces(self):
        """모든 파티클의 힘 초기화"""
        for i in range(self.num_particles):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.in_contact[i] = 0
    
    @ti.kernel
    def apply_gravity(self):
        """중력 적용"""
        for i in range(self.num_particles):
            self.force[i] += self.mass[i] * self.gravity
    
    @ti.kernel
    def apply_ground_contact(self):
        """지면 접촉 처리 (스프링-댐퍼 모델)"""
        for i in range(self.num_particles):
            pos = self.position[i]
            vel = self.velocity[i]
            r = self.radius[i]
            
            # 지면과의 거리 계산
            ground_distance = pos.y - self.ground_y
            penetration = r - ground_distance
            
            if penetration > 0.0:  # 접촉 중
                self.in_contact[i] = 1
                
                # 법선 벡터 (위쪽)
                normal = ti.Vector([0.0, 1.0, 0.0])
                
                # 스프링 힘 (침투 깊이에 비례)
                spring_force = self.contact_stiffness * penetration
                
                # 댐핑 힘 (법선 방향 속도에 비례)
                normal_velocity = vel.dot(normal)
                damping_force = -self.contact_damping * normal_velocity
                
                # 총 접촉 힘
                total_force = (spring_force + damping_force) * normal
                
                # 힘 적용 (위쪽으로만)
                if total_force.y > 0.0:
                    self.force[i] += total_force
    
    @ti.kernel
    def integrate(self, dt: ti.f32):
        """Semi-implicit Euler 적분"""
        for i in range(self.num_particles):
            # 가속도 계산
            acc = self.force[i] / self.mass[i]
            
            # 속도 업데이트 (semi-implicit)
            self.velocity[i] += acc * dt
            
            # 위치 업데이트
            self.position[i] += self.velocity[i] * dt
    
    def step(self, dt: float) -> None:
        """물리 시뮬레이션 한 스텝"""
        self.clear_forces()
        self.apply_gravity()
        self.apply_ground_contact()
        self.integrate(dt)
    
    def set_contact_parameters(self, stiffness: float, damping: float) -> None:
        """접촉 파라미터 설정"""
        self.contact_stiffness = stiffness
        self.contact_damping = damping
    
    def get_positions(self) -> np.ndarray:
        """파티클 위치 배열 반환"""
        if self.num_particles == 0:
            return np.empty((0, 3), dtype=np.float32)
        return self.position.to_numpy()[:self.num_particles]
    
    def get_velocities(self) -> np.ndarray:
        """파티클 속도 배열 반환"""
        if self.num_particles == 0:
            return np.empty((0, 3), dtype=np.float32)
        return self.velocity.to_numpy()[:self.num_particles]
    
    def get_contact_mask(self) -> np.ndarray:
        """접촉 마스크 배열 반환"""
        if self.num_particles == 0:
            return np.empty(0, dtype=np.int32)
        return self.in_contact.to_numpy()[:self.num_particles]
    
    def get_radii(self) -> np.ndarray:
        """파티클 반지름 배열 반환"""
        if self.num_particles == 0:
            return np.empty(0, dtype=np.float32)
        return self.radius.to_numpy()[:self.num_particles]
    
    def get_stats(self) -> dict:
        """시스템 통계 반환"""
        if self.num_particles == 0:
            return {
                "num_particles": 0,
                "num_contacts": 0,
                "avg_height": 0.0,
                "total_kinetic_energy": 0.0
            }
        
        positions = self.get_positions()
        velocities = self.get_velocities()
        contact_mask = self.get_contact_mask()
        masses = self.mass.to_numpy()[:self.num_particles]
        
        # 통계 계산
        num_contacts = np.sum(contact_mask)
        avg_height = np.mean(positions[:, 1])
        
        # 운동 에너지 계산
        kinetic_energies = 0.5 * masses * np.sum(velocities**2, axis=1)
        total_kinetic_energy = np.sum(kinetic_energies)
        
        return {
            "num_particles": self.num_particles,
            "num_contacts": int(num_contacts),
            "avg_height": float(avg_height),
            "total_kinetic_energy": float(total_kinetic_energy)
        }
