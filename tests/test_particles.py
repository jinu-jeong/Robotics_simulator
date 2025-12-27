"""파티클 시스템 테스트"""
import unittest
import sys
import os
import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Taichi 초기화 (테스트용)
import taichi as ti
ti.init(arch=ti.cpu)

from sim.physics.particles import ParticleSystem
from sim.collision.primitives import Plane, create_ground_plane
from sim.collision.contact import CollisionWorld, ParticlePlaneCollision


class TestParticleSystem(unittest.TestCase):
    """파티클 시스템 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.particle_system = ParticleSystem(max_particles=100)
    
    def test_initialization(self):
        """초기화 테스트"""
        # 초기 상태 확인
        self.assertEqual(self.particle_system.num_particles, 0)
        self.assertEqual(self.particle_system.max_particles, 100)
        
        # 리셋 후 파티클 생성 확인
        self.particle_system.reset(seed=42)
        self.assertGreater(self.particle_system.num_particles, 0)
        self.assertLessEqual(self.particle_system.num_particles, 100)
        
        print(f"파티클 수: {self.particle_system.num_particles}")
    
    def test_deterministic_reset(self):
        """결정적 리셋 테스트"""
        # 같은 시드로 두 번 리셋
        self.particle_system.reset(seed=42)
        positions1 = self.particle_system.get_positions().copy()
        
        self.particle_system.reset(seed=42)
        positions2 = self.particle_system.get_positions().copy()
        
        # 위치가 동일해야 함
        np.testing.assert_array_almost_equal(positions1, positions2, decimal=5)
        print("결정적 리셋 확인됨")
    
    def test_particle_properties(self):
        """파티클 속성 테스트"""
        self.particle_system.reset(seed=42)
        
        positions = self.particle_system.get_positions()
        velocities = self.particle_system.get_velocities()
        radii = self.particle_system.get_radii()
        contact_mask = self.particle_system.get_contact_mask()
        
        # 배열 크기 확인
        num_particles = self.particle_system.num_particles
        self.assertEqual(len(positions), num_particles)
        self.assertEqual(len(velocities), num_particles)
        self.assertEqual(len(radii), num_particles)
        self.assertEqual(len(contact_mask), num_particles)
        
        # 초기 속도는 0이어야 함
        np.testing.assert_array_almost_equal(velocities, 0.0, decimal=5)
        
        # 모든 파티클이 지면 위에 있어야 함
        self.assertTrue(np.all(positions[:, 1] > 0.0))
        
        # 반지름이 양수여야 함
        self.assertTrue(np.all(radii > 0.0))
        
        print(f"파티클 속성 확인됨 - 평균 높이: {np.mean(positions[:, 1]):.3f}")
    
    def test_physics_step(self):
        """물리 스텝 테스트"""
        self.particle_system.reset(seed=42)
        
        # 초기 상태 저장
        initial_positions = self.particle_system.get_positions().copy()
        initial_velocities = self.particle_system.get_velocities().copy()
        
        # 몇 스텝 실행
        dt = 0.016
        for _ in range(10):
            self.particle_system.step(dt)
        
        # 상태 변화 확인
        final_positions = self.particle_system.get_positions()
        final_velocities = self.particle_system.get_velocities()
        
        # 위치가 변해야 함 (중력으로 인해)
        position_changed = not np.allclose(initial_positions, final_positions, atol=1e-6)
        self.assertTrue(position_changed, "위치가 변하지 않음")
        
        # 속도가 변해야 함 (중력으로 인해)
        velocity_changed = not np.allclose(initial_velocities, final_velocities, atol=1e-6)
        self.assertTrue(velocity_changed, "속도가 변하지 않음")
        
        print(f"물리 스텝 후 평균 높이: {np.mean(final_positions[:, 1]):.3f}")
    
    def test_ground_contact(self):
        """지면 접촉 테스트"""
        self.particle_system.reset(seed=42)
        
        # 충분히 많은 스텝 실행 (파티클이 지면에 떨어질 때까지)
        dt = 0.016
        for _ in range(200):  # 약 3.2초
            self.particle_system.step(dt)
        
        # 접촉 상태 확인
        contact_mask = self.particle_system.get_contact_mask()
        positions = self.particle_system.get_positions()
        
        # 일부 파티클이 지면과 접촉해야 함
        num_contacts = np.sum(contact_mask)
        self.assertGreater(num_contacts, 0, "지면과 접촉한 파티클이 없음")
        
        # 접촉한 파티클들이 지면 근처에 있어야 함
        contact_indices = np.where(contact_mask > 0)[0]
        if len(contact_indices) > 0:
            contact_heights = positions[contact_indices, 1]
            max_contact_height = np.max(contact_heights)
            # 파티클 반지름 + 약간의 여유
            expected_max_height = self.particle_system.default_radius + 0.1
            self.assertLess(max_contact_height, expected_max_height, 
                          f"접촉 파티클 높이가 너무 높음: {max_contact_height}")
        
        print(f"지면 접촉 파티클 수: {num_contacts}")
    
    def test_contact_parameters(self):
        """접촉 파라미터 테스트"""
        self.particle_system.reset(seed=42)
        
        # 파라미터 변경
        new_stiffness = 2000.0
        new_damping = 100.0
        self.particle_system.set_contact_parameters(new_stiffness, new_damping)
        
        # 값이 설정되었는지 확인
        self.assertEqual(self.particle_system.contact_stiffness, new_stiffness)
        self.assertEqual(self.particle_system.contact_damping, new_damping)
        
        print("접촉 파라미터 설정 확인됨")
    
    def test_statistics(self):
        """통계 테스트"""
        self.particle_system.reset(seed=42)
        
        # 초기 통계
        stats = self.particle_system.get_stats()
        
        # 필수 키 확인
        required_keys = ['num_particles', 'num_contacts', 'avg_height', 'total_kinetic_energy']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # 값 범위 확인
        self.assertGreaterEqual(stats['num_particles'], 0)
        self.assertGreaterEqual(stats['num_contacts'], 0)
        self.assertGreaterEqual(stats['total_kinetic_energy'], 0.0)
        
        print(f"초기 통계: {stats}")
        
        # 몇 스텝 후 통계
        for _ in range(50):
            self.particle_system.step(0.016)
        
        new_stats = self.particle_system.get_stats()
        print(f"시뮬레이션 후 통계: {new_stats}")


class TestCollisionPrimitives(unittest.TestCase):
    """충돌 프리미티브 테스트"""
    
    def test_ground_plane(self):
        """지면 평면 테스트"""
        ground = create_ground_plane(y=0.0)
        
        # 지면 위 점
        distance_above = ground.distance_to_point((0.0, 1.0, 0.0))
        self.assertAlmostEqual(distance_above, 1.0, places=5)
        
        # 지면 아래 점
        distance_below = ground.distance_to_point((0.0, -1.0, 0.0))
        self.assertAlmostEqual(distance_below, -1.0, places=5)
        
        # 지면 위 점
        distance_on = ground.distance_to_point((0.0, 0.0, 0.0))
        self.assertAlmostEqual(distance_on, 0.0, places=5)
        
        print("지면 평면 거리 계산 확인됨")
    
    def test_plane_collision(self):
        """평면 충돌 테스트"""
        ground = create_ground_plane(y=0.0)
        collision = ParticlePlaneCollision(ground)
        
        # 충돌 없음
        is_colliding, penetration = collision.check_collision((0.0, 1.0, 0.0), 0.5)
        self.assertFalse(is_colliding)
        self.assertEqual(penetration, 0.0)
        
        # 충돌 있음
        is_colliding, penetration = collision.check_collision((0.0, 0.3, 0.0), 0.5)
        self.assertTrue(is_colliding)
        self.assertAlmostEqual(penetration, 0.2, places=5)
        
        print("평면 충돌 감지 확인됨")


def run_particle_tests():
    """파티클 테스트 실행"""
    print("=== 파티클 시스템 테스트 시작 ===")
    
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 클래스 추가
    suite.addTests(loader.loadTestsFromTestCase(TestParticleSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestCollisionPrimitives))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n=== 테스트 결과 ===")
    print(f"실행된 테스트: {result.testsRun}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n전체 결과: {'성공' if success else '실패'}")
    
    return success


if __name__ == "__main__":
    success = run_particle_tests()
    sys.exit(0 if success else 1)
