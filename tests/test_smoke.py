"""스모크 테스트 - 기본 기능 검증"""
import unittest
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core.config import SimConfig
from sim.core.app import get_available_archs
from sim.core.time import FixedStepper
from sim.core.stats import RuntimeStats


class TestSmokeTests(unittest.TestCase):
    """기본 기능 스모크 테스트"""
    
    def test_config_validation(self):
        """설정 검증 테스트"""
        # 정상 설정
        config = SimConfig()
        self.assertEqual(config.dt, 1.0/60.0)
        self.assertEqual(config.substeps, 1)
        self.assertEqual(config.arch, "metal")
        
        # 잘못된 dt
        with self.assertRaises(ValueError):
            SimConfig(dt=0.0)
        
        with self.assertRaises(ValueError):
            SimConfig(dt=-0.1)
        
        # 잘못된 substeps
        with self.assertRaises(ValueError):
            SimConfig(substeps=0)
        
        # 잘못된 윈도우 크기
        with self.assertRaises(ValueError):
            SimConfig(window_width=0)
        
        with self.assertRaises(ValueError):
            SimConfig(window_height=-1)
        
        # 잘못된 아키텍처
        with self.assertRaises(ValueError):
            SimConfig(arch="invalid")
    
    def test_arch_mapping(self):
        """아키텍처 매핑 테스트"""
        available_archs = get_available_archs()
        
        # CPU는 항상 사용 가능해야 함
        self.assertIn("cpu", available_archs)
        
        # 각 아키텍처가 유효한 문자열인지 확인
        valid_archs = {"cpu", "metal", "vulkan", "cuda"}
        for arch in available_archs:
            self.assertIn(arch, valid_archs)
        
        print(f"사용 가능한 아키텍처: {available_archs}")
    
    def test_fixed_stepper(self):
        """고정 타임스텝 테스트"""
        dt = 0.016  # ~60 FPS
        substeps = 2
        stepper = FixedStepper(dt, substeps)
        
        # 초기 상태
        self.assertEqual(stepper.sim_time, 0.0)
        self.assertEqual(stepper.frame_count, 0)
        self.assertFalse(stepper.paused)
        self.assertEqual(stepper.dt, dt)
        self.assertEqual(stepper.substeps, substeps)
        self.assertEqual(stepper.sub_dt, dt / substeps)
        
        # 스텝 실행
        stepper.step()
        self.assertEqual(stepper.sim_time, dt)
        self.assertEqual(stepper.frame_count, 1)
        
        # 일시정지
        stepper.pause()
        self.assertTrue(stepper.paused)
        
        # 재개
        stepper.resume()
        self.assertFalse(stepper.paused)
        
        # 리셋
        stepper.step()  # 한 번 더 진행
        stepper.reset()
        self.assertEqual(stepper.sim_time, 0.0)
        self.assertEqual(stepper.frame_count, 0)
    
    def test_runtime_stats(self):
        """런타임 통계 테스트"""
        stats = RuntimeStats(window_size=5)
        
        # 초기 상태
        self.assertEqual(stats.fps, 0.0)
        self.assertEqual(stats.total_frames, 0)
        
        # 프레임 시작 (여러 번)
        import time
        for i in range(3):
            stats.begin_frame()
            time.sleep(0.001)  # 짧은 대기
        
        # 통계 확인
        self.assertGreater(stats.total_frames, 0)
        self.assertGreaterEqual(stats.fps, 0.0)
        
        # 리셋
        stats.reset()
        self.assertEqual(stats.total_frames, 0)
        self.assertEqual(stats.fps, 0.0)
    
    def test_config_with_different_archs(self):
        """다양한 아키텍처로 설정 테스트"""
        available_archs = get_available_archs()
        
        for arch in available_archs:
            config = SimConfig(arch=arch)
            self.assertEqual(config.arch, arch)
            print(f"아키텍처 '{arch}' 설정 성공")


class TestConfigEdgeCases(unittest.TestCase):
    """설정 엣지 케이스 테스트"""
    
    def test_extreme_values(self):
        """극한값 테스트"""
        # 매우 작은 dt
        config = SimConfig(dt=1e-6)
        self.assertEqual(config.dt, 1e-6)
        
        # 매우 큰 substeps
        config = SimConfig(substeps=1000)
        self.assertEqual(config.substeps, 1000)
        
        # 큰 윈도우 크기
        config = SimConfig(window_width=4096, window_height=4096)
        self.assertEqual(config.window_width, 4096)
        self.assertEqual(config.window_height, 4096)
    
    def test_seed_handling(self):
        """시드 처리 테스트"""
        # None 시드
        config = SimConfig(seed=None)
        self.assertIsNone(config.seed)
        
        # 정수 시드
        config = SimConfig(seed=12345)
        self.assertEqual(config.seed, 12345)
        
        # 0 시드
        config = SimConfig(seed=0)
        self.assertEqual(config.seed, 0)


def run_smoke_tests():
    """스모크 테스트 실행"""
    print("=== 스모크 테스트 시작 ===")
    
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 클래스 추가
    suite.addTests(loader.loadTestsFromTestCase(TestSmokeTests))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigEdgeCases))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\\n=== 테스트 결과 ===")
    print(f"실행된 테스트: {result.testsRun}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    
    if result.failures:
        print("\\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\\n에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\\n전체 결과: {'성공' if success else '실패'}")
    
    return success


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
