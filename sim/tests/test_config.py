"""
SimConfig 테스트
"""
import platform
from ..core.config import SimConfig

# pytest가 없을 때를 위한 간단한 대체
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def raises(exception, match=None):
            class RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception.__name__} but no exception was raised")
                    if not issubclass(exc_type, exception):
                        return False
                    if match and match not in str(exc_val):
                        raise AssertionError(f"Expected message containing '{match}', got '{exc_val}'")
                    return True
            return RaisesContext()
    pytest = MockPytest()


def test_default_config():
    """기본 설정 테스트"""
    config = SimConfig()
    
    assert config.dt == 1.0 / 60.0
    assert config.substeps == 1
    assert config.gravity == -9.81
    assert config.window_width == 1024
    assert config.window_height == 768
    assert config.vsync is True
    assert config.seed is None


def test_auto_arch_selection():
    """자동 아키텍처 선택 테스트"""
    config = SimConfig(arch="auto")
    
    # 플랫폼에 따른 기본 아키텍처 확인
    system = platform.system().lower()
    if system == "darwin":
        assert config.arch == "metal"
    elif system in ["linux", "windows"]:
        assert config.arch == "vulkan"
    else:
        assert config.arch == "cpu"


def test_config_validation():
    """설정 검증 테스트"""
    # 유효한 설정
    config = SimConfig(dt=0.01, substeps=5, arch="cpu")
    assert config.dt == 0.01
    assert config.substeps == 5
    assert config.arch == "cpu"
    
    # 잘못된 dt
    with pytest.raises(ValueError, match="dt는 양수여야 합니다"):
        SimConfig(dt=-0.01)
    
    with pytest.raises(ValueError, match="dt는 양수여야 합니다"):
        SimConfig(dt=0.0)
    
    # 잘못된 substeps
    with pytest.raises(ValueError, match="substeps는 1 이상이어야 합니다"):
        SimConfig(substeps=0)
    
    with pytest.raises(ValueError, match="substeps는 1 이상이어야 합니다"):
        SimConfig(substeps=-1)
    
    # 잘못된 윈도우 크기
    with pytest.raises(ValueError, match="윈도우 크기는 양수여야 합니다"):
        SimConfig(window_width=0)
    
    with pytest.raises(ValueError, match="윈도우 크기는 양수여야 합니다"):
        SimConfig(window_height=-100)
    
    # 잘못된 아키텍처
    with pytest.raises(ValueError, match="지원하지 않는 아키텍처"):
        SimConfig(arch="invalid")


def test_effective_dt():
    """유효 dt 계산 테스트"""
    config = SimConfig(dt=0.1, substeps=5)
    assert config.effective_dt == 0.02  # 0.1 / 5


def test_target_fps():
    """목표 FPS 계산 테스트"""
    config = SimConfig(dt=1.0/30.0)  # 30 FPS
    assert config.target_fps == 30.0


def test_to_dict():
    """딕셔너리 변환 테스트"""
    config = SimConfig(
        dt=0.02,
        substeps=2,
        arch="cpu",
        window_width=800,
        window_height=600,
        seed=42
    )
    
    config_dict = config.to_dict()
    
    expected = {
        "dt": 0.02,
        "substeps": 2,
        "gravity": -9.81,
        "window_width": 800,
        "window_height": 600,
        "vsync": True,
        "arch": "cpu",
        "seed": 42,
    }
    
    assert config_dict == expected


def test_str_representation():
    """문자열 표현 테스트"""
    config = SimConfig(dt=0.02, substeps=3, arch="metal")
    config_str = str(config)
    
    assert "dt=0.02" in config_str
    assert "substeps=3" in config_str
    assert "arch=metal" in config_str
    assert "window=1024x768" in config_str


if __name__ == "__main__":
    # 간단한 스모크 테스트
    print("SimConfig 스모크 테스트 실행...")
    
    try:
        # 기본 설정 테스트
        config = SimConfig()
        print(f"✓ 기본 설정 생성 성공: {config}")
        
        # 아키텍처 매핑 테스트
        for arch in ["cpu", "metal", "vulkan", "cuda"]:
            try:
                test_config = SimConfig(arch=arch)
                print(f"✓ {arch} 아키텍처 설정 성공")
            except ValueError as e:
                print(f"✗ {arch} 아키텍처 설정 실패: {e}")
        
        # 검증 테스트
        try:
            SimConfig(dt=-1.0)
            print("✗ 음수 dt 검증 실패")
        except ValueError:
            print("✓ 음수 dt 검증 성공")
        
        try:
            SimConfig(substeps=0)
            print("✗ 0 substeps 검증 실패")
        except ValueError:
            print("✓ 0 substeps 검증 성공")
        
        print("SimConfig 스모크 테스트 완료!")
        
    except Exception as e:
        print(f"✗ 스모크 테스트 실패: {e}")
        exit(1)
