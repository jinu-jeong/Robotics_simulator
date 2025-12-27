"""
TaichiApp 테스트 (UI 윈도우 없이)
"""
from unittest.mock import patch, MagicMock
from ..core.config import SimConfig
from ..core.app import TaichiApp, init_taichi, get_arch_info

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


def test_config_creation():
    """설정 생성 테스트"""
    config = SimConfig(arch="cpu", dt=0.02, substeps=2)
    app = TaichiApp(config)
    
    assert app.config == config
    assert app.initialized is False


@patch('sim.core.app.ti')
def test_taichi_init_success(mock_ti):
    """Taichi 초기화 성공 테스트"""
    # Mock 설정
    mock_ti.cpu = "cpu_backend"
    mock_ti.init = MagicMock()
    
    config = SimConfig(arch="cpu", seed=42)
    
    # 초기화 테스트
    result = init_taichi(config)
    
    assert result is True
    mock_ti.init.assert_called_once_with(
        arch="cpu_backend",
        random_seed=42,
        debug=False,
        log_level=mock_ti.INFO
    )


@patch('sim.core.app.ti')
def test_taichi_init_with_fallback(mock_ti):
    """Taichi 초기화 폴백 테스트"""
    # Mock 설정
    mock_ti.metal = "metal_backend"
    mock_ti.cpu = "cpu_backend"
    mock_ti.reset = MagicMock()
    
    # 첫 번째 시도는 실패, 두 번째 시도는 성공
    mock_ti.init = MagicMock(side_effect=[Exception("Metal failed"), None])
    
    config = SimConfig(arch="metal", seed=None)
    
    # 초기화 테스트
    result = init_taichi(config)
    
    assert result is True
    assert config.arch == "cpu"  # 폴백으로 변경됨
    assert mock_ti.init.call_count == 2
    mock_ti.reset.assert_called_once()


@patch('sim.core.app.ti')
def test_taichi_init_failure(mock_ti):
    """Taichi 초기화 실패 테스트"""
    # Mock 설정
    mock_ti.cpu = "cpu_backend"
    mock_ti.reset = MagicMock()
    mock_ti.init = MagicMock(side_effect=Exception("Init failed"))
    
    config = SimConfig(arch="cpu")
    
    # 초기화 테스트
    result = init_taichi(config)
    
    assert result is False


def test_invalid_architecture():
    """잘못된 아키텍처 테스트"""
    config = SimConfig()
    config.arch = "invalid_arch"  # 검증 우회
    
    result = init_taichi(config)
    assert result is False


@patch('sim.core.app.ti')
def test_get_arch_info_success(mock_ti):
    """아키텍처 정보 조회 성공 테스트"""
    # Mock 설정
    mock_cfg = MagicMock()
    mock_cfg.arch = "cpu"
    mock_cfg.device_memory_GB = 8.0
    mock_cfg.debug = False
    
    mock_ti.lang.impl.current_cfg.return_value = mock_cfg
    
    info = get_arch_info()
    
    expected = {
        "arch": "cpu",
        "device_memory_GB": 8.0,
        "debug": False,
    }
    
    assert info == expected


@patch('sim.core.app.ti')
def test_get_arch_info_failure(mock_ti):
    """아키텍처 정보 조회 실패 테스트"""
    # Mock 설정
    mock_ti.lang.impl.current_cfg.side_effect = Exception("Not initialized")
    
    info = get_arch_info()
    
    assert "arch" in info
    assert info["arch"] == "unknown"
    assert "error" in info


@patch('sim.core.app.init_taichi')
def test_taichi_app_context_manager(mock_init):
    """TaichiApp 컨텍스트 매니저 테스트"""
    mock_init.return_value = True
    
    config = SimConfig(arch="cpu")
    
    with TaichiApp(config) as app:
        assert app.initialized is True
        mock_init.assert_called_once_with(config)
    
    # 컨텍스트 종료 후 정리 확인
    assert app.initialized is False


@patch('sim.core.app.init_taichi')
def test_taichi_app_initialization_failure(mock_init):
    """TaichiApp 초기화 실패 테스트"""
    mock_init.return_value = False
    
    config = SimConfig(arch="cpu")
    app = TaichiApp(config)
    
    result = app.initialize()
    
    assert result is False
    assert app.initialized is False


@patch('sim.core.app.init_taichi')
@patch('sim.core.app.get_arch_info')
def test_taichi_app_get_info(mock_get_arch_info, mock_init):
    """TaichiApp 정보 조회 테스트"""
    mock_init.return_value = True
    mock_get_arch_info.return_value = {"arch": "cpu", "debug": False}
    
    config = SimConfig(arch="cpu", dt=0.02)
    app = TaichiApp(config)
    app.initialize()
    
    info = app.get_info()
    
    assert info["initialized"] is True
    assert "config" in info
    assert "arch" in info
    assert info["config"]["arch"] == "cpu"
    assert info["config"]["dt"] == 0.02


if __name__ == "__main__":
    # 간단한 스모크 테스트 (UI 없이)
    print("TaichiApp 스모크 테스트 실행...")
    
    try:
        # 설정 생성 테스트
        config = SimConfig(arch="cpu")
        app = TaichiApp(config)
        print(f"✓ TaichiApp 생성 성공: {app.config.arch}")
        
        # 정보 조회 테스트 (초기화 전)
        info = app.get_info()
        assert info["initialized"] is False
        print("✓ 초기화 전 정보 조회 성공")
        
        # 아키텍처 매핑 테스트
        arch_map_test = {
            "cpu": "should_work",
            "metal": "might_work_on_macos",
            "vulkan": "might_work",
            "cuda": "might_work_with_gpu",
        }
        
        for arch, expected in arch_map_test.items():
            test_config = SimConfig(arch=arch)
            test_app = TaichiApp(test_config)
            print(f"✓ {arch} 아키텍처 앱 생성 성공")
        
        print("TaichiApp 스모크 테스트 완료!")
        
    except Exception as e:
        print(f"✗ 스모크 테스트 실패: {e}")
        exit(1)
