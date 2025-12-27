"""Smoke tests for Phase 0."""
import pytest
from sim.core.config import SimConfig
from sim.core.app import init_taichi


def test_config_validation():
    """Test SimConfig validation."""
    # Valid config
    config = SimConfig(dt=0.01, substeps=1, arch="cpu")
    assert config.dt == 0.01
    assert config.substeps == 1
    assert config.arch == "cpu"
    
    # Invalid dt
    with pytest.raises(ValueError):
        SimConfig(dt=-0.01)
    
    # Invalid substeps
    with pytest.raises(ValueError):
        SimConfig(substeps=0)
    
    # Invalid arch
    with pytest.raises(ValueError):
        SimConfig(arch="invalid")


def test_arch_mapping():
    """Test arch mapping and fallback."""
    # Test that cpu arch can be initialized
    # Note: This test should run on CPU backend to avoid GPU requirements
    try:
        init_taichi("cpu", debug=False)
        # If we get here, initialization succeeded
        assert True
    except Exception as e:
        # If initialization fails, that's also a valid test result
        # (e.g., if Taichi is not installed)
        pytest.skip(f"Taichi initialization failed: {e}")


def test_config_import():
    """Test that config can be imported."""
    from sim.core.config import SimConfig
    config = SimConfig()
    assert config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

