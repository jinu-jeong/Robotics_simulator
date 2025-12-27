"""Taichi initialization and backend management."""
import taichi as ti
from typing import Optional

# Track the actual arch used after initialization
_current_arch: Optional[str] = None


def init_taichi(arch: str, debug: bool = False) -> str:
    """
    Initialize Taichi with the specified backend.
    
    Args:
        arch: Backend architecture (cpu, metal, vulkan, cuda)
        debug: Enable debug mode
        
    Returns:
        The actual arch that was initialized (may differ if fallback occurred)
        
    Raises:
        RuntimeError: If initialization fails and no fallback is available
    """
    global _current_arch
    
    arch_map = {
        "cpu": ti.cpu,
        "metal": ti.metal,
        "vulkan": ti.vulkan,
        "cuda": ti.cuda,
    }
    
    if arch not in arch_map:
        raise ValueError(f"Unknown arch: {arch}")
    
    backend = arch_map[arch]
    actual_arch = arch
    
    # Try to initialize with requested backend
    try:
        ti.init(arch=backend, debug=debug)
        print(f"Taichi initialized with {arch} backend")
        _current_arch = arch
    except Exception as e:
        # Fallback: metal -> cpu
        if arch == "metal":
            print(f"Warning: Failed to initialize metal backend ({e}), falling back to cpu")
            try:
                ti.init(arch=ti.cpu, debug=debug)
                print("Taichi initialized with cpu backend (fallback)")
                actual_arch = "cpu"
                _current_arch = "cpu"
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to initialize Taichi with cpu fallback: {fallback_error}")
        else:
            raise RuntimeError(f"Failed to initialize Taichi with {arch} backend: {e}")
    
    return actual_arch


def get_arch_string() -> str:
    """
    Get the current Taichi backend as a string.
    
    Returns:
        String representation of current backend, or "unknown" if not initialized
    """
    return _current_arch if _current_arch is not None else "unknown"

