"""Auto-build helper for the C++ extension.

If ``from robosim import _cpp`` succeeds (the wheel already shipped a
compiled module), this is a no-op. Otherwise we attempt a one-time
in-place ``pip install -e .`` against the current interpreter, then
re-import.

Triggered explicitly by ``robosim.ensure_cpp_built()`` (manual) or
implicitly by the dispatch bridge on first hot-path call (auto). Set
``ROBOSIM_NO_AUTO_BUILD=1`` to disable the implicit path — the
solvers will silently fall back to pure NumPy.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


_BUILD_TRIED = False
_BUILD_RESULT: bool | None = None


def _project_root() -> Path | None:
    """Locate the repo root by walking up from this file until we hit
    ``pyproject.toml`` with ``[tool.scikit-build]``. Returns None if
    we're running from an installed wheel rather than an editable
    source checkout."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        pp = parent / "pyproject.toml"
        if pp.is_file() and "scikit-build" in pp.read_text(errors="ignore"):
            return parent
    return None


def _path_with_conda_bin() -> dict[str, str]:
    """Augment PATH so the subprocess can find the ``cmake`` shipped
    with the active Python's conda env. ``pip install`` doesn't pass
    PATH through reliably across all shells, so we add the dir
    explicitly."""
    env = os.environ.copy()
    py_bin = Path(sys.executable).parent
    env["PATH"] = f"{py_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _can_attempt_build(verbose: bool) -> tuple[bool, str]:
    """Sanity check that an in-place build has any chance of working.

    Returns (ok, reason). ``reason`` is human-readable and shown to
    the user when ``ok`` is False."""
    root = _project_root()
    if root is None:
        return False, (
            "no editable source checkout (this looks like a wheel install — "
            "rebuild the wheel locally with `pip install --no-build-isolation`)."
        )
    env_path = _path_with_conda_bin()["PATH"]
    if shutil.which("cmake", path=env_path) is None:
        return False, (
            "cmake not on PATH. Install via "
            "`conda install -c conda-forge cmake pybind11 eigen scikit-build-core` "
            "and retry."
        )
    return True, ""


def ensure_cpp_built(verbose: bool = True) -> bool:
    """Make ``robosim._cpp`` importable. Returns True on success.

    The first call performs the import attempt and (if it fails) one
    in-place build attempt; the result is cached so subsequent calls
    are cheap. Set ``ROBOSIM_NO_AUTO_BUILD=1`` to skip the build
    attempt entirely (just probes the import).
    """
    global _BUILD_TRIED, _BUILD_RESULT
    if _BUILD_TRIED:
        return bool(_BUILD_RESULT)
    _BUILD_TRIED = True

    # Fast path: already importable.
    try:
        importlib.import_module("robosim._cpp")
        _BUILD_RESULT = True
        return True
    except ImportError:
        pass

    if os.environ.get("ROBOSIM_NO_AUTO_BUILD"):
        _BUILD_RESULT = False
        if verbose:
            print("[robosim] C++ extension not built and "
                  "ROBOSIM_NO_AUTO_BUILD set — falling back to Python.")
        return False

    ok, reason = _can_attempt_build(verbose)
    if not ok:
        _BUILD_RESULT = False
        if verbose:
            print(f"[robosim] C++ extension not built: {reason}")
        return False

    root = _project_root()
    assert root is not None  # _can_attempt_build vetted this
    env = _path_with_conda_bin()

    if verbose:
        print(f"[robosim] Building C++ extension (one-time setup, "
              f"~30 s)…  (source: {root})")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(root),
             "--no-build-isolation", "--quiet"],
            env=env, capture_output=True, text=True, timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        _BUILD_RESULT = False
        if verbose:
            print(f"[robosim] Build failed to launch: {exc}. "
                  "Falling back to Python.")
        return False

    if proc.returncode != 0:
        _BUILD_RESULT = False
        if verbose:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            print(f"[robosim] Build failed (exit {proc.returncode}). "
                  f"Falling back to Python. Tail of stderr:\n{tail}")
        return False

    # Re-import after build. importlib.invalidate_caches() is needed
    # because the freshly-built ``.so`` showed up after the path
    # finder cached its absence.
    importlib.invalidate_caches()
    try:
        importlib.import_module("robosim._cpp")
        _BUILD_RESULT = True
        if verbose:
            print("[robosim] Built. C++ hot paths now active.")
        return True
    except ImportError as e:
        _BUILD_RESULT = False
        if verbose:
            print(f"[robosim] Build completed but import still failed: {e}")
        return False
