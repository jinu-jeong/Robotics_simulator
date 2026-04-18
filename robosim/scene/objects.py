"""Declaration objects for Scene.add() — builder-pattern configuration.

These objects describe *what* to create (geometry, physics, controller).
They carry no simulation state; Scene.add() turns them into live handles.

Example::

    from robosim.scene.objects import Robot, Box, Ground, FEM, CB

    robot_def = (
        Robot("urdf/arm6_gripper.urdf")
        .controller(JointPD(kp=[...], kd=[...]))
        .initial_q(HOME_Q)
        .name("arm")
    )
    box_def = (
        Box(size=0.08, mass=1.0, pos=[0.6, 0, 0.04])
        .physics(FEM(young=1e5, poisson=0.45, mesh=(8, 8, 8)))
        .name("box")
    )
"""

from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Physics specs
# ══════════════════════════════════════════════════════════════════════════════

class FEM:
    """Hex8 finite-element deformable body physics.

    Parameters
    ----------
    young          : Young's modulus  [Pa]
    poisson        : Poisson's ratio  (< 0.5)
    density        : mass density     [kg/m³]
    damping        : Rayleigh mass damping coefficient
    mesh           : (nx, ny, nz) hex-element divisions
    max_newton_iters : implicit Newton iterations per step (1 = linearised)
    dt_scale       : FEM step = arm_dt × dt_scale  (>1 = coarser sub-stepping)
    """

    def __init__(
        self,
        young:            float = 1e5,
        poisson:          float = 0.45,
        density:          float = 1000.0,
        damping:          float = 0.5,
        mesh:             tuple[int, int, int] = (8, 8, 8),
        max_newton_iters: int   = 1,
        dt_scale:         int   = 5,
    ):
        self.young            = young
        self.poisson          = poisson
        self.density          = density
        self.damping          = damping
        self.mesh             = tuple(mesh)
        self.max_newton_iters = max_newton_iters
        self.dt_scale         = dt_scale       # fem_dt = arm_dt * dt_scale

    def __repr__(self) -> str:
        return (f"FEM(E={self.young:.0e}, ν={self.poisson}, "
                f"mesh={self.mesh})")


class CB:
    """Craig-Bampton reduced-order deformable body physics (fastest deformable).

    Uses a modal reduction basis to achieve real-time rates while capturing
    low-frequency deformation modes.

    Parameters
    ----------
    n_modes : number of fixed-interface normal modes to retain
    """

    def __init__(
        self,
        young:   float = 1e5,
        poisson: float = 0.45,
        density: float = 1000.0,
        damping: float = 0.5,
        mesh:    tuple[int, int, int] = (4, 4, 4),
        n_modes: int   = 10,
    ):
        self.young   = young
        self.poisson = poisson
        self.density = density
        self.damping = damping
        self.mesh    = tuple(mesh)
        self.n_modes = n_modes

    def __repr__(self) -> str:
        return (f"CB(E={self.young:.0e}, ν={self.poisson}, "
                f"mesh={self.mesh}, n_modes={self.n_modes})")


class Rigid:
    """Standard rigid-body physics (default for Box)."""
    def __repr__(self) -> str:
        return "Rigid()"


# ══════════════════════════════════════════════════════════════════════════════
# Object declarations
# ══════════════════════════════════════════════════════════════════════════════

class Robot:
    """Declare a robot loaded from a URDF file.

    Usage::

        robot = (
            Robot("urdf/arm6_gripper.urdf")
            .controller(JointPD(kp=[...], kd=[...]))
            .initial_q(HOME_Q)
            .name("arm")
        )
    """

    def __init__(self, urdf_path: str):
        self._urdf_path  = urdf_path
        self._name:      str | None       = None
        self._controller                  = None   # JointPD or Controller
        self._initial_q: np.ndarray | None = None

    def name(self, n: str) -> "Robot":
        self._name = n
        return self

    def controller(self, ctrl) -> "Robot":
        self._controller = ctrl
        return self

    def initial_q(self, q) -> "Robot":
        self._initial_q = np.asarray(q, dtype=float)
        return self

    def __repr__(self) -> str:
        label = self._name or self._urdf_path
        ctrl  = f"  ctrl={self._controller!r}" if self._controller else ""
        return f"Robot('{label}'){ctrl}"


class Box:
    """Declare a box-shaped body.

    Parameters
    ----------
    size    : edge length (scalar → cube) or (lx, ly, lz) in metres
    mass    : total mass  [kg]
    pos     : world-space centre position (x, y, z)
    color   : RGBA display colour [0, 1]

    Chain ``.physics(FEM(...))`` or ``.physics(CB(...))`` for deformable modes.

    Usage::

        box = (
            Box(size=0.08, mass=1.0, pos=[0.6, 0.0, 0.04])
            .physics(FEM(young=1e5, poisson=0.45))
            .name("target_box")
        )
    """

    def __init__(
        self,
        size:  float | tuple | list = 0.08,
        mass:  float = 1.0,
        pos:   list | np.ndarray | None = None,
        color: list | np.ndarray | None = None,
    ):
        if isinstance(size, (int, float)):
            self._size = np.array([float(size)] * 3)
        else:
            self._size = np.asarray(size, dtype=float)

        self._mass    = float(mass)
        self._pos     = np.asarray(pos  if pos  is not None else [0.0, 0.0, 0.0],
                                   dtype=float)
        self._color   = np.asarray(color if color is not None else [0.2, 0.6, 0.9, 1.0],
                                   dtype=float)
        self._name:    str | None = None
        self._physics: FEM | CB | Rigid = Rigid()

    def physics(self, spec: "FEM | CB | Rigid") -> "Box":
        """Attach a physics spec to this box declaration."""
        self._physics = spec
        return self

    def name(self, n: str) -> "Box":
        self._name = n
        return self

    def __repr__(self) -> str:
        label = self._name or "box"
        sz    = self._size * 1e3
        return (f"Box('{label}', {sz[0]:.0f}×{sz[1]:.0f}×{sz[2]:.0f} mm, "
                f"mass={self._mass} kg, {self._physics!r})")


class Ground:
    """Declare a ground plane.

    Parameters
    ----------
    height : Z-coordinate of the ground surface [m]
    """

    def __init__(self, height: float = 0.0):
        self.height = float(height)
        self._name  = "ground"

    def name(self, n: str) -> "Ground":
        self._name = n
        return self

    def __repr__(self) -> str:
        return f"Ground(z={self.height})"
