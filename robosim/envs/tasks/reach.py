"""ReachEnv — 3-DOF arm reaches a randomly sampled Cartesian target.

State-based observation, joint-position action space (additive delta around
the current PD target), dense negative-distance reward.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces

from robosim import Scene
from robosim.scene.objects import Robot
from robosim.control.pd import JointPD
from robosim.envs.base import RoboSimEnv


URDF_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples" / "pure_simulation" / "urdf" / "simple_arm.urdf"
)

# Per-joint limits read from simple_arm.urdf (3 revolute joints).
JOINT_LO = np.array([-3.14, -2.35, -1.57])
JOINT_HI = np.array([+3.14, +2.35, +1.57])
HOME_Q   = np.array([0.0, -1.0, 0.5])

# Action: small delta on the PD target [rad] per control step.
ACTION_SCALE = 0.05

# Workspace box for sampled targets [m].
TARGET_LO = np.array([0.20, -0.30, 0.10])
TARGET_HI = np.array([0.50, +0.30, 0.50])

EE_LINK = "hand"


class ReachEnv(RoboSimEnv):
    """3-DOF arm reaches a sampled point in Cartesian workspace."""

    def __init__(self, **kwargs):
        self._target = np.zeros(3)
        super().__init__(max_episode_steps=kwargs.pop("max_episode_steps", 200), **kwargs)

    # ── scene / spaces ────────────────────────────────────────────────────────

    def _build_scene(self) -> Scene:
        scene = Scene(dt=0.005, substeps=4)
        ctrl  = JointPD(kp=80.0, kd=8.0, max_torque=50.0, gravity_comp=True)
        ctrl.target_q = HOME_Q.copy()

        robot_def = (
            Robot(str(URDF_PATH))
            .name("arm")
            .controller(ctrl)
            .initial_q(HOME_Q)
        )
        self._robot = scene.add(robot_def)
        return scene

    def _observation_space(self) -> spaces.Box:
        # [q (3), dq (3), ee_pos (3), target (3), ee→target vector (3)] = 15
        high = np.full(15, np.inf, dtype=np.float32)
        return spaces.Box(low=-high, high=high, dtype=np.float32)

    def _action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    # ── episode mechanics ────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        self._target = self._np_random.uniform(TARGET_LO, TARGET_HI)
        self._robot._controller.target_q = HOME_Q.copy()

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._robot._controller
        new_target = ctrl.target_q + ACTION_SCALE * action
        ctrl.target_q = np.clip(new_target, JOINT_LO, JOINT_HI)

    def _ee_pos(self) -> np.ndarray:
        return self._robot.fk(EE_LINK).translation.copy()

    def _observation(self) -> np.ndarray:
        q   = self._robot.q
        dq  = self._robot.dq
        ee  = self._ee_pos()
        return np.concatenate([q, dq, ee, self._target, self._target - ee]).astype(np.float32)

    def _reward_terminated(
        self, obs: np.ndarray, action: np.ndarray,
    ) -> tuple[float, bool]:
        ee   = self._ee_pos()
        dist = float(np.linalg.norm(self._target - ee))
        reward = -dist - 1e-3 * float(np.sum(action ** 2))
        terminated = dist < 0.03  # 3 cm success ball
        if terminated:
            reward += 10.0
        return reward, terminated
