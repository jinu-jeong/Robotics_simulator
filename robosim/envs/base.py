"""RoboSimEnv — base ``gymnasium.Env`` wrapper around a Scene.

A subclass implements four hooks:

* ``_build_scene()``  → construct and return a :class:`robosim.Scene`
                         (called once per ``__init__``)
* ``_observation_space()`` / ``_action_space()`` → :class:`gymnasium.spaces.Space`
* ``_apply_action(action)`` → set controller targets / external forces
* ``_observation()`` → numpy array (shape must match ``_observation_space()``)
* ``_reward_terminated(obs, action)`` → ``(reward, terminated)``

All other Gym mechanics (``reset``, ``step``, ``seed``, episode-step counter)
are handled by this base class.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "robosim.envs requires `gymnasium`. Install it via "
        "`pip install robosim[rl]` or `pip install gymnasium`."
    ) from exc


class RoboSimEnv(gym.Env):
    """Base class for RoboSim Gymnasium environments.

    Parameters
    ----------
    max_episode_steps : episode horizon (control steps, not physics substeps)
    seed              : RNG seed; can also be passed to ``reset(seed=...)``
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self._max_episode_steps = int(max_episode_steps)
        self._np_random: np.random.Generator
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0

        self.scene = self._build_scene()
        self.scene.reset_runtime()

        self.observation_space = self._observation_space()
        self.action_space      = self._action_space()

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _build_scene(self):
        raise NotImplementedError

    def _observation_space(self) -> spaces.Space:
        raise NotImplementedError

    def _action_space(self) -> spaces.Space:
        raise NotImplementedError

    def _apply_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def _observation(self) -> np.ndarray:
        raise NotImplementedError

    def _reward_terminated(
        self, obs: np.ndarray, action: np.ndarray,
    ) -> tuple[float, bool]:
        raise NotImplementedError

    def _on_reset(self) -> None:
        """Optional hook: randomise targets, perturb initial state, etc."""

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.scene.reset_runtime()
        self._step_count = 0
        self._on_reset()
        return self._observation(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        self._apply_action(action)
        self.scene.step()
        self._step_count += 1

        obs = self._observation()
        reward, terminated = self._reward_terminated(obs, action)
        truncated = self._step_count >= self._max_episode_steps
        return obs, float(reward), bool(terminated), bool(truncated), {}

    def render(self):  # noqa: D401  (no-op for headless v1)
        return None

    def close(self):
        return None
