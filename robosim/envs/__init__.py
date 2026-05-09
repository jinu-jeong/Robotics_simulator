"""RoboSim Gymnasium environments.

Importing this module registers the built-in tasks with ``gymnasium``::

    import gymnasium as gym
    import robosim.envs              # noqa: F401  (registers ids)

    env = gym.make("RoboSim/Reach-v0")

Each task lives in :mod:`robosim.envs.tasks` and subclasses
:class:`robosim.envs.base.RoboSimEnv`.
"""
from __future__ import annotations

from robosim.envs.base import RoboSimEnv

try:
    from gymnasium.envs.registration import register

    register(
        id="RoboSim/Reach-v0",
        entry_point="robosim.envs.tasks.reach:ReachEnv",
        max_episode_steps=200,
    )
except ImportError:
    pass

__all__ = ["RoboSimEnv"]
