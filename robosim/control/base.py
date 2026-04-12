"""Controller base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Abstract base for joint-space controllers."""

    @abstractmethod
    def compute(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Compute joint torques.

        Parameters
        ----------
        t : current simulation time
        q : (n_dof,) joint positions
        qd : (n_dof,) joint velocities

        Returns
        -------
        tau : (n_dof,) joint torques
        """

    def reset(self) -> None:
        """Reset internal state (e.g. integral terms)."""
