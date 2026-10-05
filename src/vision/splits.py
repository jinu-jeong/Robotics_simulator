"""Train / test splits for the vision dataset (generalization Tests A/B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import VisionDataset


@dataclass
class VisionSplit:
    """Boolean masks over vision samples."""

    train: np.ndarray
    test: np.ndarray
    contact_id: np.ndarray  # (S,) contact index of the underlying FEM sample
    force_id: np.ndarray
    meta: dict

    def __post_init__(self) -> None:
        self.train = np.asarray(self.train, dtype=bool).reshape(-1)
        self.test = np.asarray(self.test, dtype=bool).reshape(-1)
        if self.train.shape != self.test.shape:
            raise ValueError("train/test mask length mismatch")
        if np.any(self.train & self.test):
            raise ValueError("train and test overlap")


def contact_ids_from_positions(positions: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Stable integer id per unique contact location (rounded)."""
    _, inv = np.unique(np.round(np.asarray(positions, float), decimals=decimals), axis=0, return_inverse=True)
    return inv.astype(np.int64)


def force_ids_from_magnitudes(magnitudes: np.ndarray, decimals: int = 6) -> np.ndarray:
    levels = np.unique(np.round(np.asarray(magnitudes, float), decimals=decimals))
    rounded = np.round(np.asarray(magnitudes, float), decimals=decimals)
    return np.searchsorted(levels, rounded).astype(np.int64)


def make_vision_split(
    vds: VisionDataset,
    fem_contact_position: np.ndarray,
    hold_every: int = 3,
    hold_force_above: float | None = None,
) -> VisionSplit:
    """Contact-wise hold-out (Test B), optionally also drop large forces from train (Test A).

    Every ``hold_every``-th unique contact of the *FEM* dataset is held out; all
    vision views of those contacts go to the test set. Training never sees those
    contact locations. If ``hold_force_above`` is set, train also excludes samples
    with ``force_magnitude > hold_force_above`` (those stay in the evaluation pool
    only when their contact is otherwise in train – they are moved to test).
    """
    cid = contact_ids_from_positions(np.asarray(fem_contact_position)[vds.fem_index])
    fid = force_ids_from_magnitudes(vds.force_magnitude)
    n_c = int(cid.max()) + 1
    held_contacts = set(range(0, n_c, max(int(hold_every), 1)))
    test = np.array([int(c) in held_contacts for c in cid], dtype=bool)
    train = ~test
    if hold_force_above is not None:
        big = vds.force_magnitude > float(hold_force_above)
        # move large-force train samples to test (still evaluate them)
        train = train & ~big
        test = test | big
    return VisionSplit(
        train=train,
        test=test,
        contact_id=cid,
        force_id=fid,
        meta={
            "hold_every": int(hold_every),
            "hold_force_above": hold_force_above,
            "n_contacts": n_c,
            "n_held_contacts": len(held_contacts),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
        },
    )
