"""Unknown contact location via residual search on a continuous surface.

Given a reduced field ``q̂``, each candidate contact ``c`` on the allowed face
defines a one-parameter family ``q ≈ g_r(c) λ`` with

    g_r(c) = K_r⁻¹ B_r(c) d(c),   d = −n(c)

The localizer picks

    c★, λ★ = argmin_{c, λ≥0} ‖ W (g_r(c) λ − q̂) ‖

``W`` down-weights noisy high POD modes (contact *shape* lives in the first
few bending modes). After a discrete grid search the winner is refined by a
dense barycentric sample on that triangle (and edge neighbours). Force is
then re-fit at ``c★`` with the full unweighted ``g_r``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contact.contact_mapping import SurfaceContact, contact_from_face
from ..rom.reduced_mechanics import ReducedModel


def _bary_grid(n: int) -> list[np.ndarray]:
    """Regular barycentric lattice with ``n`` points along each edge (n≥2)."""
    out = []
    for i in range(n):
        for j in range(n - i):
            k = n - 1 - i - j
            w = np.array([i, j, k], float)
            out.append(w / w.sum())
    return out


def _fit_lam(g: np.ndarray, q: np.ndarray, nonneg: bool) -> tuple[float, float]:
    """Scalar LS ``λ = (g·q)/(g·g)``. Returns ``(λ, residual)``."""
    gg = float(g @ g)
    if gg < 1e-30:
        return 0.0, float(np.linalg.norm(q))
    lam = float(g @ q) / gg
    if nonneg:
        lam = max(0.0, lam)
    res = float(np.linalg.norm(g * lam - q))
    return lam, res


@dataclass
class ContactLocalization:
    contact: SurfaceContact
    magnitude: float
    force_vector: np.ndarray
    residual_rel: float
    candidate_index: int
    extra: dict = field(default_factory=dict)


@dataclass
class UnknownContactLocalizer:
    """Precomputes ``g_r`` for a grid of surface candidates; localizes in O(C r)."""

    rom: ReducedModel
    candidates: list[SurfaceContact]
    G: np.ndarray  # (C, r) columns = g_r(c)
    min_influence: float = 1e-8
    nonneg: bool = True
    r_loc: int | None = 4
    refine: bool = True
    refine_n: int = 7
    weight: str = "leading"  # none | leading | energy
    mode_weights: np.ndarray | None = None

    @classmethod
    def from_rom(
        cls,
        rom: ReducedModel,
        face_ids: np.ndarray,
        barycentric: list | np.ndarray | None = None,
        face_stride: int = 1,
        min_influence: float = 1e-8,
        nonneg: bool = True,
        r_loc: int | None = 4,
        refine: bool = True,
        refine_n: int = 7,
        weight: str = "leading",
    ) -> "UnknownContactLocalizer":
        barys = barycentric or [[1 / 3, 1 / 3, 1 / 3]]
        barys = [np.asarray(b, float) for b in barys]
        face_ids = np.asarray(face_ids, int)[:: max(int(face_stride), 1)]
        cands: list[SurfaceContact] = []
        for fi in face_ids:
            for b in barys:
                w = b / b.sum()
                cands.append(contact_from_face(rom.mapping.mesh, int(fi), w))
        G = np.zeros((len(cands), rom.r))
        for i, c in enumerate(cands):
            d = -c.normal
            G[i] = np.linalg.solve(rom.K_r, rom.B_r_xyz(c) @ d)
        return cls(
            rom, cands, G,
            min_influence=float(min_influence), nonneg=nonneg,
            r_loc=r_loc, refine=bool(refine), refine_n=int(refine_n),
            weight=str(weight),
        )

    def __len__(self) -> int:
        return len(self.candidates)

    def _weights(self) -> np.ndarray:
        if self.mode_weights is not None:
            return np.asarray(self.mode_weights, float).reshape(-1)
        r = self.rom.r
        w = np.ones(r)
        if self.weight == "none":
            return w
        r_loc = r if self.r_loc is None else max(1, min(int(self.r_loc), r))
        if self.weight == "leading":
            w[r_loc:] = 0.0
            return w
        if self.weight == "energy":
            s = np.asarray(self.rom.basis.singular_values[:r], float)
            s = np.maximum(s, 1e-30)
            w = s / s[0]
            return w
        return w

    def _g_of(self, contact: SurfaceContact) -> np.ndarray:
        return np.linalg.solve(self.rom.K_r, self.rom.B_r_xyz(contact) @ (-contact.normal))

    def _score(self, g: np.ndarray, q: np.ndarray, wt: np.ndarray) -> tuple[float, float]:
        gw, qw = g * wt, q * wt
        lam, res = _fit_lam(gw, qw, self.nonneg)
        return lam, res

    def _neighbor_faces(self, face_index: int) -> list[int]:
        mesh = self.rom.mapping.mesh
        nodes = set(int(i) for i in mesh.surface_faces[face_index])
        out = [int(face_index)]
        seen = {int(face_index)}
        for c in self.candidates:
            fi = int(c.face_index)
            if fi in seen:
                continue
            if len(nodes.intersection(int(x) for x in c.node_indices)) >= 2:
                seen.add(fi)
                out.append(fi)
        return out

    def _refine(self, q: np.ndarray, wt: np.ndarray, face_index: int) -> tuple[SurfaceContact, np.ndarray, float, float]:
        best = (None, None, np.inf, 0.0)
        for fi in self._neighbor_faces(face_index):
            for w in _bary_grid(self.refine_n):
                c = contact_from_face(self.rom.mapping.mesh, fi, w)
                g = self._g_of(c)
                if float(g @ g) < self.min_influence:
                    continue
                lam, res = self._score(g, q, wt)
                if res < best[2]:
                    best = (c, g, res, lam)
        if best[0] is None:
            raise RuntimeError("refine found no valid contact")
        return best  # type: ignore[return-value]

    def localize(self, q: np.ndarray) -> ContactLocalization:
        q = np.asarray(q, float).reshape(-1)
        if q.shape[0] != self.rom.r:
            raise ValueError(f"q must have length r={self.rom.r}")
        wt = self._weights()
        gg = np.sum(self.G * self.G, axis=1)
        valid = gg >= self.min_influence
        if not np.any(valid):
            raise RuntimeError("no candidate with sufficient influence")
        Gw = self.G * wt[None, :]
        qw = q * wt
        ggw = np.sum(Gw * Gw, axis=1)
        lam_w = (Gw @ qw) / np.maximum(ggw, 1e-30)
        if self.nonneg:
            lam_w = np.maximum(0.0, lam_w)
        res = np.linalg.norm(Gw * lam_w[:, None] - qw[None, :], axis=1)
        res = np.where(valid, res, np.inf)
        k = int(np.argmin(res))
        c = self.candidates[k]
        g = self.G[k]
        extra = {
            "n_candidates": len(self),
            "‖g‖²": float(gg[k]),
            "weight": self.weight,
            "r_loc": None if self.r_loc is None else int(self.r_loc),
            "refined": False,
        }
        if self.refine:
            c, g, _res, _lam = self._refine(q, wt, int(c.face_index))
            extra["refined"] = True
            k = -1
        # force from the full (unweighted) influence at c★
        lam, res_full = _fit_lam(g, q, self.nonneg)
        F = float(lam) * (-c.normal)
        rrel = float(res_full / max(np.linalg.norm(q), 1e-30))
        extra["‖q‖"] = float(np.linalg.norm(q))
        extra["low_signal"] = extra["‖q‖"] < 1e-4
        return ContactLocalization(c, float(lam), F, rrel, k, extra=extra)
