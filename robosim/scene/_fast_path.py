"""Stage-9 fast path: build a :class:`_cpp.scene_step.RigidSceneStep`
mirror of the Scene's rigid bookkeeping, then run the substep loop
entirely in C++ via :meth:`RigidSceneStep.step`.

Eligibility: scene is purely rigid (only :class:`RobotHandle` and
:class:`RigidBodyHandle` participants; no FEM/CB/MPM bodies). The
existing :class:`PenaltyContactSolver` keeps owning the
broad-phase + narrow-phase + per-link wrench logic for the slow
path so we still have a parity reference; the fast path simply
short-circuits to the C++ orchestrator when the substep tree is
trivially rigid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robosim.model.geometry import GeometryType

if TYPE_CHECKING:
    from robosim.scene.scene import Scene


def _have_cpp_scene_step() -> bool:
    try:
        from robosim import _cpp
        return hasattr(_cpp, "scene_step")
    except ImportError:
        return False


def is_rigid_only(scene: "Scene") -> bool:
    """True when every body in the scene is a :class:`RigidBodyHandle`.

    FEM / CB / Hybrid bodies still need their Python solvers in the
    inner loop, so we leave the slow path in charge for those scenes.
    """
    from robosim.scene.handles import RigidBodyHandle
    for bh in scene._body_handles.values():
        if not isinstance(bh, RigidBodyHandle):
            return False
    return True


def build_scene_step(scene: "Scene"):
    """Construct and populate a ``RigidSceneStep`` from ``scene``.

    Returns a dict bundle ``{step, robot_ids, body_ids}`` where:
      * ``step``: the C++ ``RigidSceneStep`` handle
      * ``robot_ids``: ``{name: robot_idx}`` for both robots and free bodies
      * ``body_ids``: ``{detector_bid: (robot_idx, link_idx, scene_bid)}``
        — a back-reference used when adjacency filters reference the
        Python detector's body ids.

    Raises if the C++ extension isn't built.
    """
    if not _have_cpp_scene_step():
        raise RuntimeError("robosim._cpp.scene_step not available")
    from robosim import _cpp
    from robosim.model._cpp_bridge import build_rbd_topology
    from robosim.scene.handles import RobotHandle, RigidBodyHandle

    ss = _cpp.scene_step.RigidSceneStep()
    robot_ids: dict[str, int] = {}
    # Map: each registered Python participant (Robot or Rigid free body)
    # gets a stable C++ robot_idx + topo. Topos kept alive by the bundle.
    topo_keepalive: list = []
    # We need to know the Python ``_body_map`` (bid → (robot_id_name, link_idx))
    # to relink the detector's adjacency filter pairs. Built below.
    py_bid_to_cpp: dict[int, int] = {}

    def _register_participant(name: str, model, gravity: np.ndarray,
                              is_free: bool):
        topo = build_rbd_topology(model)
        topo_keepalive.append(topo)
        rid = ss.add_robot(topo, np.ascontiguousarray(gravity, dtype=np.float64),
                           is_free)
        ss.set_state(rid,
                     np.ascontiguousarray(model.q,  dtype=np.float64),
                     np.ascontiguousarray(model.qd, dtype=np.float64))
        # PD (gravity-comp, max-torque clamping, mimic handling, …) stays
        # on the Python side — we re-compute tau every substep via
        # ``ctrl.compute(t, q, qd)`` and push it into C++ as the
        # baseline torque. That preserves the full JointPD feature set
        # without re-implementing it in C++.
        robot_ids[name] = rid

        # Register collision bodies. We mirror the order
        # ContactDetector used so adjacency-filter bids match.
        for li, link in enumerate(model.links):
            if not link.collisions:
                continue
            col = link.collisions[0]
            g = col.geometry
            R = np.ascontiguousarray(col.origin.rotation,    dtype=np.float64)
            t = np.ascontiguousarray(col.origin.translation, dtype=np.float64)
            if g.geometry_type == GeometryType.BOX:
                ss.add_box(rid, li, np.ascontiguousarray(g.size / 2.0,
                                                          dtype=np.float64),
                           R, t)
            elif g.geometry_type == GeometryType.SPHERE:
                ss.add_sphere(rid, li, float(g.radius), R, t)
            elif g.geometry_type == GeometryType.CYLINDER:
                ss.add_cylinder(rid, li, float(g.radius), float(g.length),
                                R, t)
            # MESH / unknown: skip — those geometries already have
            # python-side narrow-phase fallbacks our C++ path doesn't
            # cover, so the fast path won't be activated for them.

    # ── Register all robot handles ──
    for name, rh in scene._robot_handles.items():
        _register_participant(name, rh._model, rh._model.gravity,
                              is_free=False)

    # ── Register all rigid body handles (free bodies) ──
    for name, bh in scene._body_handles.items():
        _register_participant(name, bh._model, bh._model.gravity,
                              is_free=True)

    # ── Adjacency filters: replay them via Python detector's filter set ──
    # The Python contact solver already added parent/child + sibling
    # filters during register_rbd(). We replay them by looking up the
    # C++ robot/link via the matching detector body_id.
    contact = scene._contact
    if contact is not None:
        # detector._body_map: detector_bid → (robot_id_name, link_idx)
        # Our C++ side assigns body_ids sequentially per add_* call, in
        # the same order we registered links above. We rebuild the same
        # ordering here for the lookup.
        bid_to_cpp_bid: dict[int, int] = {}
        running_cpp_bid = 0
        for name in (*scene._robot_handles.keys(), *scene._body_handles.keys()):
            handle = scene._robot_handles.get(name) or scene._body_handles[name]
            model = handle._model
            cpp_link_bids: list[int] = []
            for li, link in enumerate(model.links):
                if link.collisions:
                    g = link.collisions[0].geometry
                    if g.geometry_type in (GeometryType.BOX, GeometryType.SPHERE,
                                            GeometryType.CYLINDER):
                        cpp_link_bids.append(running_cpp_bid)
                        running_cpp_bid += 1
            # Match against Python detector's body_map (which is keyed
            # by det_bid → (robot_id, link_idx)) in declaration order.
            for det_bid, (rid_name, link_idx) in contact._body_map.items():
                if rid_name != name:
                    continue
                # Find the matching index in cpp_link_bids for link_idx.
                # Walk model.links collecting indices that have collision.
                k = 0
                for li_check, link in enumerate(model.links):
                    if not link.collisions: continue
                    g = link.collisions[0].geometry
                    if g.geometry_type not in (GeometryType.BOX, GeometryType.SPHERE,
                                                GeometryType.CYLINDER):
                        continue
                    if li_check == link_idx:
                        bid_to_cpp_bid[det_bid] = cpp_link_bids[k]
                        break
                    k += 1

        # Translate adjacency filters.
        for (a, b) in contact.detector._filter_pairs:
            if a in bid_to_cpp_bid and b in bid_to_cpp_bid:
                ss.add_filter(bid_to_cpp_bid[a], bid_to_cpp_bid[b])

        py_bid_to_cpp = bid_to_cpp_bid

        # ── Ground + contact params ──
        ground = contact.detector.ground
        if ground is not None:
            ss.set_ground(float(ground.height),
                          np.ascontiguousarray(ground.normal, dtype=np.float64))
        p = contact.params
        ss.set_contact_params(float(p.stiffness), float(p.damping),
                              float(p.friction_mu), float(p.friction_eps),
                              float(p.max_penetration))

    return {
        "step": ss,
        "robot_ids": robot_ids,
        "topo_keepalive": topo_keepalive,
        "py_bid_to_cpp": py_bid_to_cpp,
    }


def fast_substep(scene: "Scene", bundle: dict, dt: float, t: float,
                 n_substeps: int = 1) -> None:
    """Run ``n_substeps`` substeps via the C++ orchestrator.

    Per-substep responsibilities:
      * Python computes ``tau`` from each robot's full Python
        :class:`JointPD` (gravity-comp, mimic, max-torque clamping).
      * State + tau are pushed to C++.
      * C++ runs FK → vel → contact → ABA → integrate.
      * State is pulled back for the next substep / frame.

    Free-floating bodies have no controller; their baseline tau stays
    zero (gravity is supplied to ABA via ``add_robot``'s ``gravity``).
    """
    ss = bundle["step"]
    robot_ids = bundle["robot_ids"]

    # ── Push current Python state + tau into C++ ──
    for name, rh in scene._robot_handles.items():
        rid = robot_ids[name]
        q  = np.ascontiguousarray(rh._model.q,  dtype=np.float64)
        qd = np.ascontiguousarray(rh._model.qd, dtype=np.float64)
        ss.set_state(rid, q, qd)
        ctrl = rh._controller
        if ctrl is not None:
            tau = ctrl.compute(t, rh._model.q, rh._model.qd)
        else:
            tau = np.zeros_like(q)
        ss.set_tau(rid, np.ascontiguousarray(tau, dtype=np.float64))
    for name, bh in scene._body_handles.items():
        rid = robot_ids[name]
        ss.set_state(rid,
                     np.ascontiguousarray(bh._model.q,  dtype=np.float64),
                     np.ascontiguousarray(bh._model.qd, dtype=np.float64))
        # Free bodies have no controller — leave baseline tau at zero
        # (it persists from add_robot's zero init).

    # ── Run substeps ──
    ss.step(float(dt), int(n_substeps))

    # ── Pull C++ state back into Python ──
    for name, rh in scene._robot_handles.items():
        rid = robot_ids[name]
        q, qd = ss.get_state(rid)
        rh._model.q = q
        rh._model.qd = qd
        rh._model.invalidate_fk_cache()
    for name, bh in scene._body_handles.items():
        rid = robot_ids[name]
        q, qd = ss.get_state(rid)
        bh._model.q = q
        bh._model.qd = qd
        bh._model.invalidate_fk_cache()
