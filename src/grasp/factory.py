"""Build a ``GraspSim`` from ``configs/grasp.yaml``."""

from __future__ import annotations

from typing import Any, Mapping

from ..fem.finger_model import FingerFEMModel
from .arm_sim import ArmGraspSim, build_arm_grasp_sim as _attach_arm
from .estimator import GraspEstimator
from .mechanics import GraspMechanics
from .sim import GraspSim


def build_grasp_sim(
    cfg: Mapping[str, Any],
    estimator_mode: str | None = None,
    prepare_vision: bool = False,
) -> GraspSim:
    model = FingerFEMModel.from_config(cfg["fem"])
    obj = cfg["object"]
    contact = cfg.get("contact", {})
    mech = GraspMechanics.build(
        model,
        contact.get("position_rel", [0.85, 1.0, 0.5]),
        obj["size"],
        float(obj["mass"]),
        float(obj["friction"]),
        object_stiffness=obj.get("stiffness"),
    )
    est_cfg = dict(cfg.get("estimator", {}))
    est_cfg["_fem_cfg"] = dict(cfg["fem"])  # lets the POD basis use a reference material
    if estimator_mode is not None:
        est_cfg["mode"] = estimator_mode
    if prepare_vision:
        est_cfg["prepare_vision"] = True
    estimator = GraspEstimator.build(mech, est_cfg, seed=int(cfg.get("seed", 0)))
    grip, ctrl = cfg["gripper"], cfg["control"]
    return GraspSim(
        mech=mech,
        estimator=estimator,
        opening_start=float(grip["opening_start"]),
        opening_min=float(grip["opening_min"]),
        close_speed=float(grip["close_speed"]),
        lift_height=float(grip["lift_height"]),
        lift_speed=float(grip["lift_speed"]),
        force_target=float(ctrl["force_target"]),
        force_tol=float(ctrl["force_tol"]),
        kp=float(ctrl["kp"]),
        ki=float(ctrl["ki"]),
        settle_s=float(ctrl["settle_s"]),
        dt=float(ctrl["dt"]),
        post_contact_hold_s=float(ctrl.get("post_contact_hold_s", 0.0)),
        unlock_squeeze_speed=float(grip.get("unlock_squeeze_speed", grip["close_speed"])),
        preload_force=float(ctrl.get("preload_force", 1.2)),
    )


def build_arm_grasp_sim(cfg: Mapping[str, Any], estimator_mode: str | None = None) -> ArmGraspSim:
    """Stages A/B/D jaw sim mounted on the Stage C 7-DoF arm."""
    return _attach_arm(
        build_grasp_sim(cfg, estimator_mode=estimator_mode, prepare_vision=True),
        cfg.get("arm", {}),
    )
