"""Pure success predicates (reward terms are added in Phase 6)."""
from __future__ import annotations

import numpy as np

from .assets import CUBE_HALF

REACH_TOL = 0.03
LIFT_HEIGHT = 0.08
PUSH_TOL = 0.04


def is_success(task: str, ee_pos, cube_pos, goal_xy) -> bool:
    ee_pos, cube_pos = np.asarray(ee_pos), np.asarray(cube_pos)
    if task == "reach":
        return bool(np.linalg.norm(ee_pos - cube_pos) < REACH_TOL)
    if task == "lift":
        return bool(cube_pos[2] - CUBE_HALF > LIFT_HEIGHT and np.linalg.norm(ee_pos - cube_pos) < 0.05)
    if task == "push":
        return bool(np.linalg.norm(cube_pos[:2] - np.asarray(goal_xy)) < PUSH_TOL)
    raise ValueError(task)


# ---------------------------------------------------------------- episode return (RL reward, PLAN v1 P6)
DT = 0.1
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # Panda joint velocity limits (rad/s)

DEFAULT_WEIGHTS = {"success": 10.0, "jerk_act": 0.5, "jerk_ee": 50.0, "limit": 0.05, "vel": 0.1, "collision": 0.1}


def episode_return(actions, infos, w: dict | None = None):
    """R = 10*success - l_j*sum||a_t-2a_{t-1}+a_{t-2}||^2 - l_jk*sum||d3 ee||^2 - l_lim*sum 1[q within 5% of limit]
           - l_vel*sum relu(|qd|-0.8 qd_max) - l_col*sum 1[arm-table or arm-distractor contact].
    actions [T,4] executed, infos: per-step dicts (ee_pos, qpos, qvel, contacts, success).
    Returns (R, {component: signed contribution}). Each term is reported separately."""
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    a = np.asarray(actions, dtype=np.float64)
    ee = np.stack([i["ee_pos"] for i in infos]).astype(np.float64)
    q = np.stack([i["qpos"] for i in infos])
    qd = np.stack([i["qvel"] for i in infos])
    d2a = a[2:] - 2 * a[1:-1] + a[:-2] if len(a) >= 3 else np.zeros((0, a.shape[-1]))
    d3e = np.diff(ee, n=3, axis=0) if len(ee) >= 4 else np.zeros((0, 3))
    margin = 0.05 * (JOINT_HI - JOINT_LO)
    near = ((q < JOINT_LO + margin) | (q > JOINT_HI - margin)).any(-1)
    col = np.array([(i["contacts"]["arm_table"] + i["contacts"]["arm_distractor"]) > 0 for i in infos])
    comp = {
        "success": w["success"] * float(infos[-1]["success"]),
        "jerk_act": -w["jerk_act"] * float((d2a**2).sum()),
        "jerk_ee": -w["jerk_ee"] * float((d3e**2).sum()),
        "limit": -w["limit"] * float(near.sum()),
        "vel": -w["vel"] * float(np.maximum(np.abs(qd) - 0.8 * QD_MAX, 0).sum()),
        "collision": -w["collision"] * float(col.sum()),
    }
    return float(sum(comp.values())), comp
