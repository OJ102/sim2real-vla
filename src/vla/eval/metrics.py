"""Eval metrics: Wilson CI, success normalised by the expert ceiling, smoothness and safety statistics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DT = 0.1  # control period (s)
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # Panda joint velocity limits (rad/s)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def episode_metrics(actions: np.ndarray, infos: list[dict]) -> dict:
    """actions [T,4] executed, infos: per-step info dicts (ee_pos, qpos, qvel, contacts, success)."""
    ee = np.stack([i["ee_pos"] for i in infos])
    q = np.stack([i["qpos"] for i in infos])
    qd = np.stack([i["qvel"] for i in infos])
    if len(ee) >= 4:
        jerk = np.diff(ee, n=3, axis=0) / DT**3
        jerk_rms = float(np.sqrt((jerk**2).sum(-1).mean()))
    else:
        jerk_rms = float("nan")
    a = np.asarray(actions)
    sd = float(np.abs(a[2:] - 2 * a[1:-1] + a[:-2]).mean()) if len(a) >= 3 else float("nan")
    margin = 0.05 * (JOINT_HI - JOINT_LO)
    near = ((q < JOINT_LO + margin) | (q > JOINT_HI - margin)).any(-1)
    vel = (np.abs(qd) > 0.8 * QD_MAX).any(-1)
    col = np.array([(i["contacts"]["arm_table"] + i["contacts"]["arm_distractor"]) > 0 for i in infos])
    return {"success": bool(infos[-1]["success"]), "steps": len(infos), "ee_jerk_rms": jerk_rms,
            "action_2nd_diff": sd, "joint_limit_viol": int(near.sum()), "vel_viol": int(vel.sum()),
            "collision_viol": int(col.sum())}


def aggregate(eps: list[dict], ceiling: float | None = None) -> dict:
    n = len(eps)
    k = sum(e["success"] for e in eps)
    m = lambda key: float(np.nanmean([e[key] for e in eps])) if n else float("nan")  # noqa: E731
    out = {"n": n, "successes": k, "success": k / max(n, 1), "ci95": wilson(k, n), "mean_steps": m("steps"),
           "ee_jerk_rms": m("ee_jerk_rms"), "action_2nd_diff": m("action_2nd_diff"),
           "joint_limit_viol_per_ep": m("joint_limit_viol"), "vel_viol_per_ep": m("vel_viol"),
           "collision_viol_per_ep": m("collision_viol")}
    if ceiling:
        out["expert_ceiling"] = ceiling
        out["success_vs_ceiling"] = out["success"] / ceiling
    return out


def load_ceiling(path: str = "results/expert_ceiling.json") -> dict:
    p = Path(path)
    return json.load(open(p))["ceiling"] if p.exists() else {}
