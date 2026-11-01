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
