"""Damped-least-squares IK: target EE position + fixed (gripper-down) orientation -> 7 joint targets."""
from __future__ import annotations

import mujoco
import numpy as np


class DLSIK:
    def __init__(self, model: mujoco.MjModel, site: str = "grasp", damping: float = 0.05, iters: int = 20):
        self.m = model
        self.d = mujoco.MjData(model)  # scratch
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.damping, self.iters = damping, iters
        self.lo, self.hi = model.jnt_range[:7, 0].copy(), model.jnt_range[:7, 1].copy()
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def solve(self, q0: np.ndarray, target_pos: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
        d, m = self.d, self.m
        q = q0[:7].copy()
        for _ in range(self.iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self.site]
            mat = d.site_xmat[self.site].reshape(3, 3)
            e_pos = target_pos - pos
            # orientation error as rotation vector of R_t R^T
            rel = target_mat @ mat.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rel.ravel())
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            err = np.concatenate([e_pos, 0.5 * e_rot])
            if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
            J = np.vstack([self.jp[:, :7], self.jr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), err)
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q
