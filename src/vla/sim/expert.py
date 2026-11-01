"""Scripted waypoint expert (uses privileged state) with small action noise.

    uv run python -m vla.sim.expert --episodes 100 --split train
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .env import ACT_SCALE, TASKS, TabletopEnv

HOVER_Z = 0.12
GRASP_Z = 0.025
LIFT_Z = 0.20
PUSH_Z = 0.032  # keep pads off the table so friction does not stall the push


class ScriptedExpert:
    def __init__(self, task: str, rng: np.random.Generator | None = None, noise: float = 0.05):
        self.task, self.rng, self.noise = task, rng or np.random.default_rng(0), noise
        self.phase, self.count = 0, 0

    def _move(self, ee, target, gain=1.0):
        return np.clip((target - ee) / ACT_SCALE * gain, -1, 1)

    def act(self, env: TabletopEnv) -> np.ndarray:
        s = env.privileged()
        ee, cube = s["ee_pos"], s["cube_pos"][s["target_idx"]]
        xy_err = np.linalg.norm(ee[:2] - cube[:2])
        v, g = np.zeros(3), -1.0
        if self.task == "reach":
            # hover above, then drop to just above the cube
            if self.phase == 0:
                v = self._move(ee, np.array([cube[0], cube[1], HOVER_Z]))
                if xy_err < 0.01:
                    self.phase = 1
            else:
                v = self._move(ee, np.array([cube[0], cube[1], cube[2] + 0.01]))
        elif self.task == "lift":
            if self.phase == 0:
                v = self._move(ee, np.array([cube[0], cube[1], HOVER_Z]))
                if xy_err < 0.008 and abs(ee[2] - HOVER_Z) < 0.01:
                    self.phase = 1
            if self.phase == 1:
                v = self._move(ee, np.array([cube[0], cube[1], GRASP_Z]), gain=0.6)
                if abs(ee[2] - GRASP_Z) < 0.006:
                    self.phase = 2
            if self.phase == 2:
                g = 1.0
                v = self._move(ee, np.array([cube[0], cube[1], GRASP_Z + 0.004]))
                if s["grip_width"] < 0.046:
                    self.count += 1
                if self.count >= 3:
                    self.phase = 3
            if self.phase == 3:
                g = 1.0
                v = self._move(ee, np.array([cube[0], cube[1], LIFT_Z]), gain=0.6)
        else:  # push (fingers half closed: fully closed pads can jam and destabilise the sim)
            g = 0.6
            goal = s["goal_xy"]
            d = goal - cube[:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            if self.phase == 0:
                start = cube[:2] - 0.075 * d
                v = self._move(ee, np.array([start[0], start[1], HOVER_Z]))
                if np.linalg.norm(ee[:2] - start) < 0.01 and abs(ee[2] - HOVER_Z) < 0.01:
                    self.phase = 1
            if self.phase == 1:
                start = cube[:2] - 0.075 * d
                v = self._move(ee, np.array([start[0], start[1], PUSH_Z]))
                if abs(ee[2] - PUSH_Z) < 0.006:
                    self.phase = 2
            if self.phase == 2:
                tgt = goal - 0.035 * d
                v = self._move(ee, np.array([tgt[0], tgt[1], PUSH_Z]), gain=0.35)
        v = np.clip(v + self.rng.normal(0, self.noise, 3) * (np.abs(v) > 0.05), -1, 1)
        return np.array([*v, g], dtype=np.float32)


def run_episode(env: TabletopEnv, seed: int, noise: float = 0.05) -> dict:
    env.reset(seed=seed)
    ex = ScriptedExpert(env.task, np.random.default_rng(10_000 + seed), noise)
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(ex.act(env))
        done = term or trunc
    return {"success": bool(info["success"]), "steps": info["step"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def evaluate(episodes: int, split: str, seed0: int = 0, tasks=TASKS, render_obs: bool = False) -> dict:
    out = {}
    for task in tasks:
        env = TabletopEnv(task=task, split=split, render_obs=render_obs)
        res = [run_episode(env, seed0 + i) for i in range(episodes)]
        k = int(sum(r["success"] for r in res))
        out[task] = {"successes": k, "ci95": wilson(k, episodes), "split": split, "seed0": seed0,
                     "success": float(np.mean([r["success"] for r in res])),
                     "mean_steps": float(np.mean([r["steps"] for r in res])), "episodes": episodes}
        print(f"{task}: success={out[task]['success']:.3f} mean_steps={out[task]['mean_steps']:.1f} "
              f"(n={episodes}, split={split})", flush=True)
        env.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--split", default="train", help="train|heldout|heldout_photo|all|comma list")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--render", action="store_true", help="render images too (slower; needed to test EGL)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    splits = ["train", "heldout", "heldout_photo"] if a.split == "all" else a.split.split(",")
    res = {sp: evaluate(a.episodes, sp, a.seed0, a.tasks.split(","), a.render) for sp in splits}
    if a.out:
        json.dump({"episodes": a.episodes, "seed0": a.seed0, "noise": 0.05, "ceiling": res}, open(a.out, "w"), indent=1)
