"""Minimal subprocess vector env (shared by eval and RL rollout workers).

Each worker process owns TabletopEnvs (cached per (task, split)), sets MUJOCO_EGL_DEVICE_ID round-robin from
VLA_EGL_DEVICES and asserts an NVIDIA GL renderer once. Finished envs are simply not stepped (no autoreset).
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


def _worker(conn, wid: int, require_nvidia: bool, env_kwargs: dict):
    from vla.sim.egl import assert_nvidia_renderer
    from vla.sim.env import TabletopEnv, set_egl_device
    from vla.sim.expert import ScriptedExpert

    set_egl_device(wid)
    envs, checked = {}, False

    def get(task, split):
        nonlocal checked
        if (task, split) not in envs:
            envs[(task, split)] = TabletopEnv(task=task, split=split, **env_kwargs)
            if not checked and require_nvidia:
                assert_nvidia_renderer(envs[(task, split)])
                checked = True
        return envs[(task, split)]

    cur = None
    while True:
        cmd, *a = conn.recv()
        try:
            if cmd == "reset":
                task, split, seed, target = a
                cur = get(task, split)
                obs, info = cur.reset(seed=seed, options=None if target is None else {"target_idx": target})
                conn.send((obs, _slim(info)))
            elif cmd == "step":
                obs, r, term, trunc, info = cur.step(a[0])
                conn.send((obs, r, term, trunc, _slim(info)))
            elif cmd == "expert_episode":  # run the scripted expert end-to-end, return the trajectory (no images)
                task, split, seed, noise = a
                env = get(task, split)
                env.reset(seed=seed)
                ex = ScriptedExpert(task, np.random.default_rng(10_000 + seed), noise)
                traj, done = [], False
                while not done:
                    act = ex.act(env)
                    _, _, term, trunc, info = env.step(act)
                    traj.append((act, _slim(info)))
                    done = term or trunc
                conn.send(traj)
            elif cmd == "close":
                for e in envs.values():
                    e.close()
                conn.send(None)
                return
        except Exception as e:  # noqa: BLE001
            conn.send(RuntimeError(f"worker {wid}: {type(e).__name__}: {e}"))


def _slim(info: dict) -> dict:
    keep = ("success", "qpos", "qvel", "ee_pos", "contacts", "instruction", "step", "task")
    return {k: info[k] for k in keep if k in info}


class SubprocVecEnv:
    def __init__(self, n: int, require_nvidia: bool | None = None, env_kwargs: dict | None = None):
        import sys

        self.n = n
        require = sys.platform.startswith("linux") if require_nvidia is None else require_nvidia
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, i, require, env_kwargs or {}), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def _recv(self, i):
        r = self.conns[i].recv()
        if isinstance(r, Exception):
            raise r
        return r

    def reset(self, idx: list[int], task: str, split: str, seeds: list[int], targets: list | None = None):
        for k, i in enumerate(idx):
            self.conns[i].send(("reset", task, split, int(seeds[k]), None if targets is None else targets[k]))
        return [self._recv(i) for i in idx]  # [(obs, info)]

    def step(self, idx: list[int], actions: np.ndarray):
        for k, i in enumerate(idx):
            self.conns[i].send(("step", actions[k]))
        return [self._recv(i) for i in idx]  # [(obs, r, term, trunc, info)]

    def expert_episodes(self, idx: list[int], task: str, split: str, seeds: list[int], noise: float = 0.05):
        for k, i in enumerate(idx):
            self.conns[i].send(("expert_episode", task, split, int(seeds[k]), noise))
        return [self._recv(i) for i in idx]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close",))
                c.recv()
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=5)
