"""Parallel expert-demo generation -> HDF5 shards + manifest.json + stats.json (+ subset index files).

    python -m vla.data.generate --config configs/data.yaml --num-workers 32 [key=value ...]

Shard layout (frames of all episodes in a shard are concatenated):
  images uint8[N,H,W,3] (lzf, chunked per frame), proprio f32[N,11], actions f32[N,4],
  ep_start i64[E], ep_len i64[E], seed i64[E]; string arrays per episode: task, instruction, color. attrs: split, dr_nominal.
Only successful episodes are kept. Frame t = (obs_t, expert action_t); the final success obs has no action.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from vla.utils.config import load_config

COLORS = ("red", "green", "blue")


@dataclass
class DataConfig:
    out_dir: str = ""
    tasks: list = field(default_factory=lambda: ["reach", "lift", "push"])
    colors: list = field(default_factory=lambda: list(COLORS))
    episodes_per_combo: int = 1000
    chunk_size: int = 50
    seed_base: int = 100000
    split: str = "train"
    dr_nominal: list = field(default_factory=list)
    expert_noise: float = 0.05
    image_size: int = 128
    max_attempt_factor: float = 2.0


def resolve_out_dir(cfg: DataConfig) -> Path:
    if cfg.out_dir:
        return Path(cfg.out_dir)
    base = os.environ.get("VLA_DATA")
    if not base:
        raise SystemExit("set out_dir=... or VLA_DATA in .env")
    return Path(base) / "demos"


def make_jobs(cfg: DataConfig) -> list[dict]:
    jobs, n_chunks = [], -(-cfg.episodes_per_combo // cfg.chunk_size)
    for ti, task in enumerate(cfg.tasks):
        for ci, color in enumerate(cfg.colors):
            for k in range(n_chunks):
                n = min(cfg.chunk_size, cfg.episodes_per_combo - k * cfg.chunk_size)
                # disjoint seed block per (task, colour, chunk): 100k seeds each is far more than ever attempted
                seed0 = cfg.seed_base + ((ti * len(cfg.colors) + ci) * n_chunks + k) * cfg.chunk_size * 4
                jobs.append({"task": task, "color": color, "chunk": k, "n": n, "seed0": seed0})
    return jobs


_CFG: DataConfig | None = None
_ENVS: dict = {}
_WORKER: dict = {}


def _init_worker(cfg: DataConfig, require_nvidia: bool):
    global _CFG
    _CFG = cfg
    wid = max(mp.current_process()._identity[0] - 1, 0) if mp.current_process()._identity else 0
    from vla.sim.env import set_egl_device

    set_egl_device(wid)
    _WORKER.update(id=wid, egl=os.environ.get("MUJOCO_EGL_DEVICE_ID"), checked=False, require_nvidia=require_nvidia)


def _get_env(task: str):
    from vla.sim.env import TabletopEnv

    if task not in _ENVS:
        env = TabletopEnv(task=task, split=_CFG.split, image_size=_CFG.image_size, dr_nominal=tuple(_CFG.dr_nominal))
        if not _WORKER["checked"]:  # assert once per worker: software GL would silently render on CPU
            from vla.sim.egl import assert_nvidia_renderer

            if _WORKER["require_nvidia"]:
                r = assert_nvidia_renderer(env)
                print(f"[worker {_WORKER['id']}] EGL device {_WORKER['egl']} GL_RENDERER={r}", flush=True)
            _WORKER["checked"] = True
        _ENVS[task] = env
    return _ENVS[task]


def run_job(job: dict) -> dict:
    import h5py

    from vla.sim.expert import ScriptedExpert

    cfg, env = _CFG, _get_env(job["task"])
    tidx = COLORS.index(job["color"])
    eps, attempts, seed = [], 0, job["seed0"]
    max_att = int(job["n"] * cfg.max_attempt_factor)
    t0 = time.time()
    while len(eps) < job["n"] and attempts < max_att:
        obs, info = env.reset(seed=seed, options={"target_idx": tidx})
        ex = ScriptedExpert(job["task"], np.random.default_rng(10_000_000 + seed), cfg.expert_noise)
        imgs, pro, acts, done = [obs["image"]], [obs["proprio"]], [], False
        cps = [env.privileged()["cube_pos"]]  # world positions of all 3 cubes, aligned with images[row]
        while not done:
            a = ex.act(env)
            obs, _, term, trunc, info = env.step(a)
            acts.append(a)
            done = term or trunc
            if not done:
                imgs.append(obs["image"])
                pro.append(obs["proprio"])
                cps.append(env.privileged()["cube_pos"])
        attempts += 1
        if info["success"]:
            eps.append({"images": np.stack(imgs), "proprio": np.stack(pro), "actions": np.stack(acts)[: len(imgs)],
                        "cube_pos": np.stack(cps).astype(np.float32), "seed": seed, "instruction": info["instruction"]})
        seed += 1
    out = resolve_out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{job['task']}_{job['color']}_{job['chunk']:04d}.h5"
    if eps:
        tmp = out / (name + ".tmp")
        with h5py.File(tmp, "w") as f:
            n = sum(len(e["actions"]) for e in eps)
            im = f.create_dataset("images", (n, cfg.image_size, cfg.image_size, 3), "uint8", compression="lzf",
                                  chunks=(1, cfg.image_size, cfg.image_size, 3))
            lens = np.array([len(e["actions"]) for e in eps])
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            for e, s0, ln in zip(eps, starts, lens):
                im[s0:s0 + ln] = e["images"]
            f["proprio"] = np.concatenate([e["proprio"] for e in eps]).astype(np.float32)
            f["actions"] = np.concatenate([e["actions"] for e in eps]).astype(np.float32)
            f["cube_pos"] = np.concatenate([e["cube_pos"] for e in eps]).astype(np.float32)  # [N,3,3]
            f["target_idx"] = np.full(len(eps), tidx, dtype=np.int64)
            f["ep_start"], f["ep_len"] = starts.astype(np.int64), lens.astype(np.int64)
            f["seed"] = np.array([e["seed"] for e in eps], dtype=np.int64)
            dt = h5py.string_dtype()
            f.create_dataset("task", data=np.array([job["task"]] * len(eps), dtype=object), dtype=dt)
            f.create_dataset("color", data=np.array([job["color"]] * len(eps), dtype=object), dtype=dt)
            f.create_dataset("instruction", data=np.array([e["instruction"] for e in eps], dtype=object), dtype=dt)
            f.attrs["split"], f.attrs["dr_nominal"] = cfg.split, ",".join(cfg.dr_nominal)
        os.replace(tmp, out / name)  # atomic: a shard exists only when complete
    return {"shard": name if eps else None, "task": job["task"], "color": job["color"], "episodes": len(eps),
            "attempts": attempts, "frames": int(sum(len(e["actions"]) for e in eps)), "secs": time.time() - t0}


def compute_stats(out: Path, shards: list[str]) -> dict:
    import h5py

    pro, txy, n, missing = [], [], 0, 0
    for s in shards:
        with h5py.File(out / s, "r") as f:
            pro.append(f["proprio"][:])
            n += len(pro[-1])
            if "cube_pos" in f and "target_idx" in f:
                cp, ti = f["cube_pos"][:], f["target_idx"][:]
                for st, ln, k in zip(f["ep_start"][:], f["ep_len"][:], ti):
                    txy.append(cp[st:st + ln, k, :2])
            else:
                missing += 1
    p = np.concatenate(pro)
    st = {"n_frames": n, "proprio_mean": p.mean(0).tolist(), "proprio_std": (p.std(0) + 1e-6).tolist(),
          "action_low": -1.0, "action_high": 1.0}
    if txy and not missing:
        t = np.concatenate(txy)
        st["target_xy_mean"], st["target_xy_std"] = t.mean(0).tolist(), (t.std(0) + 1e-6).tolist()
    elif missing:
        print(f"WARNING: {missing} shard(s) lack cube_pos/target_idx: target_xy stats skipped", flush=True)
    return st


def write_subsets(out: Path, shards: list[str], fracs=(10, 30, 100)) -> None:
    """subset_{pct}.json: stratified (first ceil(pct%) episodes of every shard) [[shard, ep_idx], ...]."""
    import h5py

    per = {}
    for s in shards:
        with h5py.File(out / s, "r") as f:
            per[s] = len(f["ep_len"])
    for pct in fracs:
        idx = [[s, e] for s, n in per.items() for e in range(max(1, -(-n * pct // 100)))]
        json.dump(idx, open(out / f"subset_{pct}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-require-nvidia", action="store_true", help="skip GL_RENDERER check (local macOS)")
    ap.add_argument("overrides", nargs="*", help="key=value")
    a = ap.parse_args()
    cfg = load_config(DataConfig, a.config, a.overrides)
    out = resolve_out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in make_jobs(cfg)
            if not (out / f"{j['task']}_{j['color']}_{j['chunk']:04d}.h5").exists()]  # resumable
    print(f"{len(jobs)} jobs ({len(make_jobs(cfg))} total) -> {out}, {a.num_workers} workers", flush=True)
    t0, results = time.time(), []
    if os.uname().sysname == "Linux" and not os.environ.get("VLA_EGL_DEVICES", "").strip():
        from vla.sim.egl import recommended_ids

        ids = recommended_ids()  # probe once here; spawned workers inherit the env
        if not ids:
            raise SystemExit("no A100 EGL device found; set VLA_EGL_DEVICES")
        os.environ["VLA_EGL_DEVICES"] = ",".join(map(str, ids))
        print(f"VLA_EGL_DEVICES auto-probed: {os.environ['VLA_EGL_DEVICES']}", flush=True)
    require = not a.no_require_nvidia and os.uname().sysname == "Linux"
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.num_workers, initializer=_init_worker, initargs=(cfg, require)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {sum(x['episodes'] for x in results)} eps, {time.time()-t0:.0f}s", flush=True)
    shards = sorted(p.name for p in out.glob("*.h5"))
    stats = compute_stats(out, shards)
    json.dump(stats, open(out / "stats.json", "w"))
    import h5py

    n_eps, per_task = 0, {}
    for s in shards:
        with h5py.File(out / s, "r") as f:
            k = len(f["ep_len"])
            n_eps += k
            per_task[s.split("_")[0]] = per_task.get(s.split("_")[0], 0) + k
    att = {}
    for r in results:
        d = att.setdefault(r["task"], [0, 0])
        d[0] += r["episodes"]
        d[1] += r["attempts"]
    write_subsets(out, shards)
    json.dump({"config": asdict(cfg), "shards": shards, "episodes": n_eps, "per_task": per_task,
               "expert_success_this_run": {t: v[0] / max(v[1], 1) for t, v in att.items()}},
              open(out / "manifest.json", "w"), indent=1)
    print(f"DONE: {n_eps} episodes, {stats['n_frames']} frames in {len(shards)} shards; per task {per_task}; "
          f"expert success this run { {t: round(v[0]/max(v[1],1),3) for t, v in att.items()} }")


if __name__ == "__main__":
    main()
