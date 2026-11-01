"""Rollout worker (one per rollout rank): generates GRPO batches with the InferenceEngine and writes them atomically
to /dev/shm. Schedule (S = max staleness): produce batch k with the current weights; after batch k >= S, join the
weight sync (receives learner version k-S+1). S=0: serial on-policy. S=1: rollout k+1 overlaps learner update k."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from vla.infer.engine import InferenceEngine
from vla.model.prompts import encode_instructions
from vla.rl.weight_sync import sync_weights
from vla.sim.rewards import episode_return
from vla.sim.vec_env import SubprocVecEnv


def batch_path(shm: Path, rank: int, k: int) -> Path:
    return shm / f"rollout{rank}_batch{k:05d}.npz"


def collect_batch(vec: SubprocVecEnv, engine: InferenceEngine, cfg, seeds: list[int], p_mean, p_std, gen):
    """One GRPO batch: len(seeds) distinct seeds x G rollouts each (same seed => same DR draw + layout)."""
    G = cfg.group_size
    tasks = [cfg.tasks[j % len(cfg.tasks)] for j in range(len(seeds))]
    ep_seed = [s for s in seeds for _ in range(G)]
    ep_task = [t for t in tasks for _ in range(G)]
    n = len(ep_seed)
    obs, instr = {}, {}
    for task in sorted(set(ep_task)):
        idx = [i for i in range(n) if ep_task[i] == task]
        for i, (o, info) in zip(idx, vec.reset(idx, task, "train", [ep_seed[i] for i in idx])):
            obs[i], instr[i] = o, info["instruction"]
    active, acts, infos = set(range(n)), {i: [] for i in range(n)}, {i: [] for i in range(n)}
    rec = {k: [] for k in ("image", "proprio", "bins", "logp", "ep")}
    t0, env_steps, H = time.perf_counter(), 0, engine.model.cfg.chunk
    tok = engine.model.tokenizer
    while active:
        ids = sorted(active)
        proprio = np.stack([obs[i]["proprio"] for i in ids])
        if p_mean is not None:
            proprio = ((proprio - p_mean) / p_std).astype(np.float32)
        dev = next(engine.model.parameters()).device
        batch = {"image": torch.from_numpy(np.stack([obs[i]["image"] for i in ids])).to(dev),
                 "proprio": torch.from_numpy(proprio).to(dev),
                 **{k: v.to(dev) for k, v in encode_instructions(tok, [instr[i] for i in ids]).items()}}
        out = engine.generate(batch, generator=gen)
        chunk = out["actions"].float().cpu().numpy()
        rec["image"].append(batch["image"].cpu().numpy()); rec["proprio"].append(proprio)
        rec["bins"].append(out["bins"].cpu().numpy()); rec["logp"].append(out["logp"].float().cpu().numpy())
        rec["ep"].append(np.array(ids))
        for h in range(H):
            live = [i for i in ids if i in active]
            if not live:
                break
            a = np.stack([chunk[ids.index(i), h] for i in live]).astype(np.float32)
            for i, (o, _, term, trunc, info) in zip(live, vec.step(live, a)):
                obs[i] = o
                acts[i].append(a[live.index(i)])
                infos[i].append(info)
                env_steps += 1
                if term or trunc:
                    active.discard(i)
    gen_time = time.perf_counter() - t0
    R, comps, succ = [], [], []
    for i in range(n):
        r, c = episode_return(np.stack(acts[i]), infos[i], cfg.reward_weights)
        R.append(r); comps.append(c); succ.append(float(infos[i][-1]["success"]))
    ep = np.concatenate(rec["ep"])
    texts = [instr[i] for i in range(n)]
    ids_pad = encode_instructions(tok, texts)  # same left-padded layout the learner will rebuild from texts
    data = {"image": np.concatenate(rec["image"]), "proprio": np.concatenate(rec["proprio"]),
            "bins": np.concatenate(rec["bins"]), "logp": np.concatenate(rec["logp"]), "ep": ep,
            "ep_return": np.array(R, np.float32), "ep_success": np.array(succ, np.float32),
            "ep_seed": np.array(ep_seed), "instr": np.array(texts), "group": np.arange(n) // G}
    for k in comps[0]:
        data[f"comp_{k}"] = np.array([c[k] for c in comps], np.float32)
    meta = {"env_steps": env_steps, "gen_time": gen_time}
    del ids_pad
    return data, meta


def rollout_loop(cfg, roles, rank: int, device, model, sync_pg, p_mean, p_std, shm: Path, log=print):
    vec = SubprocVecEnv(cfg.n_envs)
    engine = InferenceEngine(model, cfg.temperature)
    gen = torch.Generator(device=device).manual_seed(cfg.seed * 7919 + rank)
    S, K = cfg.staleness, cfg.iterations
    ridx = roles.rollout_ranks.index(rank)
    idle = 0.0
    try:
        # initial sync: identical starting weights on all roles even without a shared init checkpoint
        info = sync_weights(model, sync_pg, roles.sync_src, 0, sync_dtype=cfg.sync_dtype_t)
        engine.version = info["version"]
        log(f"[rollout {rank}] initial sync v{engine.version} {info['latency_ms']:.0f} ms")
        for k in range(K):
            seeds = [cfg.seed0 + (k * len(roles.rollout_ranks) + ridx) * cfg.seeds_per_iter + j
                     for j in range(cfg.seeds_per_iter)]
            data, meta = collect_batch(vec, engine, cfg, seeds, p_mean, p_std, gen)
            data["version"] = np.int64(engine.version)
            meta.update(idle_s=idle, version=engine.version)
            data["meta"] = np.array(json.dumps(meta))
            path = batch_path(shm, rank, k)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **data)
            os.replace(tmp, path)  # atomic publish
            log(f"[rollout {rank}] batch {k} v{engine.version} return {data['ep_return'].mean():.2f} "
                f"succ {data['ep_success'].mean():.2f} {meta['env_steps']/meta['gen_time']:.0f} env-steps/s")
            idle = 0.0
            if k >= S:  # learner broadcasts after update j for j < K-S: exactly K-S syncs (k = S..K-1)
                t0 = time.perf_counter()
                info = sync_weights(model, sync_pg, roles.sync_src, 0, sync_dtype=cfg.sync_dtype_t)
                idle = time.perf_counter() - t0
                engine.version = info["version"]
    finally:
        vec.close()
