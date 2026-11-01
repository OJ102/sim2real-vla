"""Closed-loop evaluation: batched rollouts of the VLA (either head) or the scripted expert on fixed seed sets.

    python -m vla.eval.rollout_eval --policy vla --ckpt $VLA_RUNS/bc_disc/final.pt --model-config configs/model_0p5b.yaml \
        --head discrete --data-dir $VLA_DATA/demos --splits train,heldout,heldout_photo --episodes 200 --n-envs 32 \
        --out results/eval_bc_disc.json --quiet-wait
    python -m vla.eval.rollout_eval --policy expert --out results/eval_expert.json

Seeds: eval episode i of every (split, task) uses seed eval_seed0 + i (< 100000; demo-training seeds start at 100000).
Action chunks are executed open-loop for `--exec-horizon` steps (default = chunk size 4).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from vla.eval.metrics import aggregate, episode_metrics, load_ceiling
from vla.sim.vec_env import SubprocVecEnv
from vla.utils.gpu import quiet_wait, snapshot, visible_gpus

SPLITS = ("train", "heldout", "heldout_photo")
EVAL_SEED0 = 1000


def run_vla(vec, policy, task, split, seeds, horizon, max_steps=100):
    """Rollout waves of len(vec) episodes. Returns list of episode metric dicts."""
    out = []
    for w in range(0, len(seeds), vec.n):
        wseeds = seeds[w:w + vec.n]
        idx = list(range(len(wseeds)))
        res = vec.reset(idx, task, split, wseeds)
        obs = {i: r[0] for i, r in zip(idx, res)}
        instr = {i: r[1]["instruction"] for i, r in zip(idx, res)}
        acts, infos, active, queue = {i: [] for i in idx}, {i: [] for i in idx}, set(idx), {}
        t = 0
        while active:
            act_idx = sorted(active)
            if t % horizon == 0:
                chunk = policy.act(np.stack([obs[i]["image"] for i in act_idx]),
                                   np.stack([obs[i]["proprio"] for i in act_idx]), [instr[i] for i in act_idx])
                queue = {i: chunk[k] for k, i in enumerate(act_idx)}
            a = np.stack([queue[i][t % horizon] for i in act_idx]).astype(np.float32)
            for i, (o, _, term, trunc, info) in zip(act_idx, vec.step(act_idx, a)):
                obs[i] = o
                acts[i].append(queue[i][t % horizon])
                infos[i].append(info)
                if term or trunc:
                    active.discard(i)
            t += 1
        out += [episode_metrics(np.stack(acts[i]), infos[i]) for i in idx]
    return out


def run_expert(vec, task, split, seeds, noise=0.05):
    out = []
    for w in range(0, len(seeds), vec.n):
        wseeds = seeds[w:w + vec.n]
        idx = list(range(len(wseeds)))
        for traj in vec.expert_episodes(idx, task, split, wseeds, noise):
            out.append(episode_metrics(np.stack([a for a, _ in traj]), [i for _, i in traj]))
    return out


def evaluate(policy, vec, splits, tasks, episodes, horizon, ceiling, seed0=EVAL_SEED0, log=print):
    results = {}
    for split in splits:
        results[split] = {}
        for task in tasks:
            seeds = list(range(seed0, seed0 + episodes))
            t0 = time.time()
            eps = run_expert(vec, task, split, seeds) if policy is None else \
                run_vla(vec, policy, task, split, seeds, horizon)
            agg = aggregate(eps, ceiling.get(split, {}).get(task, {}).get("success"))
            agg["eval_secs"] = time.time() - t0
            results[split][task] = agg
            log(f"{split}/{task}: success {agg['success']:.3f} CI [{agg['ci95'][0]:.2f},{agg['ci95'][1]:.2f}] "
                f"n={agg['n']} jerk {agg['ee_jerk_rms']:.1f} ({agg['eval_secs']:.0f}s)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["vla", "expert"], default="vla")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model-config", default="configs/model_0p5b.yaml")
    ap.add_argument("--head", default="")
    ap.add_argument("--data-dir", default=None, help="demo dir holding stats.json (proprio normalisation)")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--tasks", default="lift,push")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--exec-horizon", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--quiet-wait", action="store_true")
    ap.add_argument("--ceiling", default="results/expert_ceiling.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--override", nargs="*", default=[], help="model config key=value (e.g. attn_impl=eager)")
    a = ap.parse_args()

    meta = {"policy": a.policy, "ckpt": a.ckpt, "head": a.head, "episodes": a.episodes, "seed0": a.seed0,
            "exec_horizon": a.exec_horizon, "tasks": a.tasks, "splits": a.splits}
    if a.quiet_wait:
        meta["quiet_wait"] = quiet_wait(visible_gpus() or [0], util_max=15, hold=60, timeout=1200)
    meta["gpu_snapshot_before"] = snapshot()
    policy = None
    if a.policy == "vla":
        import torch

        from vla.infer.policy import VLAPolicy

        dev = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        stats = os.path.join(a.data_dir, "stats.json") if a.data_dir else None
        policy = VLAPolicy(a.model_config, a.ckpt, stats, a.head, dev, torch.bfloat16 if a.bf16 else torch.float32,
                           a.override)
    vec = SubprocVecEnv(a.n_envs)
    try:
        res = evaluate(policy, vec, a.splits.split(","), a.tasks.split(","), a.episodes, a.exec_horizon,
                       load_ceiling(a.ceiling), a.seed0)
    finally:
        vec.close()
    meta["gpu_snapshot_after"] = snapshot()
    meta["contended"] = bool(meta["gpu_snapshot_before"].get("contended") or meta["gpu_snapshot_after"].get("contended")
                             or meta.get("quiet_wait", {}).get("quiet") is False)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"meta": meta, "results": res}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out} (contended={meta['contended']})")


if __name__ == "__main__":
    main()
