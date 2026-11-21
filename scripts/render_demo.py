"""P8 demo videos: closed-loop rollouts of a VLA checkpoint (either head) or the scripted expert, rendered from the scene
camera at a higher resolution than the 128 px policy input, with a text overlay; plus one tasks x splits grid video.

    python scripts/render_demo.py --ckpt $VLA_RUNS/keep/bc_gdisc3_best.pt --model-config configs/model_0p5b.yaml \
        --head discrete --data-dir $VLA_DATA/demos --episodes 3 --name bc_gdisc3 [--temperature 1] [--exec-horizon 4]
    python scripts/render_demo.py --policy expert --episodes 3
    # CPU smoke (Mac, random-init tiny model):
    python scripts/render_demo.py --model-config configs/model_tiny.yaml --head discrete --tasks reach --splits train \
        --episodes 1 --res 128 --no-require-nvidia --name smoke

Seeds are not cherry-picked: cell (task, split) uses the eval seeds seed0 + i for i < --episodes, exactly like
vla.eval.rollout_eval (and the same per-batch sampling generator when --temperature is set). The grid shows episode
--grid-episode (default 0, the first seed) of every cell, and its label carries the cell's success over all its seeds.
Output: results/demo/<name>/{task}_{split}_s{seed}.mp4, grid.mp4, index.json (seed, success, steps per episode).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from vla.eval.rollout_eval import EVAL_SEED0, SPLITS, sample_seed
from vla.sim.env import TASKS, TabletopEnv, set_egl_device

FPS = 10          # env control rate (10 Hz): videos play in real time
HOLD_S = 1.5      # the final frame (with the SUCCESS/FAIL banner) is held this long


class HiResCam:
    """Second renderer on the env's scene camera (same DR'd pose/fov as the policy view), at video resolution."""

    def __init__(self, env: TabletopEnv, res: int):
        import mujoco

        from vla.sim.assets import CAM_NAME

        self.env, self.cam = env, CAM_NAME
        self.r = mujoco.Renderer(env.model, res, res)

    def __call__(self) -> np.ndarray:
        self.r.update_scene(self.env.data, camera=self.cam)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def overlay(frame: np.ndarray, lines: list[str], result: bool | None = None) -> np.ndarray:
    """Text lines on a translucent top band; result=True/False adds a SUCCESS/FAIL banner at the bottom."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGBA")
    H = img.height
    fs = max(10, H // 26)
    f = _font(fs)
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, img.width, (fs + 4) * len(lines) + 6], fill=(0, 0, 0, 150))
    if result is not None:
        bh = int(fs * 2.2)
        d.rectangle([0, H - bh, img.width, H], fill=(20, 140, 40, 210) if result else (170, 30, 30, 210))
    img = Image.alpha_composite(img, band)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        d.text((6, 4 + i * (fs + 4)), s, fill=(255, 255, 255, 255), font=f)
    if result is not None:
        big = _font(int(fs * 1.6))
        msg = "SUCCESS" if result else "FAIL"
        w = d.textlength(msg, font=big)
        d.text(((img.width - w) / 2, H - int(fs * 2.2) + fs * 0.25), msg, fill=(255, 255, 255, 255), font=big)
    return np.asarray(img.convert("RGB"))


def rollout_cell(policy, task: str, split: str, seeds: list[int], horizon: int, res: int, seed0: int,
                 require_nvidia: bool) -> list[dict]:
    """All seeds of one (task, split) cell as ONE batch (like a rollout_eval wave with n_envs >= len(seeds)).
    policy=None -> scripted expert (same rng scheme as SubprocVecEnv.expert_episode). Returns per-episode dicts with
    raw hi-res frames (frame 0 = after reset), success, steps, instruction."""
    from vla.sim.expert import ScriptedExpert

    envs = [TabletopEnv(task=task, split=split, render_obs=policy is not None) for _ in seeds]
    if require_nvidia:
        from vla.sim.egl import assert_nvidia_renderer

        assert_nvidia_renderer(envs[0])
    cams = [HiResCam(e, res) for e in envs]
    try:
        obs, info = zip(*[e.reset(seed=s) for e, s in zip(envs, seeds)])
        obs = list(obs)
        eps = [{"seed": s, "instruction": i["instruction"], "frames": [c()], "success": False, "steps": 0}
               for s, i, c in zip(seeds, info, cams)]
        experts = [ScriptedExpert(task, np.random.default_rng(10_000 + s)) for s in seeds] if policy is None else None
        gen = None
        if policy is not None and getattr(policy, "engine", None) is not None:  # --temperature: wave 0 of this cell
            gen = policy.make_generator(sample_seed(seed0, split, task, 0))
        active, queue, t = set(range(len(seeds))), {}, 0
        while active:
            idx = sorted(active)
            if policy is None:
                acts = {i: experts[i].act(envs[i]) for i in idx}
            else:
                if t % horizon == 0:
                    chunk = policy.act(np.stack([obs[i]["image"] for i in idx]), np.stack([obs[i]["proprio"] for i in idx]),
                                       [eps[i]["instruction"] for i in idx], **({} if gen is None else {"generator": gen}))
                    queue = {i: chunk[k] for k, i in enumerate(idx)}
                acts = {i: queue[i][t % horizon].astype(np.float32) for i in idx}
            for i in idx:
                obs[i], _, term, trunc, inf = envs[i].step(acts[i])
                eps[i]["frames"].append(cams[i]())
                eps[i]["steps"] = inf["step"]
                eps[i]["success"] = bool(inf["success"])
                if term or trunc:
                    active.discard(i)
            t += 1
    finally:
        for c in cams:
            c.close()
        for e in envs:
            e.close()
    return eps


def annotate(ep: dict, task: str, split: str, label: str) -> list[np.ndarray]:
    n = len(ep["frames"])
    out = []
    for k, f in enumerate(ep["frames"]):
        last = k == n - 1
        out.append(overlay(f, [ep["instruction"], f"{task}/{split}  seed {ep['seed']}  {label}",
                               f"step {k}/{ep['steps']}"], ep["success"] if last else None))
    return out + [out[-1]] * int(HOLD_S * FPS)


def write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=FPS, macro_block_size=1, quality=8)


def grid_video(cells: dict, tasks: list[str], splits: list[str], cell_px: int) -> list[np.ndarray]:
    """rows = tasks, cols = splits; each cell: (annotated frames, label). Shorter cells hold their last frame."""
    from PIL import Image

    T = max(len(fr) for fr, _ in cells.values())
    blank = np.zeros((cell_px, cell_px, 3), np.uint8)
    rs = {k: [np.asarray(Image.fromarray(f).resize((cell_px, cell_px), Image.BILINEAR)) for f in fr]
          for k, (fr, _) in cells.items()}
    frames = []
    for t in range(T):
        rows = []
        for task in tasks:
            row = [rs[(task, s)][min(t, len(rs[(task, s)]) - 1)] if (task, s) in rs else blank for s in splits]
            rows.append(np.concatenate(row, 1))
        frames.append(np.concatenate(rows, 0))
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["vla", "expert"], default="vla")
    ap.add_argument("--ckpt", default=None, help="final.pt / best.pt / DCP dir; omitted = random init (smoke tests)")
    ap.add_argument("--model-config", default="configs/model_0p5b.yaml")
    ap.add_argument("--head", default="")
    ap.add_argument("--data-dir", default=None, help="demo dir holding stats.json (proprio normalisation)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--episodes", type=int, default=3, help="first k eval seeds per (task, split)")
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--exec-horizon", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=None, help="discrete head: sample softmax(logits/T); absent = greedy")
    ap.add_argument("--res", type=int, default=448, help="video resolution (square, <= 640 offscreen buffer)")
    ap.add_argument("--grid-cell", type=int, default=320, help="grid cell size in px")
    ap.add_argument("--grid-episode", type=int, default=0, help="which seed index each grid cell shows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="model config key=value")
    ap.add_argument("--no-require-nvidia", action="store_true", help="skip the NVIDIA EGL renderer check (CPU/Mac)")
    ap.add_argument("--name", default=None, help="output subdir (default: ckpt stem, or 'expert')")
    ap.add_argument("--out-root", default="results/demo")
    a = ap.parse_args()

    if not 16 <= a.res <= 640:
        ap.error("--res must be in [16, 640] (MuJoCo offscreen buffer is 640x640)")
    if not 0 <= a.grid_episode < a.episodes:
        ap.error("--grid-episode must be < --episodes")
    T = a.temperature
    if T is not None and not (np.isfinite(T) and T > 0):
        ap.error("--temperature must be finite and > 0 (omit for greedy)")
    tasks, splits = a.tasks.split(","), a.splits.split(",")
    for t in tasks:
        if t not in TASKS:
            ap.error(f"unknown task {t!r}")
    for s in splits:
        if s not in SPLITS:
            ap.error(f"unknown split {s!r}")
    require_nvidia = sys.platform.startswith("linux") and not a.no_require_nvidia
    set_egl_device(0)

    policy, label = None, "expert"
    if a.policy == "vla":
        import torch

        from vla.infer.policy import VLAPolicy

        dev = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        stats = os.path.join(a.data_dir, "stats.json") if a.data_dir else None
        policy = VLAPolicy(a.model_config, a.ckpt, stats, a.head, dev, torch.bfloat16 if a.bf16 else torch.float32,
                           a.override, temperature=T)
        if T is not None and policy.engine is None:
            warnings.warn(f"--temperature ignored for head={policy.cfg.head}; running greedy")
            T = None
        label = f"{policy.cfg.head} {'greedy' if T is None else f'T={T:g}'}"
    elif T is not None:
        warnings.warn("--temperature ignored for --policy expert")
        T = None
    name = a.name or (Path(a.ckpt).stem if a.ckpt else ("expert" if policy is None else "random_init"))
    out = Path(a.out_root) / name
    out.mkdir(parents=True, exist_ok=True)

    index = {"meta": {"policy": a.policy, "ckpt": a.ckpt, "model_config": a.model_config, "head": a.head,
                      "temperature": T, "exec_horizon": a.exec_horizon, "seed0": a.seed0, "episodes": a.episodes,
                      "res": a.res, "grid_episode": a.grid_episode}, "cells": {}}
    cells = {}
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    for task in tasks:
        for split in splits:
            eps = rollout_cell(policy, task, split, seeds, a.exec_horizon, a.res, a.seed0, require_nvidia)
            n_ok = sum(e["success"] for e in eps)
            rec = []
            for j, ep in enumerate(eps):
                frames = annotate(ep, task, split, label)
                fn = f"{task}_{split}_s{ep['seed']}.mp4"
                write_mp4(out / fn, frames)
                rec.append({"seed": ep["seed"], "success": ep["success"], "steps": ep["steps"],
                            "instruction": ep["instruction"], "video": fn})
                if j == a.grid_episode:
                    cells[(task, split)] = (frames, f"{n_ok}/{len(eps)}")
            index["cells"][f"{task}/{split}"] = {"success": n_ok / len(eps), "n": len(eps), "episodes": rec}
            print(f"{task}/{split}: success {n_ok}/{len(eps)}  "
                  + " ".join(f"s{r['seed']}:{'OK' if r['success'] else 'fail'}({r['steps']})" for r in rec), flush=True)
    grid = grid_video(cells, tasks, splits, a.grid_cell)
    grid = [_grid_labels(g, tasks, splits, cells, a.grid_cell) for g in grid]
    write_mp4(out / "grid.mp4", grid)
    index["grid"] = {"video": "grid.mp4", "rows": tasks, "cols": splits, "episode_index": a.grid_episode}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {out}/ ({sum(len(c['episodes']) for c in index['cells'].values())} episode videos + grid.mp4, index.json)")


def _grid_labels(frame: np.ndarray, tasks, splits, cells, px: int) -> np.ndarray:
    """Per-cell tag, bottom right: 'cell k/n' = that cell's success over ALL its seeds, not just the one shown."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    f = _font(max(10, px // 18))
    for r, task in enumerate(tasks):
        for c, split in enumerate(splits):
            if (task, split) not in cells:
                continue
            s = f"cell {cells[(task, split)][1]}"
            w = d.textlength(s, font=f)
            th = px // 18 + 4
            # above the (scaled) SUCCESS/FAIL banner, which covers the bottom 2.2/26 of each cell
            x, y = (c + 1) * px - w - 6, int(r * px + px * (1 - 2.2 / 26)) - th - 4
            d.rectangle([x - 3, y - 2, x + w + 3, y + px // 18 + 4], fill=(0, 0, 0))
            d.text((x, y), s, fill=(255, 230, 80), font=f)
    return np.asarray(img)


if __name__ == "__main__":
    main()
