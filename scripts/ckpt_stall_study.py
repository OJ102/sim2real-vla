"""E: async DCP checkpoint stall study. {sync, async} x {NFS, /dev/shm}, interleaved per round.
/dev/shm leg: 5 saves; NFS leg: 2 saves, each deleted right after it is published (home snapshots make NFS churn costly).

    torchrun --nproc_per_node=4 scripts/ckpt_stall_study.py --out results/ckpt_stall.json \
        [--shm-dir $VLA_CKPT/_study] [--nfs-dir $VLA_RUNS/_ckpt_study] [--model-config configs/model_0p5b.yaml]

Per save: `stall_s` = wall time the training loop is blocked inside Checkpointer.save (what the loop pays);
`publish_s` = time until `latest` is published (end-to-end); `step_s_during` = median step time while an async write is
in flight vs `step_s_idle` (interference). Synthetic batches; the model is trained for a few steps so Adam state exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    from vla.model.prompts import encode_instructions
    from vla.model.vla import VLA, VLAConfig
    from vla.train.checkpoint import Checkpointer
    from vla.train.dist import init_dist, wrap_fsdp2
    from vla.utils.config import load_config
    from vla.utils.gpu import snapshot, visible_gpus

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="configs/model_0p5b.yaml")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--shm-dir", default=os.environ.get("VLA_CKPT", "/dev/shm") + "/_ckpt_stall_study")
    ap.add_argument("--nfs-dir", default=os.environ.get("VLA_RUNS", "runs") + "/_ckpt_stall_study")
    ap.add_argument("--shm-saves", type=int, default=5)
    ap.add_argument("--nfs-saves", type=int, default=2)
    ap.add_argument("--steps-between", type=int, default=4, help="training steps run after each save")
    ap.add_argument("--out", default="results/ckpt_stall.json")
    a = ap.parse_args()

    rank, world, local, device = init_dist()
    cuda = device.type == "cuda"
    mcfg = load_config(VLAConfig, a.model_config, [])
    torch.manual_seed(0)
    model = VLA(mcfg)
    if cuda:
        model.to(device)
    model = wrap_fsdp2(model, bf16=cuda) if (world > 1 or cuda) else model
    base = model.module if hasattr(model, "module") else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=cuda)
    B = a.micro_batch
    batch = {"image": torch.randint(0, 255, (B, 128, 128, 3), dtype=torch.uint8, device=device),
             "proprio": torch.randn(B, 11, device=device), "actions": torch.rand(B, 4, 4, device=device) * 2 - 1,
             **{k: v.to(device) for k, v in encode_instructions(base.tokenizer, ["lift the red cube"] * B).items()}}

    def train_step() -> float:
        t0 = time.perf_counter()
        model(batch)["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(3):  # warm up + materialise optimizer state
        train_step()
    idle = [train_step() for _ in range(8)]
    legs = {}
    for loc, d, n in (("shm", a.shm_dir, a.shm_saves), ("nfs", a.nfs_dir, a.nfs_saves)):
        for mode in ("sync", "async"):
            run_dir = Path(d) / f"{mode}"
            if rank == 0:
                shutil.rmtree(run_dir, ignore_errors=True)
            dist.barrier() if dist.is_initialized() else None
            legs[(mode, loc)] = {"ck": Checkpointer(run_dir, mode, keep=1, log=lambda m: print(m, flush=True)),
                                 "n": n, "recs": [], "dir": run_dir}
    snap_before = snapshot(gpus=visible_gpus() or None)
    step = 0
    for rep in range(max(a.shm_saves, a.nfs_saves)):
        for key, leg in legs.items():  # interleaved: sync-shm, async-shm, sync-nfs, async-nfs each round
            if rep >= leg["n"]:
                continue
            ck: Checkpointer = leg["ck"]
            step += 1
            t0 = time.perf_counter()
            stall = ck.save(step, model, opt, {"step": step})
            during = []
            for _ in range(a.steps_between):  # keep training while an async write is (maybe) in flight
                during.append(train_step())
            ck.wait()
            publish = time.perf_counter() - t0
            size = sum(f.stat().st_size for f in (leg["dir"] / f"step_{step:06d}").glob("*")) if rank == 0 else 0
            leg["recs"].append({"stall_s": stall, "publish_s": publish, "step_s_during": statistics.median(during),
                                "ckpt_gb_rank0_files": size / 2**30, "failed": ck.n_failed})
            if key[1] == "nfs":  # delete right away (snapshots keep deleted bytes charged; keep churn minimal)
                dist.barrier() if dist.is_initialized() else None
                if rank == 0:
                    shutil.rmtree(leg["dir"], ignore_errors=True)
            if rank == 0:
                print(f"{key} save {rep}: stall {stall:.2f}s publish {publish:.1f}s", flush=True)
    snap_after = snapshot(gpus=visible_gpus() or None)
    if rank == 0:
        out = {"world": world, "model_config": a.model_config, "step_s_idle_median": statistics.median(idle),
               "gpu_snapshot_before": snap_before, "gpu_snapshot_after": snap_after,
               "contended": bool(snap_before.get("contended") or snap_after.get("contended")), "legs": {}}
        for (mode, loc), leg in legs.items():
            st = [r["stall_s"] for r in leg["recs"]]
            out["legs"][f"{mode}_{loc}"] = {"saves": len(st), "stall_s_median": statistics.median(st),
                                           "stall_s_min": min(st), "stall_s_max": max(st),
                                           "publish_s_median": statistics.median(r["publish_s"] for r in leg["recs"]),
                                           "step_s_during_median": statistics.median(r["step_s_during"] for r in leg["recs"]),
                                           "records": leg["recs"]}
        for loc in ("shm", "nfs"):
            s, asy = out["legs"][f"sync_{loc}"], out["legs"][f"async_{loc}"]
            out[f"async_stall_over_sync_{loc}"] = asy["stall_s_median"] / max(s["stall_s_median"], 1e-9)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}; async/sync stall ratio shm={out['async_stall_over_sync_shm']:.3f} nfs={out['async_stall_over_sync_nfs']:.3f}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
