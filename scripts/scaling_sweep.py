"""P4b scaling sweep: {FSDP2, ZeRO-3} x {1,2,4 GPUs} x {act_ckpt off,on}, 120 steps each (first 20 dropped), `repeats`
interleaved passes (ABAB..) so drift cancels, quiet-wait before every run, GPU snapshots embedded per run.

    uv run python scripts/scaling_sweep.py --out results/scaling.json [--repeats 2] [--gpus 0,1,2,4] [--dry-run]
Report per config: samples/s (median over repeats of per-run median), MFU, peak mem (process-local max_memory_allocated),
scaling efficiency = throughput_N / (N * throughput_1) for the same (backend, act_ckpt)."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from vla.utils.gpu import quiet_wait, snapshot


def run_one(cfg: dict, gpus: list[int], extra: list[str], port: int, runs_dir: str, data_dir: str, quiet: bool) -> dict:
    n = cfg["n_gpu"]
    name = f"scale_{cfg['backend']}_{n}g_ac{int(cfg['act_ckpt'])}_r{cfg['rep']}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus[:n])), "PYTORCH_CUDA_ALLOC_CONF":
           "expandable_segments:True", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    rec = {**cfg, "name": name, "gpus": gpus[:n]}
    if quiet:
        rec["quiet_wait"] = quiet_wait(gpus[:n], util_max=15, hold=60, timeout=1200)
    rec["gpu_snapshot_before"] = snapshot(gpus=gpus[:n])
    cmd = ["torchrun", "--nnodes=1", f"--nproc_per_node={n}", "--rdzv_backend=c10d",
           f"--rdzv_endpoint=localhost:{port}", "-m", "vla.train.bc", "--config", "configs/bc.yaml", f"name={name}",
           f"runs_dir={runs_dir}", f"data_dir={data_dir}", f"backend={cfg['backend']}", f"act_ckpt={str(cfg['act_ckpt']).lower()}",
           "steps=120", "log_every=1", "val_every=100000", "ckpt_every=100000", "export_final=false", *extra]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    rec["returncode"] = r.returncode
    rec["gpu_snapshot_after"] = snapshot(gpus=gpus[:n])
    rec["contended"] = bool(rec["gpu_snapshot_before"].get("contended") or rec["gpu_snapshot_after"].get("contended")
                            or rec.get("quiet_wait", {}).get("quiet") is False)
    mf = Path(runs_dir) / name / "metrics.jsonl"
    if r.returncode != 0 or not mf.exists():
        rec["error"] = (r.stderr or "")[-1500:]
        return rec
    rows = [json.loads(x) for x in mf.read_text().splitlines()]
    rows = [x for x in rows if "step_time" in x and x["step"] > 20]
    rec["samples_per_s"] = statistics.median(x["samples_per_s"] for x in rows)
    rec["mfu"] = statistics.median(x["mfu"] for x in rows)
    rec["step_time"] = statistics.median(x["step_time"] for x in rows)
    rec["peak_mem_gb_per_rank"] = [max(x["mem_alloc_gb_per_rank"][i] for x in rows) for i in range(n)]
    return rec


def summarize(runs: list[dict]) -> list[dict]:
    groups: dict = {}
    for r in runs:
        if "samples_per_s" in r:
            groups.setdefault((r["backend"], r["n_gpu"], r["act_ckpt"]), []).append(r)
    out = []
    for (b, n, ac), rs in sorted(groups.items()):
        sp = [r["samples_per_s"] for r in rs]
        out.append({"backend": b, "n_gpu": n, "act_ckpt": ac, "repeats": len(rs),
                    "samples_per_s_median": statistics.median(sp), "samples_per_s_min": min(sp),
                    "samples_per_s_max": max(sp), "mfu_median": statistics.median(r["mfu"] for r in rs),
                    "peak_mem_gb_max_rank": max(max(r["peak_mem_gb_per_rank"]) for r in rs),
                    "any_contended": any(r["contended"] for r in rs)})
    base = {(o["backend"], o["act_ckpt"]): o["samples_per_s_median"] for o in out if o["n_gpu"] == 1}
    for o in out:
        b1 = base.get((o["backend"], o["act_ckpt"]))
        o["scaling_efficiency"] = o["samples_per_s_median"] / (o["n_gpu"] * b1) if b1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scaling.json")
    ap.add_argument("--gpus", default="0,1,2,4")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--backends", default="fsdp,deepspeed")
    ap.add_argument("--n-gpus", default="1,2,4")
    ap.add_argument("--runs-dir", default=os.environ.get("VLA_RUNS", "runs"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-quiet-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra bc.py key=value overrides (e.g. micro_batch=8)")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    data_dir = a.data_dir or str(Path(os.environ.get("VLA_DATA", "")) / "demos")
    cfgs = [{"backend": b, "n_gpu": int(n), "act_ckpt": ac}
            for b, n, ac in itertools.product(a.backends.split(","), a.n_gpus.split(","), [False, True])]
    runs, port = [], 29700
    for rep in range(a.repeats):  # interleaved: every config once per pass
        for c in cfgs:
            c = {**c, "rep": rep}
            if a.dry_run:
                print("would run", c)
                continue
            port += 1
            r = run_one(c, gpus, a.extra, port, a.runs_dir, data_dir, not a.no_quiet_wait)
            runs.append(r)
            print(f"{r['name']}: " + (f"{r['samples_per_s']:.1f} samp/s mfu {r['mfu']*100:.1f}% contended={r['contended']}"
                                      if "samples_per_s" in r else f"FAILED rc={r['returncode']}"), flush=True)
            json.dump({"configs": summarize(runs), "runs": runs}, open(a.out, "w"), indent=1)
    if not a.dry_run:
        print(json.dumps(summarize(runs), indent=1))


if __name__ == "__main__":
    main()
