"""GPU snapshots and quiet-window waiting for shared-GPU benchmarking (PLAN v2 'Shared-GPU defensibility').

snapshot(): per-GPU util/mem, foreign compute apps (pids not in our process tree) and their memory, host load,
`contended` = foreign GPU util above `foreign_util_max` on the GPUs we use. Without nvidia-smi: {"available": False}.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time


def _smi(args: list[str]) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=30)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return [[c.strip() for c in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def _own_pids() -> set[int]:
    """This process and all descendants (rollout workers, dataloader workers)."""
    me = os.getpid()
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True, text=True).stdout.splitlines()[1:]
        kids: dict[int, list[int]] = {}
        for ln in out:
            p, pp = (int(x) for x in ln.split())
            kids.setdefault(pp, []).append(p)
    except Exception:  # noqa: BLE001
        return {me}
    seen, stack = set(), [me]
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(kids.get(p, []))
    return seen


def visible_gpus() -> list[int]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [int(x) for x in v.split(",") if x.strip().isdigit()] or []


def snapshot(foreign_util_max: float = 20.0, gpus: list[int] | None = None) -> dict:
    snap = {"time": time.time(), "host_load": list(os.getloadavg()), "available": False}
    if shutil.which("nvidia-smi") is None:
        return snap
    try:
        rows = _smi(["--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu"])
        apps = _smi(["--query-compute-apps=pid,gpu_uuid,used_memory,process_name"])
    except Exception as e:  # noqa: BLE001
        snap["error"] = str(e)
        return snap
    mine = _own_pids()
    want = set(gpus if gpus is not None else visible_gpus())
    uuid2idx = {r[1]: int(r[0]) for r in rows}
    gl = []
    for r in rows:
        idx = int(r[0])
        foreign = [{"pid": int(p[0]), "mem_mib": float(p[2]) if p[2].replace(".", "").isdigit() else None,
                    "name": p[3]} for p in apps if uuid2idx.get(p[1]) == idx and int(p[0]) not in mine]
        own_mem = sum(float(p[2]) for p in apps if uuid2idx.get(p[1]) == idx and int(p[0]) in mine
                      and p[2].replace(".", "").isdigit())
        gl.append({"index": idx, "name": r[2], "mem_used_mib": float(r[3]), "mem_total_mib": float(r[4]),
                   "util_pct": float(r[5]), "temp_c": float(r[6]), "own_mem_mib": own_mem, "foreign_apps": foreign})
    snap["gpus"], snap["available"] = gl, True
    ours = [g for g in gl if not want or g["index"] in want]
    # util cannot be attributed per process; if we run GPU work ourselves (own_mem>0) it is an upper bound on foreign
    snap["contended"] = any(g["util_pct"] > foreign_util_max and (g["foreign_apps"] or g["own_mem_mib"] == 0)
                            for g in ours)
    return snap


def quiet_wait(gpus: list[int], util_max: float = 15.0, hold: float = 60.0, timeout: float = 1200.0,
               poll: float = 2.0, verbose: bool = True) -> dict:
    """Block until all `gpus` stay at util <= util_max for `hold` seconds. On timeout return (contended=True) anyway."""
    t0, quiet_since = time.time(), None
    if shutil.which("nvidia-smi") is None:
        return {"quiet": None, "waited_s": 0.0, "reason": "no nvidia-smi"}
    while True:
        rows = _smi(["--query-gpu=index,utilization.gpu"])
        util = {int(r[0]): float(r[1]) for r in rows}
        ok = all(util.get(g, 100.0) <= util_max for g in gpus)
        now = time.time()
        if ok:
            quiet_since = quiet_since or now
            if now - quiet_since >= hold:
                return {"quiet": True, "waited_s": now - t0}
        else:
            quiet_since = None
        if now - t0 >= timeout:
            if verbose:
                print(f"quiet_wait: timeout after {timeout:.0f}s, util={util}: running CONTENDED", flush=True)
            return {"quiet": False, "waited_s": now - t0, "util_at_timeout": util}
        time.sleep(poll)
