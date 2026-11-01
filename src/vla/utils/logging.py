"""MetricsLogger: JSONL (source of truth) + TensorBoard, written to <run_dir>/{metrics.jsonl,tb/}."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir: str | os.PathLike, enabled: bool = True, tensorboard: bool = True):
        self.enabled = enabled  # set False on non-zero ranks
        self.run_dir = Path(run_dir)
        self._tb = None
        self._f = None
        if not enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.run_dir / "metrics.jsonl", "a", buffering=1)
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.run_dir / "tb"))

    def log(self, step: int, metrics: dict) -> None:
        if not self.enabled:
            return
        rec = {"step": int(step), "time": time.time(), **{k: _py(v) for k, v in metrics.items()}}
        self._f.write(json.dumps(rec) + "\n")
        if self._tb is not None:
            for k, v in rec.items():
                if k not in ("step", "time") and isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self._f:
            self._f.close()
        if self._tb:
            self._tb.close()


def _py(v):
    try:
        return v.item()  # numpy / torch scalar
    except AttributeError:
        return v


def host_mem() -> dict:
    """Host-side memory diagnostics for silent-death hunting: process RSS, its cgroup's usage/limit/events, /dev/shm use."""
    import shutil

    out: dict = {}
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                out["rss_gb"] = int(ln.split()[1]) / 2**20
    except OSError:
        pass
    try:
        cg = [x for x in open("/proc/self/cgroup").read().splitlines() if x.startswith("0::")][0][3:]
        base = f"/sys/fs/cgroup{cg}"
        for key, f in (("cgroup_mem_gb", "memory.current"), ("cgroup_mem_max_gb", "memory.max"),
                       ("cgroup_mem_high_gb", "memory.high")):
            try:
                v = open(f"{base}/{f}").read().strip()
                out[key] = float(v) / 2**30 if v != "max" else float("inf")
            except OSError:
                pass
        try:
            ev = dict(x.split() for x in open(f"{base}/memory.events").read().splitlines())
            out["cgroup_oom_kill"] = int(ev.get("oom_kill", 0))
        except OSError:
            pass
    except (OSError, IndexError):
        pass
    try:
        out["shm_used_gb"] = shutil.disk_usage("/dev/shm").used / 2**30
    except OSError:
        pass
    return out
