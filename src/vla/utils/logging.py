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
