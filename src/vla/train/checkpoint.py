"""DCP checkpointing: sync or async, `latest` updated atomically only after a save fully completes, keep last N.

Layout: <run>/ckpt/step_XXXXXX/ (DCP directory), <run>/ckpt/latest (text file: "step_XXXXXX").
State saved: model, optimizer, `extra` (step, epoch, steps_in_epoch, scheduler, sampler state). RNG is not saved:
the training loop re-seeds every step from (seed, step, rank), so resume is deterministic without it.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict


def step_name(step: int) -> str:
    return f"step_{step:06d}"


class Checkpointer:
    def __init__(self, ckpt_dir: str | os.PathLike, mode: str = "sync", keep: int = 2, mirror_to: str | None = None):
        assert mode in ("sync", "async")
        self.dir, self.mode, self.keep, self.mirror_to = Path(ckpt_dir), mode, keep, mirror_to
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self._pending = None  # (future, step)
        # async_save stages to CPU and writes in a thread; it needs a CPU-capable group (gloo) even in nccl runs
        self._pg = dist.new_group(backend="gloo") if (dist.is_initialized() and mode == "async") else None
        self.last_stall_s = 0.0

    # ------------------------------------------------------------------ save
    def save(self, step: int, model, optimizer, extra: dict) -> float:
        """Returns the wall time this call blocked the training loop (the 'stall')."""
        t0 = time.perf_counter()
        self._finalize_pending(block=True)  # at most one save in flight
        msd, osd = get_state_dict(model, optimizer)
        state = {"model": msd, "optim": osd, "extra": extra}
        path = str(self.dir / step_name(step))
        if self.mode == "sync":
            dcp.save(state, checkpoint_id=path)
            self._publish(step)
        else:
            fut = dcp.async_save(state, checkpoint_id=path, process_group=self._pg)
            self._pending = (fut, step)
        self.last_stall_s = time.perf_counter() - t0
        return self.last_stall_s

    def poll(self) -> None:
        """Call every step: publishes `latest` as soon as an in-flight async save finishes (non-blocking)."""
        self._finalize_pending(block=False)

    def wait(self) -> None:
        self._finalize_pending(block=True)

    def _finalize_pending(self, block: bool) -> None:
        if self._pending is None:
            return
        fut, step = self._pending
        if not block:
            # collective consensus on the (separate) gloo group so all ranks decide together; polling is called
            # by every rank every step, so this never mismatches the training collectives
            done = torch.tensor([int(fut.done())])
            if dist.is_initialized():
                dist.all_reduce(done, op=dist.ReduceOp.MIN, group=self._pg)
            if not done.item():
                return
        fut.result()
        self._pending = None
        # every rank must have finished writing before rank 0 publishes
        if dist.is_initialized():
            dist.barrier(group=self._pg)
        self._publish(step, barrier=False)

    def _publish(self, step: int, barrier: bool = True) -> None:
        if barrier and dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            tmp = self.dir / "latest.tmp"
            tmp.write_text(step_name(step))
            os.replace(tmp, self.dir / "latest")  # atomic
            steps = sorted(p.name for p in self.dir.glob("step_*") if p.is_dir())
            for old in steps[: -self.keep]:
                shutil.rmtree(self.dir / old, ignore_errors=True)
            if self.mirror_to:  # background mirror (e.g. /dev/shm -> NFS); never blocks training
                subprocess.Popen(["rsync", "-a", "--delete", f"{self.dir}/", f"{self.mirror_to}/"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------ load
    def latest(self) -> str | None:
        f = self.dir / "latest"
        return f.read_text().strip() if f.exists() else None

    def load_latest(self, model, optimizer, extra_template: dict) -> dict | None:
        name = self.latest()
        if name is None:
            return None
        msd, osd = get_state_dict(model, optimizer)
        state = {"model": msd, "optim": osd, "extra": dict(extra_template)}
        dcp.load(state, checkpoint_id=str(self.dir / name))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
        return state["extra"]


def export_final(model, path: str | os.PathLike, dtype=torch.bfloat16, group=None) -> None:
    """Model weights only (no optimizer): full state dict gathered to rank 0 and saved as one .pt file."""
    sd, _ = get_state_dict(model, [], options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if not dist.is_initialized() or dist.get_rank(group) == 0:
        torch.save({k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}, path)
    if dist.is_initialized():
        dist.barrier(group=group)  # group = the ranks that hold the model (e.g. learner ranks only in RL)


class PreemptFlag:
    """SIGTERM/SIGUSR1 set a flag; the loop checks it at step boundaries (all ranks agree via all-reduce)."""

    def __init__(self):
        self.flag = False
        for s in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(s, lambda *_: setattr(self, "flag", True))

    def should_stop(self, device) -> bool:
        if not dist.is_initialized():
            return self.flag
        t = torch.tensor([int(self.flag)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())
