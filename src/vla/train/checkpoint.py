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
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException  # derives from BaseException, not Exception!
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict


def describe_ckpt_error(e: BaseException, max_msg: int = 4000) -> str:
    """Root cause of a (DCP) checkpoint failure on one line. str(CheckpointException) starts with DCP's own collective
    plumbing (utils.py all_reduce -> map_fun), which hid the real error behind a 300-char cut. This lists, per distinct
    cause, the ranks that hit it, the exception type + full message (cause chain included) and the last frame."""
    if isinstance(e, CheckpointException):
        groups: dict[str, list] = {}
        for r, f in sorted(e.failures.items()):
            exc, trace = (f if (isinstance(f, tuple) and len(f) == 2 and isinstance(f[0], BaseException))
                          else (f, None))
            if not isinstance(exc, BaseException):
                groups.setdefault(repr(exc)[:max_msg], []).append(r)
                continue
            last = trace[-1] if trace else (traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None)
            where = f" @ {last.filename}:{last.lineno} in {last.name}" if last else ""
            groups.setdefault(describe_ckpt_error(exc, max_msg) + where, []).append(r)
        return "CheckpointException root cause(s): " + " | ".join(f"ranks {rs}: {msg}" for msg, rs in groups.items())
    parts, seen, cur = [], set(), e
    while cur is not None and id(cur) not in seen and len(parts) < 5:  # explicit cause chain (raise X from Y), or context
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {str(cur)[:max_msg]}")
        cur = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)
    return " <- caused by ".join(parts)


def step_name(step: int) -> str:
    return f"step_{step:06d}"


class SlowWriter(dcp.FileSystemWriter):
    """Test aid: DCP writer that sleeps before writing, to force long overlap between an in-flight async save and the
    training loop (reproduces collective-ordering bugs on CPU)."""

    def __init__(self, path, delay_s: float, **kw):
        super().__init__(path, **kw)
        self.delay_s = delay_s

    def prepare_local_plan(self, plan):
        # stagger ranks so DCP's planning collectives are issued at different times on different ranks
        if dist.get_rank() == dist.get_world_size() - 1:
            time.sleep(self.delay_s)
        return super().prepare_local_plan(plan)

    def write_data(self, plan, planner):
        time.sleep(self.delay_s)
        return super().write_data(plan, planner)


class Checkpointer:
    def __init__(self, ckpt_dir: str | os.PathLike, mode: str = "sync", keep: int = 2, mirror_to: str | None = None,
                 timeout_min: float = 10.0, slow_s: float = 0.0, shared_pg_bug: bool = False, log=None):
        assert mode in ("sync", "async")
        self.dir, self.mode, self.keep, self.mirror_to = Path(ckpt_dir), mode, keep, mirror_to
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self._pending = None  # (future, step)
        # async_save stages to CPU and writes in a thread; it needs a CPU-capable group (gloo) even in nccl runs
        self.slow_s = slow_s
        self.log = log or (lambda m: print(m, flush=True))
        self.n_failed = 0
        shared_pg_bug = shared_pg_bug or os.environ.get("VLA_CKPT_SHARED_PG") == "1"  # repro of the old deadlock
        td = timedelta(minutes=timeout_min)
        # `_pg` is used ONLY by DCP's background save thread. `_ctl` carries our own consensus/barrier collectives
        # from the main thread: sharing one gloo group between two threads interleaves collectives and deadlocks.
        self._pg = dist.new_group(backend="gloo", timeout=td) if (dist.is_initialized() and mode == "async") else None
        self._ctl = self._pg if shared_pg_bug else (
            dist.new_group(backend="gloo", timeout=td) if dist.is_initialized() else None)
        self.last_stall_s = 0.0

    # ------------------------------------------------------------------ save
    def save(self, step: int, model, optimizer, extra: dict) -> float:
        """Returns the wall time this call blocked the training loop (the 'stall'). A failed save (disk full, I/O
        error) is logged loudly, never advances `latest`, and never raises: training continues."""
        t0 = time.perf_counter()
        self._finalize_pending(block=True)  # at most one save in flight
        path = str(self.dir / step_name(step))
        try:
            msd, osd = get_state_dict(model, optimizer)
            state = {"model": msd, "optim": osd, "extra": extra}
            if self.mode == "sync":
                dcp.save(state, checkpoint_id=path)
                self._publish_if_all_ok(step, True)
            else:
                kw = {"storage_writer": SlowWriter(path, self.slow_s)} if self.slow_s else {"checkpoint_id": path}
                fut = dcp.async_save(state, process_group=self._pg, **kw)
                self._pending = (fut, step)
        except (Exception, CheckpointException) as e:  # noqa: BLE001
            self._failed(step, e)
            if self.mode == "sync":
                self._publish_if_all_ok(step, False)
        self.last_stall_s = time.perf_counter() - t0
        return self.last_stall_s

    def _failed(self, step: int, e: BaseException) -> None:
        self.n_failed += 1
        self.log(f"!!! CHECKPOINT SAVE FAILED at step {step}: {describe_ckpt_error(e)} (continuing; `latest` unchanged)")
        # full detail, untruncated, on stderr (one copy per rank): local traceback + DCP's per-rank tracebacks
        sys.stderr.write(f"[rank {self.rank}] checkpoint step {step} failure detail:\n"
                         + "".join(traceback.format_exception(type(e), e, e.__traceback__)) + str(e) + "\n")
        sys.stderr.flush()

    def _publish_if_all_ok(self, step: int, ok: bool) -> None:
        """All ranks agree (min over ok flags); only then publish `latest`. Otherwise clean the partial dir."""
        flag = torch.tensor([int(ok)])
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._ctl)
        if flag.item():
            self._publish(step, barrier=False)
        elif self.rank == 0:
            shutil.rmtree(self.dir / step_name(step), ignore_errors=True)

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
                dist.all_reduce(done, op=dist.ReduceOp.MIN, group=self._ctl)
            if not done.item():
                return
        ok = True
        try:
            fut.result()
        except (Exception, CheckpointException) as e:  # noqa: BLE001
            ok = False
            self._failed(step, e)
        self._pending = None
        self._publish_if_all_ok(step, ok)  # all-reduce on the control group doubles as the "everyone finished" barrier

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
    """Model weights only (no optimizer): full state dict gathered to rank 0 and saved as one .pt file.
    Atomic: written to a hidden tmp file in the same directory, fsynced, then os.replace'd onto `path`, so a crash or
    ENOSPC mid-write never leaves a truncated `path` (an existing `path` stays intact until the new one is complete)."""
    sd, _ = get_state_dict(model, [], options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    err = None
    if not dist.is_initialized() or dist.get_rank(group) == 0:
        path = Path(path)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        try:
            with open(tmp, "wb") as f:
                torch.save({k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except (OSError, RuntimeError) as e:  # torch.save raises RuntimeError (PyTorchStreamWriter) for open/ENOSPC; tell every rank so they all raise together instead of hanging
            err = repr(e)
            try:
                tmp.unlink(missing_ok=True)  # don't leave a partial tmp file eating the disk
            except OSError:
                pass
    if dist.is_initialized():
        objs = [err]
        dist.broadcast_object_list(objs, src=dist.get_global_rank(group, 0) if group is not None else 0, group=group)
        err = objs[0]
    if err:
        raise OSError(err)


class PreemptFlag:
    """SIGTERM/SIGUSR1 set a flag; the loop checks it at step boundaries (all ranks agree via all-reduce).
    A second signal while the flag is already set (rank stuck in a collective, so the loop never reaches the check)
    hard-exits with 143."""

    def __init__(self):
        self.flag = False

        def handler(*_):
            if self.flag:
                os._exit(143)
            self.flag = True

        for s in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(s, handler)

    def should_stop(self, device) -> bool:
        if not dist.is_initialized():
            return self.flag
        t = torch.tensor([int(self.flag)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())
