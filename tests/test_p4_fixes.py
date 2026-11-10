"""P4 fixes (CPU/gloo): f5727c3 backend=deepspeed lr schedule, 8ac5cce describe_ckpt_error, 50e9389 async save failure
leaves no pending collective, 71b3724 save_final + scaling_sweep.rm_ckpt."""
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKEDS = Path(__file__).resolve().parent / "fakeds"      # CPU stand-in `deepspeed` package (optimizer is not a torch Optimizer)
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else os.environ.get("GLOO_SOCKET_IFNAME", "lo"),
       "TOKENIZERS_PARALLELISM": "false"}
ENV.pop("VLA_RUNS", None)
ENV.pop("VLA_CKPT", None)


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def _bc(data, runs, name, *extra, resume=False, env=None, nproc=2, timeout=420):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={nproc}",
           "--master_addr=127.0.0.1", f"--master_port={_port()}", "-m", "vla.train.bc", "--config", "configs/bc_tiny.yaml",
           *(["--resume"] if resume else []), f"data_dir={data}", f"runs_dir={runs}", f"name={name}", *extra]
    r = subprocess.run(cmd, cwd=ROOT, env=env or ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2500:]
    return r


def _metrics(runs, name):
    return [json.loads(l) for l in (Path(runs) / name / "metrics.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=4", "chunk_size=2",
                        "tasks=[lift]", "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


# ================================================================== 1) f5727c3 backend=deepspeed
DS_ARGS = ("backend=deepspeed", "steps=4", "log_every=1", "ckpt_every=100000", "export_final=false")


def _ds_env():
    return {**ENV, "PYTHONPATH": os.pathsep.join([str(FAKEDS), ENV.get("PYTHONPATH", "")]).rstrip(os.pathsep)}


def test_deepspeed_lr_follows_schedule(demos, tmp_path):
    from vla.train.bc import lr_at
    r = _bc(demos, tmp_path, "ds", *DS_ARGS, env=_ds_env(), nproc=1)    # 1 rank: 2 ranks interleave stdout lines
    want = [[1e-4 * lr_at(s, 4, 5), 1e-5 * lr_at(s, 4, 5)] for s in range(4)]       # both param groups
    # the lr the (fake) engine actually used at each optimizer step, per rank -- the real check: the logged metric
    # for deepspeed is computed from lr_at, not read from the optimizer
    used = [[float(a), float(b)] for a, b in re.findall(r"^FAKEDS lr (\S+) (\S+)$", r.stdout, re.M)]
    assert len(used) == 4, r.stdout[-1500:]                                             # one per optimizer step (accum=4)
    assert used == [[pytest.approx(x, rel=1e-5) for x in g] for g in want]
    m = [x for x in _metrics(tmp_path, "ds") if "lr" in x]
    assert [x["step"] for x in m] == [1, 2, 3, 4]
    assert [x["lr"] for x in m] == pytest.approx([w[0] for w in want], rel=1e-9)


def test_deepspeed_resume_ignored(demos, tmp_path):
    r = _bc(demos, tmp_path, "dsr", *DS_ARGS, resume=True, env=_ds_env())
    assert "--resume ignored: backend=deepspeed" in r.stdout + r.stderr
    assert not list(Path(tmp_path, "dsr").rglob("step_*"))                              # wrote no checkpoints


# ================================================================== 2) 8ac5cce describe_ckpt_error
def _frame(fn="filesystem.py", line=123, name="write_data"):
    return traceback.StackSummary.from_list([("train.py", 1, "main", "x"), (fn, line, name, "y")])


def test_describe_one_rank_enospc():
    from torch.distributed.checkpoint.api import CheckpointException
    from vla.train.checkpoint import describe_ckpt_error
    e = CheckpointException("dcp plumbing", {1: (OSError(28, "No space left on device"), _frame())})
    s = describe_ckpt_error(e)
    for part in ("ranks [1]", "OSError", "Errno 28", "No space left on device", "@ filesystem.py:123 in write_data"):
        assert part in s, (part, s)
    assert "dcp plumbing" not in s


def test_describe_groups_identical_causes():
    from torch.distributed.checkpoint.api import CheckpointException
    from vla.train.checkpoint import describe_ckpt_error
    e = CheckpointException("x", {1: (OSError(28, "No space left on device"), _frame()),
                                  0: (OSError(28, "No space left on device"), _frame())})
    s = describe_ckpt_error(e)
    assert s.count("ranks [") == 1 and "ranks [0, 1]" in s, s
    # different causes -> separate groups
    e2 = CheckpointException("x", {0: (OSError(28, "No space left on device"), _frame()),
                                   1: (PermissionError(13, "Permission denied"), _frame())})
    s2 = describe_ckpt_error(e2)
    assert "ranks [0]: OSError" in s2 and "ranks [1]: PermissionError" in s2, s2


def test_describe_plain_exception_cause_chain():
    from vla.train.checkpoint import describe_ckpt_error
    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as inner:
            raise RuntimeError("save failed") from inner
    except RuntimeError as e:
        s = describe_ckpt_error(e)
    assert s == "RuntimeError: save failed <- caused by OSError: [Errno 28] No space left on device", s
    assert describe_ckpt_error(ValueError("x")) == "ValueError: x"


# ================================================================== 3) 50e9389 async save failure, 2 ranks gloo
ASYNC = r'''
"""usage: async_fail.py <case> <workdir>; case in write_one | write_all | launch_one. Save at step 2 fails (injected)."""
import errno, os, sys, time
from pathlib import Path
import torch, torch.distributed as dist, torch.distributed.checkpoint as dcp, torch.multiprocessing as mp
CASE, WORK = sys.argv[1], Path(sys.argv[2])

class FailWriter(dcp.FileSystemWriter):
    def __init__(self, path, fail):
        super().__init__(path); self.fail = fail
    def write_data(self, plan, planner):
        if self.fail:
            time.sleep(0.3); raise OSError(errno.ENOSPC, "No space left on device (injected)")
        return super().write_data(plan, planner)

def worker(rank, world):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=f"file://{WORK}/rdzv", rank=rank, world_size=world)
    from vla.train import checkpoint as C
    orig, state = dcp.async_save, {"step": 0}
    def patched(sd, *, process_group=None, checkpoint_id=None, storage_writer=None, **kw):
        if state["step"] == 2:
            if CASE == "launch_one" and rank == world - 1:
                raise RuntimeError("injected staging failure on one rank (sync part of async_save)")
            if CASE in ("write_one", "write_all"):
                fail = CASE == "write_all" or rank == world - 1
                return orig(sd, process_group=process_group, storage_writer=FailWriter(checkpoint_id, fail), **kw)
        return orig(sd, process_group=process_group, checkpoint_id=checkpoint_id, storage_writer=storage_writer, **kw)
    C.dcp.async_save = patched
    ck = C.Checkpointer(WORK / "ckpt", "async", keep=2, timeout_min=0.5, log=lambda m: print(f"[r{rank}] {m}", flush=True))
    model = torch.nn.Linear(8, 8); opt = torch.optim.AdamW(model.parameters())
    for step in range(1, 9):
        state["step"] = step
        model(torch.randn(4, 8)).sum().backward(); opt.step(); opt.zero_grad()
        t = torch.ones(1); dist.all_reduce(t)          # the "training" collective
        assert t.item() == world
        print(f"[r{rank}] STEP {step}", flush=True)
        ck.poll()
        if step % 2 == 0 and step <= 6:
            ck.save(step, model, opt, {"step": step})
        time.sleep(0.2)
    ck.wait()
    print(f"[r{rank}] DONE latest={ck.latest()} n_failed={ck.n_failed}", flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    ctx = mp.spawn(worker, args=(2,), nprocs=2, join=False)
    t0 = time.time()
    try:
        while not ctx.join(timeout=2):
            if time.time() - t0 > 150:
                print("HANG: workers still running after 150 s", flush=True); [p.kill() for p in ctx.processes]; sys.exit(3)
    except Exception as e:
        print(f"WORKER FAILED: {e!r}"[:3000], flush=True); sys.exit(4)
    lat = WORK / "ckpt/latest"
    print("LATEST", lat.read_text().strip() if lat.exists() else None)
    print("DIRS", ",".join(sorted(p.name for p in (WORK / "ckpt").glob("step_*"))))
'''


@pytest.mark.parametrize("case", ["write_one", "write_all", "launch_one"])
def test_async_save_failure_training_continues(tmp_path, case):
    script = tmp_path / "async_fail.py"; script.write_text(ASYNC)
    r = subprocess.run([sys.executable, str(script), case, str(tmp_path / "w")], cwd=ROOT, env=ENV, capture_output=True,
                       text=True, timeout=240)
    out = r.stdout
    assert "HANG" not in out and r.returncode == 0, (r.returncode, out[-2500:], r.stderr[-2500:])
    for rank in (0, 1):
        assert [int(m) for m in re.findall(rf"\[r{rank}\] STEP (\d+)", out)] == list(range(1, 9)), out[-2000:]
        assert f"[r{rank}] DONE latest=step_000006 n_failed=1" in out, out[-2000:]
        assert re.search(rf"\[r{rank}\] !!! CHECKPOINT SAVE FAILED at step 2:", out), out[-2000:]
    assert "LATEST step_000006" in out and "DIRS step_000004,step_000006" in out, out[-1000:]
    # root cause reaches the log (8ac5cce), grouped by rank
    if case == "write_one":
        assert "ranks [1]: OSError: [Errno 28] No space left on device (injected)" in out
    elif case == "write_all":
        assert "ranks [0, 1]: OSError: [Errno 28] No space left on device (injected)" in out
    else:
        assert "[r1] !!! CHECKPOINT SAVE FAILED at step 2: RuntimeError: injected staging failure" in out
        assert "[r0] !!! CHECKPOINT SAVE FAILED at step 2: RuntimeError: async save launched here but failed to launch" in out


@pytest.mark.skipif(sys.platform == "darwin", reason="num_workers>0: bc collate is a closure, unpicklable under spawn (pre-existing)")
def test_dataloader_timeout_set_when_workers(demos, tmp_path):
    _bc(demos, tmp_path, "nw", "steps=2", "num_workers=1", "ckpt_every=100000", "export_final=false", "save_final=false")


def test_dataloader_timeout_wired():
    """Source-level: the DataLoader gets timeout=pg_timeout_min*60 only when num_workers > 0 (runtime test is Linux-only)."""
    src = (ROOT / "src/vla/train/bc.py").read_text()
    assert "timeout=cfg.pg_timeout_min * 60 if cfg.num_workers > 0 else 0" in src


# ================================================================== 4) 71b3724 save_final + rm_ckpt
@pytest.mark.parametrize("save_final", [False, True])
def test_save_final_flag(demos, tmp_path, save_final):
    name = f"sf{int(save_final)}"
    _bc(demos, tmp_path, name, "steps=3", "ckpt_every=100000", "export_final=false", f"save_final={str(save_final).lower()}")
    ck = tmp_path / name / "ckpt"
    steps = sorted(p.name for p in ck.glob("step_*")) if ck.exists() else []
    if save_final:
        assert steps == ["step_000003"] and (ck / "latest").read_text().strip() == "step_000003"
    else:
        assert steps == [] and not (ck / "latest").exists()
    assert (tmp_path / name / "metrics.jsonl").exists()


def _sweep():
    spec = importlib.util.spec_from_file_location("scaling_sweep", ROOT / "scripts/scaling_sweep.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _tree(root, name):
    (root / name / "ckpt" / "step_000003").mkdir(parents=True)
    (root / name / "ckpt" / "step_000003" / "x.distcp").write_bytes(b"0" * 10)
    (root / name / "metrics.jsonl").write_text("{}\n")


def test_rm_ckpt_both_locations_keeps_metrics(tmp_path, monkeypatch):
    runs, vck = tmp_path / "runs", tmp_path / "vck"
    _tree(runs, "scale_a"); _tree(runs, "scale_ab")                     # scale_ab: other run, name shares a prefix
    (vck / "scale_a" / "ckpt" / "step_000003").mkdir(parents=True); (vck / "scale_a" / "best.pt").write_bytes(b"b")
    (vck / "scale_ab" / "ckpt").mkdir(parents=True)
    monkeypatch.setenv("VLA_CKPT", str(vck))
    _sweep().rm_ckpt("scale_a", str(runs))
    assert not (runs / "scale_a" / "ckpt").exists() and not (vck / "scale_a").exists()
    assert (runs / "scale_a" / "metrics.jsonl").read_text() == "{}\n"
    assert (runs / "scale_ab" / "ckpt" / "step_000003").exists() and (vck / "scale_ab" / "ckpt").exists()


def test_rm_ckpt_without_vla_ckpt_and_missing_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("VLA_CKPT", raising=False)
    runs = tmp_path / "runs"; _tree(runs, "r")
    m = _sweep()
    m.rm_ckpt("r", str(runs))
    assert not (runs / "r" / "ckpt").exists() and (runs / "r" / "metrics.jsonl").exists()
    m.rm_ckpt("r", str(runs))                                           # already gone: no error
    m.rm_ckpt("never_ran", str(tmp_path / "nope"))


def test_scaling_sweep_cmd_passes_save_final_false_and_cleans_on_failure(tmp_path, monkeypatch):
    """run_one must pass save_final=false and call rm_ckpt even when the training subprocess raises."""
    m = _sweep()
    seen, removed = {}, []
    monkeypatch.setattr(m, "snapshot", lambda **k: {})
    monkeypatch.setattr(m, "quiet_wait", lambda *a, **k: {"quiet": True})
    monkeypatch.setattr(m, "rm_ckpt", lambda name, runs_dir: removed.append((name, runs_dir)))

    def boom(cmd, **k):
        seen["cmd"] = cmd
        raise KeyboardInterrupt
    monkeypatch.setattr(m.subprocess, "run", boom)
    cfg = {"backend": "fsdp", "n_gpu": 1, "act_ckpt": False, "rep": 0}
    with pytest.raises(KeyboardInterrupt):
        m.run_one(cfg, [0], [], _port(), str(tmp_path), str(tmp_path), False)
    assert "save_final=false" in seen["cmd"] and "ckpt_every=100000" in seen["cmd"]
    assert removed == [("scale_fsdp_1g_ac0_r0", str(tmp_path))]


# ================================================================== 5) aaecd78 MFU under ZeRO-3 (pre-wrap param count)
def test_flops_per_sample_uses_prewrap_count_when_params_partitioned():
    import torch
    from vla.model.vla import VLA, VLAConfig, count_params
    from vla.train.bc import flops_per_sample
    from vla.utils.config import load_config
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), []))
    n = count_params(m)["total"]
    f0 = flops_per_sample(m, 50)
    assert n > 0 and f0 == flops_per_sample(m, 50, n)
    for p in m.parameters():                                   # ZeRO-3 stand-in: partitioned params have numel()==0
        p.data = torch.empty(0)
    assert count_params(m)["total"] == 0
    assert flops_per_sample(m, 50, n) == f0                    # fixed path: exact
    c = m.llm.config
    collapsed = flops_per_sample(m, 50)                        # old path: the 6*N term vanishes
    assert collapsed == 50 * 12 * c.num_hidden_layers * c.hidden_size * 50 and collapsed < 0.5 * f0


def _params_line(out):
    m = re.search(r"params=([\d.]+)M", out)
    assert m, out[-1500:]
    return float(m.group(1))


def _implied_flops(runs, name, world=1, peak=312.0, gb=8):
    """flops/sample recovered from the logged mfu: mfu = flops_s * global_batch / dt / (world * peak * 1e12)."""
    return [x["mfu"] * world * peak * 1e12 * x["step_time"] / gb for x in _metrics(runs, name) if "mfu" in x]


def test_deepspeed_mfu_and_params_match_fsdp(demos, tmp_path):
    base = ("steps=3", "log_every=1", "ckpt_every=100000", "export_final=false", "save_final=false")
    rf = _bc(demos, tmp_path, "mfu_fsdp", "backend=fsdp", *base, nproc=1)
    rd = _bc(demos, tmp_path, "mfu_ds", "backend=deepspeed", *base, nproc=1,
             env={**_ds_env(), "FAKEDS_PARTITION": "1"})               # params numel()==0 after initialize
    assert _params_line(rd.stdout) == _params_line(rf.stdout) > 0
    ff, fd = _implied_flops(tmp_path, "mfu_fsdp"), _implied_flops(tmp_path, "mfu_ds")
    assert len(ff) == len(fd) == 3 and all(x > 0 for x in fd)
    assert fd == pytest.approx(ff, rel=1e-6)                           # same flops/sample, not the attention-only term
