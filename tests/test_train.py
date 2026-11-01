"""P4 tests (CPU, gloo, tiny model): BC trainer end to end via torchrun, checkpointing, resume, preemption."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else os.environ.get("GLOO_SOCKET_IFNAME", "lo"),
       "TOKENIZERS_PARALLELISM": "false"}
ENV.pop("VLA_RUNS", None)


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cmd(data, runs, name, extra=(), resume=False, nproc=2):
    c = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={nproc}",
         "--master_addr=127.0.0.1", f"--master_port={_port()}", "-m", "vla.train.bc", "--config", "configs/bc_tiny.yaml"]
    if resume:
        c.append("--resume")
    return c + [f"data_dir={data}", f"runs_dir={runs}", f"name={name}", "log_every=5", *extra]


def run(data, runs, name, extra=(), resume=False, nproc=2, timeout=420):
    return subprocess.run(_cmd(data, runs, name, extra, resume, nproc), cwd=ROOT, env=ENV, capture_output=True,
                          text=True, timeout=timeout)


def metrics(runs, name):
    p = Path(runs) / name / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def losses(runs, name):
    out = {}
    for r in metrics(runs, name):
        if "loss" in r:
            out[r["step"]] = r["loss"]      # later records (post-resume) overwrite
    return out


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=4", "chunk_size=2",
                        "tasks=[lift]", "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


def _ok(r):
    assert r.returncode == 0, (r.stdout[-1500:] + r.stderr[-2500:])


# ------------------------------------------------------------------ 1) basic run
def test_basic_run_artifacts(demos, runs):
    r = run(demos, runs, "basic", ["steps=20"])
    _ok(r)
    d = Path(runs) / "basic"
    recs = [m for m in metrics(runs, "basic") if "loss" in m]
    assert [m["step"] for m in recs] == [5, 10, 15, 20]
    for m in recs:
        for k in ("loss", "tok_acc", "grad_norm", "step_time", "samples_per_s", "mfu", "mem_alloc_gb_per_rank", "ckpt_stall_s"):
            assert k in m, k
        assert all(map(lambda x: x == x and abs(x) < 1e9, [m["loss"], m["grad_norm"], m["samples_per_s"]]))
        assert m["step_time"] > 0 and m["samples_per_s"] > 0 and 0 <= m["mfu"] < 1
        assert len(m["mem_alloc_gb_per_rank"]) == 2
    assert recs[-1]["loss"] < recs[0]["loss"] * 1.5          # not diverging
    assert any("val_loss" in m for m in recs)                # val_every=10
    assert (d / "config.yaml").is_file() and (d / "tb").exists()
    ck = d / "ckpt"
    assert (ck / "step_000010").is_dir() and (ck / "step_000020").is_dir()
    assert (ck / "latest").read_text().strip() == "step_000020"
    assert not list(ck.glob("*.tmp")) and (recs[1]["ckpt_stall_s"] > 0)
    sd = torch.load(d / "final.pt", map_location="cpu", weights_only=True)
    assert isinstance(sd, dict) and sd and all(isinstance(v, torch.Tensor) for v in sd.values())
    assert not any(("optim" in k or "exp_avg" in k or k in ("step", "state")) for k in sd), "optimizer keys in final.pt"
    assert any(k.startswith("llm.") for k in sd) and any(k.startswith("projector.") for k in sd)
    assert "world=2" in r.stdout and "accum=2" in r.stdout


# ------------------------------------------------------------------ 2) kill / resume exactness
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_kill_and_resume_matches_uninterrupted(demos, runs, mode):
    common = ["steps=30", "ckpt_every=10", f"ckpt_mode={mode}"]
    _ok(run(demos, runs, f"ref_{mode}", common))
    ref = losses(runs, f"ref_{mode}")
    assert set(ref) == {5, 10, 15, 20, 25, 30}

    name = f"kill_{mode}"
    r = run(demos, runs, name, common + ["exit_after_step=17"])
    assert r.returncode != 0
    ck = Path(runs) / name / "ckpt"
    assert (ck / "latest").read_text().strip() == "step_000010"
    assert (ck / "step_000010").is_dir()
    assert not (Path(runs) / name / "final.pt").exists()
    r = run(demos, runs, name, common, resume=True)
    _ok(r)
    assert "resume_from=10" in r.stdout
    got = losses(runs, name)
    for s in (20, 25, 30):
        assert got[s] == ref[s], (mode, s, got[s], ref[s])        # bitwise-equal losses after resume
    assert (ck / "latest").read_text().strip() == "step_000030" and (Path(runs) / name / "final.pt").is_file()
    assert any("event_resume_from" in m for m in metrics(runs, name))


def test_resume_without_checkpoint_starts_fresh(demos, runs):
    r = run(demos, runs, "fresh_resume", ["steps=10", "ckpt_every=10"], resume=True)
    _ok(r)
    assert "resume_from" not in r.stdout and 10 in losses(runs, "fresh_resume")


# ------------------------------------------------------------------ 3) keep=N
def test_keep_two_newest(demos, runs):
    _ok(run(demos, runs, "keep2", ["steps=30", "ckpt_every=5"]))
    ck = Path(runs) / "keep2" / "ckpt"
    assert sorted(p.name for p in ck.glob("step_*")) == ["step_000025", "step_000030"]
    assert (ck / "latest").read_text().strip() == "step_000030"


def test_keep_one_async(demos, runs):
    _ok(run(demos, runs, "keep1", ["steps=20", "ckpt_every=5", "ckpt_mode=async", "ckpt_keep=1"]))
    ck = Path(runs) / "keep1" / "ckpt"
    assert [p.name for p in ck.glob("step_*")] == ["step_000020"] and (ck / "latest").read_text().strip() == "step_000020"


# ------------------------------------------------------------------ 4) async: latest only after complete (unit, no dist)
def test_async_latest_only_after_future_completes(tmp_path, monkeypatch):
    import concurrent.futures as cf

    import torch.distributed.checkpoint as dcp

    from vla.train.checkpoint import Checkpointer

    m = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters())
    m(torch.randn(2, 4)).sum().backward(); opt.step()
    pending = {}

    def fake_async_save(state, checkpoint_id=None, process_group=None, **kw):
        fut = cf.Future()
        pending[checkpoint_id] = (fut, state)
        Path(checkpoint_id).mkdir(parents=True)             # partial dir exists before completion
        return fut

    monkeypatch.setattr(dcp, "async_save", fake_async_save)
    ck = Checkpointer(tmp_path, "async", keep=2)
    ck.save(10, m, opt, {"step": 10})
    assert (tmp_path / "step_000010").is_dir() and ck.latest() is None     # partial dir NOT advertised
    ck.poll(); assert ck.latest() is None
    pending[str(tmp_path / "step_000010")][0].set_result(None)
    ck.poll()
    assert ck.latest() == "step_000010"
    # a new save first finalizes the previous one, and latest never moves ahead of completion
    ck.save(20, m, opt, {"step": 20})
    assert ck.latest() == "step_000010"
    pending[str(tmp_path / "step_000020")][0].set_result(None)
    ck.wait(); assert ck.latest() == "step_000020"
    # failed save must not publish
    ck.save(30, m, opt, {"step": 30})
    pending[str(tmp_path / "step_000030")][0].set_exception(RuntimeError("disk full"))
    with pytest.raises(RuntimeError):
        ck.wait()
    assert ck.latest() == "step_000020"


def test_sync_checkpoint_roundtrip_no_dist(tmp_path):
    from vla.train.checkpoint import Checkpointer

    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4); opt = torch.optim.AdamW(m.parameters())
    m(torch.randn(2, 4)).sum().backward(); opt.step()
    ck = Checkpointer(tmp_path, "sync", keep=2)
    assert ck.load_latest(m, opt, {"step": 0}) is None
    ck.save(5, m, opt, {"step": 5, "note": 1})
    w = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.zero_()
    ex = ck.load_latest(m, opt, {"step": 0, "note": 0})
    assert ex["step"] == 5 and torch.equal(m.weight, w)
    for s in (6, 7, 8):
        ck.save(s, m, opt, {"step": s})
    assert sorted(p.name for p in tmp_path.glob("step_*")) == ["step_000007", "step_000008"] and ck.latest() == "step_000008"


# ------------------------------------------------------------------ 5) SIGTERM
def test_sigterm_saves_and_resumes(demos, runs):
    name = "preempt"
    p = subprocess.Popen(_cmd(demos, runs, name, ["steps=4000", "ckpt_every=100000", "ckpt_mode=sync"]), cwd=ROOT,
                         env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 240 and not any(m.get("step", 0) >= 10 for m in metrics(runs, name)):
            assert p.poll() is None, p.stdout.read()
            time.sleep(0.5)
        else:
            assert any(m.get("step", 0) >= 10 for m in metrics(runs, name)), "trainer never reached step 10"
        os.killpg(p.pid, signal.SIGTERM)                    # like a scheduler: whole process group
        out, _ = p.communicate(timeout=120)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    # torchrun's agent itself exits non-zero when it receives SIGTERM; the contract is the workers' behaviour
    line = [l for l in out.splitlines() if l.startswith("preempted: saved step")]
    assert line, out[-2500:]
    n = int(line[0].split("step ")[1].split(",")[0])
    ck = Path(runs) / name / "ckpt"
    assert (ck / "latest").read_text().strip() == f"step_{n:06d}" and (ck / f"step_{n:06d}").is_dir()
    assert not (Path(runs) / name / "final.pt").exists()    # preemption is not completion
    r = run(demos, runs, name, [f"steps={n + 5}", "ckpt_every=100000"], resume=True)
    _ok(r)
    assert f"resume_from={n}" in r.stdout and (Path(runs) / name / "final.pt").is_file()


# ------------------------------------------------------------------ 6) config validation
def test_batch_divisibility_assertion(demos, runs):
    r = run(demos, runs, "badbatch", ["global_batch=7", "steps=2"])
    assert r.returncode != 0 and "global_batch must divide by micro_batch*world" in r.stderr
    r = run(demos, runs, "badbatch2", ["global_batch=6", "micro_batch=2", "steps=2"])        # 6 % (2*2) != 0
    assert r.returncode != 0 and "global_batch must divide" in r.stderr
    r = run(demos, runs, "badbackend", ["backend=zero9", "steps=2"])
    assert r.returncode != 0


# ------------------------------------------------------------------ 7) schedule / flops
def test_lr_schedule():
    from vla.train.bc import lr_at

    assert lr_at(0, 100, 10) == pytest.approx(0.1) and lr_at(9, 100, 10) == pytest.approx(1.0)
    assert all(lr_at(s + 1, 100, 10) > lr_at(s, 100, 10) for s in range(9))              # linear warmup
    assert lr_at(10, 100, 10) == pytest.approx(1.0)                                        # cosine starts at 1
    assert lr_at(55, 100, 10) == pytest.approx(0.5, abs=0.02)                              # midpoint ~0.5
    assert lr_at(100, 100, 10) == pytest.approx(0.0, abs=1e-9) and lr_at(500, 100, 10) == pytest.approx(0.0, abs=1e-9)
    xs = [lr_at(s, 100, 10) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:])) and min(xs) >= 0
    assert lr_at(0, 10, 0 + 1) == 1.0 and lr_at(3, 5, 5) == pytest.approx(0.8)


def test_flops_per_sample_formula():
    from vla.model.vla import VLA, VLAConfig, count_params
    from vla.train.bc import flops_per_sample
    from vla.utils.config import load_config

    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), []))
    n, c = count_params(m)["total"], m.llm.config
    for S in (10, 50):
        assert flops_per_sample(m, S) == pytest.approx(S * (6 * n + 12 * c.num_hidden_layers * c.hidden_size * S))
    assert flops_per_sample(m, 20) > 2 * flops_per_sample(m, 10) * 0.99      # superlinear (attention term)


# ------------------------------------------------------------------ 8) ddp / single process / flow
def test_ddp_backend_runs(demos, runs):
    _ok(run(demos, runs, "ddp", ["backend=ddp", "steps=10", "ckpt_every=10"]))
    recs = [m for m in metrics(runs, "ddp") if "loss" in m]
    assert [m["step"] for m in recs] == [5, 10] and (Path(runs) / "ddp" / "ckpt" / "latest").read_text().strip() == "step_000010"


def test_flow_head_trains_fsdp(demos, runs):
    _ok(run(demos, runs, "flowrun", ["head=flow", "steps=10", "ckpt_every=10"]))
    assert all(m["loss"] == m["loss"] for m in metrics(runs, "flowrun") if "loss" in m)


def test_ddp_and_fsdp_same_first_loss(demos, runs):
    """Same seed/data/init: step-5 loss must agree between DDP and FSDP2 (up to float noise)."""
    _ok(run(demos, runs, "cmp_fsdp", ["steps=5", "ckpt_every=100"]))
    _ok(run(demos, runs, "cmp_ddp", ["steps=5", "ckpt_every=100", "backend=ddp"]))
    a, b = losses(runs, "cmp_fsdp")[5], losses(runs, "cmp_ddp")[5]
    assert a == pytest.approx(b, rel=1e-3), (a, b)
