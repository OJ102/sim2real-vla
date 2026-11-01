"""bc.py stop_at (plan step 5): stop, save DCP + stop_<step>.pt, resume with the same lr schedule; keep_best export."""
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ENV = {**__import__("os").environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def run(data, runs, name, *extra, resume=False):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={_port()}", "-m", "vla.train.bc", "--config", "configs/bc_tiny.yaml"]
    if resume:
        cmd.append("--resume")
    r = subprocess.run(cmd + [f"data_dir={data}", f"runs_dir={runs}", f"name={name}", "log_every=1", "ckpt_every=100",
                              *extra], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=420)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return r


def recs(runs, name):
    out = {}
    for l in (Path(runs) / name / "metrics.jsonl").read_text().splitlines():
        r = json.loads(l)
        if "loss" in r:
            out[r["step"]] = r
    return out


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("so_demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=4", "chunk_size=2", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0
    return out


def test_stop_at_then_resume_keeps_schedule(demos, tmp_path):
    common = ["steps=8", "warmup=3"]
    run(demos, tmp_path, "ref", *common)
    run(demos, tmp_path, "stp", *common, "stop_at=3")
    d = tmp_path / "stp"
    assert (d / "stop_3.pt").is_file() and (d / "ckpt" / "latest").read_text().strip() == "step_000003"
    assert not (d / "final.pt").exists() and max(recs(tmp_path, "stp")) == 3
    sd = torch.load(d / "stop_3.pt", map_location="cpu", weights_only=True)
    assert any(k.startswith("llm.") for k in sd) and not any("exp_avg" in k for k in sd)
    r = run(demos, tmp_path, "stp", *common, resume=True)
    assert "resume_from=3" in r.stdout
    ref, got = recs(tmp_path, "ref"), recs(tmp_path, "stp")
    assert max(got) == 8 and (d / "final.pt").is_file()
    for s in (4, 5, 8):
        assert got[s]["lr"] == pytest.approx(ref[s]["lr"], rel=1e-9), (s, got[s]["lr"], ref[s]["lr"])     # same cosine continuation
        assert got[s]["loss"] == ref[s]["loss"], s


def test_stop_at_zero_runs_to_completion_and_keep_best(demos, tmp_path):
    run(demos, tmp_path, "full", "steps=10", "val_every=5", "val_batches=1", "keep_best=true")
    d = tmp_path / "full"
    assert (d / "final.pt").is_file() and not list(d.glob("stop_*.pt"))
    assert (d / "best.pt").is_file()
    sd = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
    assert any(k.startswith("llm.") for k in sd)
    v = [r for r in recs(tmp_path, "full").values() if "val_main_loss" in r]
    assert v and "val_main_loss" in v[0] and "val_ground_loss" not in v[0]          # no head -> no ground metrics


def test_grounded_run_logs_ground_metrics_and_oversamples(demos, tmp_path):
    run(demos, tmp_path, "gr", "steps=10", "val_every=5", "val_batches=1", "ground_weight=0.2", "early_k=4", "early_w=3")
    rs = list(recs(tmp_path, "gr").values())
    assert all("main_loss" in r and "ground_loss" in r and r["ground_loss"] >= 0 for r in rs)
    v = [r for r in rs if "val_main_loss" in r]
    assert v and all(k in v[0] for k in ("val_ground_loss", "val_ground_err_cm", "val_ground_err_cm_t0"))
    assert all(r["loss"] == pytest.approx(r["main_loss"] + 0.2 * r["ground_loss"], rel=1e-3) for r in rs)
    assert (tmp_path / "gr" / "final.pt").is_file() and (tmp_path / "gr" / "best.pt").is_file()


def test_vla_ckpt_redirects_stop_and_best(demos, tmp_path):
    import os
    ck = tmp_path / "ck"
    env_old = ENV.get("VLA_CKPT")
    ENV["VLA_CKPT"] = str(ck)
    try:
        run(demos, tmp_path, "red", "steps=6", "stop_at=3", "val_every=3", "val_batches=1")
        d = tmp_path / "red"
        assert (ck / "red" / "stop_3.pt").is_file() and not (d / "stop_3.pt").exists() and not list(d.glob("stop_*.pt"))
        assert (ck / "red" / "ckpt" / "latest").read_text().strip() == "step_000003"
        run(demos, tmp_path, "red2", "steps=8", "val_every=4", "val_batches=1")
        assert (ck / "red2" / "best.pt").is_file() and (tmp_path / "red2" / "best.pt").is_file()      # copy in the run dir
        assert (tmp_path / "red2" / "final.pt").is_file() and not (ck / "red2" / "final.pt").exists()  # final.pt stays home
        a = torch.load(ck / "red2" / "best.pt", map_location="cpu", weights_only=True)
        b = torch.load(tmp_path / "red2" / "best.pt", map_location="cpu", weights_only=True)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        # stop_at run: best.pt copy lands in the run dir at stop time as well
        assert (tmp_path / "red" / "best.pt").is_file() or not (ck / "red" / "best.pt").exists()
    finally:
        ENV.pop("VLA_CKPT", None)
        if env_old:
            ENV["VLA_CKPT"] = env_old
