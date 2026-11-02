"""bc.py warm start (init_from): shape-filtered strict=False load of a full-state .pt, used to start flow from a disc ckpt."""
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vla.model.vla import VLA, VLAConfig
from vla.train.bc import warm_start
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ENV = {**__import__("os").environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}


def tiny(head, gw, seed=0):
    torch.manual_seed(seed)
    return VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), [f"head={head}", f"ground_weight={gw}"]))


@pytest.fixture(scope="module")
def disc_pt(tmp_path_factory):
    d = tmp_path_factory.mktemp("init")
    m = tiny("discrete", 0.2, seed=11)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))                   # make it differ from any fresh init
    path = d / "disc.pt"
    torch.save({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in m.state_dict().items()}, path)   # like export_final
    return path


# ------------------------------------------------------------------ (1) unit
def test_warm_start_disc_into_flow(disc_pt):
    sd = torch.load(disc_pt, weights_only=True)
    m = tiny("flow", 0.01)
    fresh = tiny("flow", 0.01)                                    # same seed -> identical fresh init
    msgs = []
    info = warm_start(m, str(disc_pt), msgs.append)
    dit_n = sum(1 for k in m.state_dict() if k.startswith("dit."))
    assert dit_n > 0 and info["left_at_init_prefixes"] == {"dit": dit_n}
    assert info["n_skipped_shape"] == 0 and info["n_skipped_missing"] == 0
    assert info["n_loaded"] == len(sd) == len(m.state_dict()) - dit_n
    cur, ref = m.state_dict(), fresh.state_dict()
    for k, v in sd.items():
        want = v.to(cur[k].dtype) if v.is_floating_point() else v
        assert torch.equal(cur[k], want), k                        # shared keys equal the .pt, cast to the model dtype
        assert cur[k].dtype == ref[k].dtype
    assert any(not torch.equal(cur[k], ref[k]) for k in sd if cur[k].is_floating_point())     # they did change from init
    for k in cur:
        if k.startswith("dit."):
            assert torch.equal(cur[k], ref[k]), k                  # DiT untouched
    assert msgs and msgs[0].startswith(f"init_from {disc_pt}: loaded {info['n_loaded']}")


def test_warm_start_counts_shape_mismatch_and_unknown_keys(disc_pt, tmp_path):
    sd = torch.load(disc_pt, weights_only=True)
    emb_keys = [k for k in sd if "embed_tokens" in k or k.endswith("lm_head.weight")]
    assert emb_keys
    for k in emb_keys:
        sd[k] = sd[k][:-8]                                          # tamper: fewer action rows
    sd["bogus.extra.weight"] = torch.zeros(3); sd["dit_old.w"] = torch.zeros(2)
    p = tmp_path / "t.pt"; torch.save(sd, p)
    m = tiny("flow", 0.01); fresh = tiny("flow", 0.01)
    info = warm_start(m, str(p), lambda s: None)
    assert info["n_skipped_shape"] == len(emb_keys) and info["n_skipped_missing"] == 2
    assert info["n_loaded"] == len(torch.load(disc_pt, weights_only=True)) - len(emb_keys)
    for k in emb_keys:                                              # skipped params keep their fresh init
        assert torch.equal(m.state_dict()[k], fresh.state_dict()[k])
    assert info["left_at_init_prefixes"].get("llm") == len(emb_keys) and info["left_at_init_prefixes"]["dit"] > 0
    # loading does not crash and the model still runs
    assert all(torch.isfinite(p_).all() for p_ in m.parameters())


def test_warm_start_same_arch_roundtrip_is_exact(tmp_path):
    a, b = tiny("discrete", 0.2, 1), tiny("discrete", 0.2, 2)
    torch.save(a.state_dict(), tmp_path / "a.pt")
    info = warm_start(b, str(tmp_path / "a.pt"), lambda s: None)
    assert info["left_at_init_prefixes"] == {} and info["n_skipped_shape"] == info["n_skipped_missing"] == 0
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# ------------------------------------------------------------------ (2)-(4) bc.py end to end
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("if_demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=4", "chunk_size=2", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0
    return out


def run(demos, runs, name, *extra, resume=False):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={port()}", "-m", "vla.train.bc", "--config", "configs/bc_tiny.yaml"] + (["--resume"] if resume else [])
    r = subprocess.run(cmd + [f"data_dir={demos}", f"runs_dir={runs}", f"name={name}", "log_every=1", "ckpt_every=3", "pg_timeout_min=2",
                              *extra], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=420)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return r


def losses(runs, name):
    return {json.loads(l)["step"]: json.loads(l)["loss"] for l in (Path(runs) / name / "metrics.jsonl").read_text().splitlines() if '"loss"' in l}


def test_bc_init_from_end_to_end(demos, disc_pt, tmp_path):
    r = run(demos, tmp_path, "X", "head=flow", "ground_weight=0.01", "steps=6", f"init_from={disc_pt}")
    line = [l for l in r.stdout.splitlines() if "init_from " in l and "loaded" in l]
    assert line and str(disc_pt) in line[0] and "left at init: {'dit':" in line[0], r.stdout[-1500:]
    assert (tmp_path / "X" / "final.pt").is_file() and 6 in losses(tmp_path, "X")


def test_resume_ignores_init_from(demos, disc_pt, tmp_path):
    run(demos, tmp_path, "R", "head=flow", "ground_weight=0.01", "steps=6")
    r = run(demos, tmp_path, "R", "head=flow", "ground_weight=0.01", "steps=9", f"init_from={disc_pt}", resume=True)
    assert "init_from ignored: --resume restores" in r.stdout and "resume_from=6" in r.stdout
    assert "loaded" not in "".join(l for l in r.stdout.splitlines() if l.startswith("init_from ") and "ignored" not in l)
    assert 9 in losses(tmp_path, "R")
    # --resume with init_from but NO checkpoint yet: warm start applies
    r2 = run(demos, tmp_path, "R2", "head=flow", "ground_weight=0.01", "steps=3", f"init_from={disc_pt}", resume=True)
    assert "init_from ignored" not in r2.stdout and "init_from " in r2.stdout and "loaded" in r2.stdout


def test_init_from_none_is_bit_identical_and_real_warm_start_differs(demos, disc_pt, tmp_path):
    run(demos, tmp_path, "base", "head=flow", "ground_weight=0.01", "steps=4")
    run(demos, tmp_path, "none", "head=flow", "ground_weight=0.01", "steps=4", "init_from=null")
    a, b = losses(tmp_path, "base"), losses(tmp_path, "none")
    assert a == b and a[1] == b[1]                                   # bit-identical, every logged step
    run(demos, tmp_path, "warm", "head=flow", "ground_weight=0.01", "steps=4", f"init_from={disc_pt}")
    assert losses(tmp_path, "warm")[1] != a[1]                       # the warm start really changes the starting point
