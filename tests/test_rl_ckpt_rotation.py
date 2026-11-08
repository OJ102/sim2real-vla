"""RL mid-run checkpoint rotation: keep only the latest <shm_dir>/<run>_iter{n}.pt, delete it after final.pt is exported.

CPU/gloo, configs/rl_tiny.yaml, ckpt_every=1, iterations=3 -> mid-run exports at iter1 and iter2 (none at K).
Intermediate observation without a src hook: a runpy wrapper patches vla.train.checkpoint.export_final
and records the set of _iter*.pt files at ENTRY to every export (= state at each iteration boundary, after the
previous rotation finished), then torch.load()s the latest one to prove it is a complete checkpoint.
"""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}

WRAP = '''
import json, os, runpy, torch
from pathlib import Path
import vla.train.checkpoint as C
SHM, NAME, LOG = Path(os.environ["ROT_SHM"]), os.environ["ROT_NAME"], os.environ["ROT_LOG"]
def spy(orig):
    def f(model, path, *a, **k):
        files = sorted(p.name for p in SHM.glob(f"{NAME}_iter*.pt"))
        loadable = None
        if files:
            try:
                loadable = isinstance(torch.load(SHM / files[-1], map_location="cpu"), dict)
            except Exception as e:
                loadable = repr(e)
        with open(LOG, "a") as fh:
            fh.write(json.dumps({"target": Path(path).name, "iter_files": files, "loadable": loadable}) + "\\n")
        return orig(model, path, *a, **k)
    return f
if __name__ == "__main__":  # spawn workers re-import this file as __mp_main__: must not re-run grpo
    # patch at the source BEFORE vla.rl.learner / vla.rl.grpo bind it via `from vla.train.checkpoint import export_final`
    # (runpy re-executes grpo in a fresh namespace, so patching an imported vla.rl.grpo would be a no-op)
    C.export_final = spy(C.export_final)
    import sys
    if "vla.rl.learner" in sys.modules:  # already imported (e.g. by a sitecustomize): rebind its copy too
        sys.modules["vla.rl.learner"].export_final = C.export_final
    runpy.run_module("vla.rl.grpo", run_name="__main__")
'''


# rot_x / rot2 are other runs whose names start with "rot"; rot_iter1.ptx is not a .pt
DECOYS = ("rot_x_iter1.pt", "rot2_iter1.pt", "rot_iter1.ptx", "other_iter3.pt")


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


@pytest.fixture(scope="module")
def rotated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rot")
    name, shm_dir, log = "rot", tmp / "shm", tmp / "exports.jsonl"
    script = tmp / "wrap.py"; script.write_text(WRAP)
    shm_dir.mkdir()
    for d in DECOYS:                                # other runs' / non-matching files: rotation must not touch them
        (shm_dir / d).write_bytes(b"decoy")
    env = {**ENV, "ROT_SHM": str(shm_dir), "ROT_NAME": name, "ROT_LOG": str(log)}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={_port()}", str(script), "--config", "configs/rl_tiny.yaml", f"runs_dir={tmp / 'runs'}",
           f"shm_dir={shm_dir}", f"name={name}", "iterations=3", "staleness=1", "ckpt_every=1"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=800)
    if r.returncode != 0:
        (tmp / "stderr.txt").write_text(r.stderr)
    assert r.returncode == 0, f"full stderr: {tmp / 'stderr.txt'}\n" + r.stdout[-2000:] + r.stderr[-2500:]
    events = [json.loads(l) for l in log.read_text().splitlines()]
    return {"events": events, "shm_dir": shm_dir, "name": name, "run": tmp / "runs" / name}


def test_exports_happened_as_configured(rotated):
    """Sanity of the harness: mid-run exports at iter1, iter2 (not at K), then final.pt."""
    ev = rotated["events"]
    assert [e["target"] for e in ev] == ["rot_iter1.pt", "rot_iter2.pt", "final.pt"], ev


def test_at_most_one_iter_ckpt_at_every_boundary(rotated):
    for e in rotated["events"]:
        assert len(e["iter_files"]) <= 1, e


def test_latest_kept_until_final_export(rotated):
    ev = rotated["events"]
    assert ev[0]["iter_files"] == []                                        # nothing before the first export
    assert ev[1]["iter_files"] == ["rot_iter1.pt"] and ev[1]["loadable"] is True
    assert ev[2]["iter_files"] == ["rot_iter2.pt"] and ev[2]["loadable"] is True   # the latest survives up to final


def test_none_left_after_finish_and_final_exists(rotated):
    left = sorted(p.name for p in rotated["shm_dir"].glob(f"{rotated['name']}_iter*.pt"))
    assert left == [], left
    final = rotated["run"] / "final.pt"
    assert final.exists() and final.stat().st_size > 0
    import torch
    assert isinstance(torch.load(final, map_location="cpu"), dict)


def test_other_runs_files_untouched(rotated):
    for d in DECOYS:
        assert (rotated["shm_dir"] / d).read_bytes() == b"decoy", d


def test_no_atomic_write_tmp_files_left(rotated):
    leftovers = [str(p) for d in (rotated["shm_dir"], rotated["run"]) for p in d.rglob(".*.tmp*")]
    assert leftovers == []


# ---------------------------------------------------------------- prune_iter_ckpts unit
def test_prune_iter_ckpts_exact_name_keep_and_missing_dir(tmp_path):
    from vla.rl.learner import iter_ckpt_path, prune_iter_ckpts
    shm = tmp_path / "foo"
    assert prune_iter_ckpts(shm) == []                              # parent exists, nothing to prune
    assert prune_iter_ckpts(tmp_path / "nope" / "foo") == []        # parent missing: no crash
    mine = [iter_ckpt_path(shm, n) for n in (1, 2, 10)]
    assert [p.name for p in mine] == ["foo_iter1.pt", "foo_iter2.pt", "foo_iter10.pt"] and mine[0].parent == tmp_path
    others = [tmp_path / n for n in ("foo_iter5_iter1.pt", "foobar_iter1.pt", "foo_iterX.pt", "foo_iter1.pt.bak", "xfoo_iter1.pt")]
    for p in mine + others:
        p.write_bytes(b"x")
    removed = prune_iter_ckpts(shm, keep=mine[2])
    assert sorted(p.name for p in removed) == ["foo_iter1.pt", "foo_iter2.pt"]
    assert mine[2].exists() and all(p.exists() for p in others)
    assert [p.name for p in prune_iter_ckpts(shm)] == ["foo_iter10.pt"] and all(p.exists() for p in others)
