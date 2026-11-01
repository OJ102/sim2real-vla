"""Amendment A (data side): counterfactual colour swap in VLADataset, plus bc.py val_ground_err_cm_swap / _t0."""
import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vla.data.dataset import VLADataset, make_collate
from vla.model.prompts import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["red", "green", "blue"]
RX = re.compile(r"the (red|green|blue) cube")


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("swap_demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=2", "chunk_size=2", "tasks=[lift,push]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=400)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return out


def items(ds, idx=None):
    return [ds[i] for i in (range(len(ds)) if idx is None else idx)]


def test_p0_reproduces_current_batches_exactly(demos):
    a, b = VLADataset(str(demos), chunk=4), VLADataset(str(demos), chunk=4, swap_p=0.0, seed=5)
    for i in range(0, len(a), 5):
        x, y = a[i], b[i]
        assert "swap" not in x and "swap" not in y
        assert x["instruction"] == y["instruction"]
        for k in ("image", "proprio", "actions", "action_mask", "ground_xy", "ground_mask"):
            assert torch.equal(x[k], y[k]), k
    col = make_collate(ByteTokenizer())
    ba, bb = col([a[0], a[4]]), col([b[0], b[4]])
    assert set(ba) == set(bb) and "swap" not in bb and all(torch.equal(ba[k], bb[k]) for k in ba)


def test_deterministic_per_seed_epoch_index_and_epoch_changes(demos):
    ds1 = VLADataset(str(demos), swap_p=0.5, seed=3); ds2 = VLADataset(str(demos), swap_p=0.5, seed=3)
    for e in (0, 1):
        ds1.set_epoch(e); ds2.set_epoch(e)
        for i in range(0, len(ds1), 3):
            a, b = ds1[i], ds2[i]
            assert a["instruction"] == b["instruction"] and a["swap"] == b["swap"] and torch.equal(a["ground_xy"], b["ground_xy"])
            assert a["swap"] == ds1[i]["swap"]                                  # repeatable within an epoch
    ds1.set_epoch(0); ds2.set_epoch(1)
    diff = sum(ds1[i]["instruction"] != ds2[i]["instruction"] or ds1[i]["swap"] != ds2[i]["swap"] for i in range(len(ds1)))
    assert diff > 0                                                              # different epoch can differ
    ds3 = VLADataset(str(demos), swap_p=0.5, seed=4); ds3.set_epoch(0)
    assert any(ds1[i]["instruction"] != ds3[i]["instruction"] for i in range(len(ds1)))   # seed matters


def test_swap_rate_and_swapped_row_semantics(demos):
    base = VLADataset(str(demos), chunk=4)
    ds = VLADataset(str(demos), chunk=4, swap_p=0.5, seed=1); ds.set_epoch(0)
    st = json.loads((demos / "stats.json").read_text())
    mean, std = np.array(st["target_xy_mean"]), np.array(st["target_xy_std"])
    n_swap = 0
    for i in range(len(ds)):
        it, b0 = ds[i], base[i]
        assert it["swap"].dtype == torch.float32 and it["swap"].ndim == 0
        if it["swap"] == 0:
            assert it["instruction"] == b0["instruction"] and torch.equal(it["ground_xy"], b0["ground_xy"])
            continue
        n_swap += 1
        m0, m1 = RX.search(b0["instruction"]), RX.search(it["instruction"])
        assert m0 and m1 and m0.group(1) != m1.group(1) and m1.group(1) in COLORS            # different colour word
        assert RX.sub("", b0["instruction"]) == RX.sub("", it["instruction"])                 # rest of the text untouched
        e = int(ds.frame_ep[i]); t = i - int(ds.ep_first[e]); row = ds.ep_start[e] + t
        with h5py.File(demos / ds.shards[int(ds.ep_shard[e])], "r") as f:
            want = (f["cube_pos"][row, COLORS.index(m1.group(1)), :2] - mean) / std
        np.testing.assert_allclose(it["ground_xy"].numpy(), want, atol=1e-5)                  # label = that cube's xy
        assert it["ground_mask"] == 1.0
        for k in ("image", "proprio", "actions"):
            assert torch.equal(it[k], b0[k])                                                  # observations/actions unchanged
    assert 0.35 < n_swap / len(ds) < 0.65, n_swap / len(ds)


def test_p1_swaps_everything_uniformly_to_other_colours(demos):
    ds = VLADataset(str(demos), swap_p=1.0, seed=0); ds.set_epoch(0)
    base = VLADataset(str(demos))
    seen = {}
    for i in range(len(ds)):
        it = ds[i]
        assert it["swap"] == 1.0
        c0, c1 = RX.search(base[i]["instruction"]).group(1), RX.search(it["instruction"]).group(1)
        assert c0 != c1
        seen.setdefault(c0, set()).add(c1)
    assert seen.get("red", set()) <= {"green", "blue"} and len(seen.get("red", set())) == 2     # both other colours occur


def test_push_yellow_zone_untouched(demos):
    ds = VLADataset(str(demos), swap_p=1.0, seed=0); ds.set_epoch(0)
    pushes = [i for i in range(len(ds)) if "push" in ds[i]["instruction"]]
    assert pushes
    for i in pushes[:30]:
        ins = ds[i]["instruction"]
        assert ins.startswith("push the ") and ins.endswith(" cube to the yellow zone") and "yellow zone" in ins


def test_collate_stacks_swap(demos):
    ds = VLADataset(str(demos), swap_p=0.5, seed=2); ds.set_epoch(0)
    b = make_collate(ByteTokenizer())([ds[i] for i in range(6)])
    assert b["swap"].shape == (6,) and b["swap"].dtype == torch.float32 and set(b["swap"].tolist()) <= {0.0, 1.0}


def test_legacy_shard_never_swaps(demos, tmp_path):
    leg = tmp_path / "legacy"; leg.mkdir()
    with h5py.File(demos / "lift_red_0000.h5", "r") as src, h5py.File(leg / "lift_red_0000.h5", "w") as dst:
        for k in src:
            if k not in ("cube_pos", "target_idx"):
                src.copy(k, dst)
        dst.attrs.update(src.attrs)
    st = json.loads((demos / "stats.json").read_text()); st.pop("target_xy_mean", None); st.pop("target_xy_std", None)
    (leg / "stats.json").write_text(json.dumps(st))
    ds = VLADataset(str(leg), swap_p=1.0, seed=0); ds.set_epoch(0)
    base = VLADataset(str(leg))
    for i in range(len(ds)):
        it = ds[i]
        assert it["instruction"] == base[i]["instruction"] and it["ground_mask"] == 0.0 and not it.get("swap", 0)


def test_val_part_never_swapped_by_default(demos):
    v = VLADataset(str(demos), part="val")
    assert all("swap" not in v[i] for i in range(len(v)))


# ---- bc.py: val metrics
def test_bc_logs_swap_and_t0_val_metrics(demos, tmp_path):
    import socket

    def port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
    env = {**__import__("os").environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}

    def run(name, *extra):
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                            "--master_addr=127.0.0.1", f"--master_port={port()}", "-m", "vla.train.bc", "--config", "configs/bc_tiny.yaml",
                            f"data_dir={demos}", f"runs_dir={tmp_path}", f"name={name}", "steps=10", "val_every=5", "val_batches=2",
                            "log_every=5", "ckpt_every=100", "pg_timeout_min=1", *extra], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        return [json.loads(l) for l in (tmp_path / name / "metrics.jsonl").read_text().splitlines() if '"loss"' in l]

    v = [r for r in run("sw", "ground_weight=0.2", "ground_swap_p=0.25") if "val_main_loss" in r]
    assert v and all(k in v[0] for k in ("val_ground_err_cm_swap", "val_ground_err_cm_t0", "val_ground_err_cm", "val_main_loss"))
    assert all(np.isfinite(v[0][k]) for k in ("val_ground_err_cm_swap", "val_ground_err_cm_t0", "val_main_loss"))
    v0 = [r for r in run("nosw", "ground_weight=0.2", "ground_swap_p=0.0") if "val_main_loss" in r]
    assert v0 and "val_ground_err_cm_swap" not in v0[0] and "val_ground_err_cm_t0" in v0[0]      # swap pass only when p>0
    assert all("main_loss" in r for r in run("sw2", "ground_weight=0.2", "ground_swap_p=0.25"))
