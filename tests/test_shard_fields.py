"""Grounding data fields (plan steps 1-2): cube_pos / target_idx in shards, ground_xy in the dataset, stats."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vla.data.dataset import VLADataset, make_collate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("g_demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=1", "chunk_size=1", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return out


def test_shards_have_cube_pos_and_target_idx(demos):
    for color, idx in (("red", 0), ("blue", 2)):
        with h5py.File(demos / f"lift_{color}_0000.h5", "r") as f:
            n, e = f["actions"].shape[0], len(f["ep_len"])
            assert f["cube_pos"].shape == (n, 3, 3) and f["cube_pos"].dtype == np.float32
            assert f["target_idx"].shape == (e,) and f["target_idx"].dtype == np.int64 and (f["target_idx"][:] == idx).all()
            assert f["images"].shape[0] == f["proprio"].shape[0] == f["cube_pos"].shape[0]
            cp = f["cube_pos"][:]
            assert np.isfinite(cp).all() and (cp[..., 2] > 0).all() and (cp[..., 2] < 0.2).all()   # cubes on the table
            assert (np.abs(cp[..., :2]) < 1.0).all()


def test_cube_pos_aligned_with_frame(demos):
    """cube_pos[row] must be the cube position at the frame images[row]: frame 0 equals the env state after reset."""
    from vla.sim.env import TabletopEnv

    with h5py.File(demos / "lift_blue_0000.h5", "r") as f:
        seed, cp0, img0, tidx = int(f["seed"][0]), f["cube_pos"][0], f["images"][0], int(f["target_idx"][0])
    env = TabletopEnv("lift", "train")
    try:
        obs, _ = env.reset(seed=seed, options={"target_idx": tidx})
        np.testing.assert_allclose(env.privileged()["cube_pos"], cp0, atol=1e-5)
        assert np.array_equal(obs["image"], img0)
    finally:
        env.close()


def test_dataset_ground_xy_matches_normalised_target(demos):
    ds = VLADataset(str(demos), chunk=4)
    st = json.loads((demos / "stats.json").read_text())
    mean, std = np.array(st["target_xy_mean"]), np.array(st["target_xy_std"])
    assert mean.shape == (2,) and std.shape == (2,) and (std > 0).all()
    checked = 0
    for i in range(0, len(ds), 7):
        e = int(ds.frame_ep[i]); t = i - int(ds.ep_first[e]); row = ds.ep_start[e] + t
        with h5py.File(demos / ds.shards[int(ds.ep_shard[e])], "r") as f:
            ti = int(f["target_idx"][int(np.nonzero(f["ep_start"][:] == ds.ep_start[e])[0][0])])
            want = (f["cube_pos"][row, ti, :2] - mean) / std
        it = ds[i]
        assert it["ground_xy"].shape == (2,) and it["ground_xy"].dtype == torch.float32 and it["ground_mask"] == 1.0
        np.testing.assert_allclose(it["ground_xy"].numpy(), want, atol=1e-5)
        checked += 1
    assert checked > 3
    assert ds.frame_t[0] == 0 and len(ds.frame_t) == len(ds)                                  # per-frame time index
    batch = make_collate(__import__("vla.model.prompts", fromlist=["ByteTokenizer"]).ByteTokenizer())([ds[0], ds[3]])
    assert batch["ground_xy"].shape == (2, 2) and batch["ground_mask"].shape == (2,)


def test_stats_normalisation_is_correct(demos):
    ds = VLADataset(str(demos))
    xs = torch.stack([ds[i]["ground_xy"] for i in range(len(ds))])
    assert xs.mean(0).abs().max() < 1e-3 and (xs.std(0, unbiased=False) - 1).abs().max() < 1e-2


def test_legacy_shard_gives_mask_zero(demos, tmp_path):
    """A shard without cube_pos (old data) still loads; ground_mask is 0 and nothing crashes."""
    leg = tmp_path / "legacy"; leg.mkdir()
    with h5py.File(demos / "lift_red_0000.h5", "r") as src, h5py.File(leg / "lift_red_0000.h5", "w") as dst:
        for k in src:
            if k not in ("cube_pos", "target_idx"):
                src.copy(k, dst)
        dst.attrs.update(src.attrs)
    st = json.loads((demos / "stats.json").read_text())
    st.pop("target_xy_mean", None); st.pop("target_xy_std", None)
    (leg / "stats.json").write_text(json.dumps(st))
    ds = VLADataset(str(leg))
    it = ds[0]
    assert it["ground_mask"] == 0.0 and torch.isfinite(it["ground_xy"]).all()
    b = make_collate(__import__("vla.model.prompts", fromlist=["ByteTokenizer"]).ByteTokenizer())([ds[0], ds[1]])
    assert (b["ground_mask"] == 0).all()
