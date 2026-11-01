"""P2 tests (CPU, small): demo generation, VLADataset, resumable sampler, DR ablation groups, EGL helpers."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vla.data.dataset import VLADataset, make_collate
from vla.data.sampler import ResumableDistributedSampler
from vla.sim.assets import CUBE_BASE_RGB, CUBE_COLORS
from vla.sim.env import TabletopEnv
from vla.sim.randomization import NOMINAL_GROUPS, nominalize, sample_dr

ROOT = Path(__file__).resolve().parents[1]
CMD = [sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
       "--no-require-nvidia"]


def _gen(out):
    return subprocess.run(CMD + [f"out_dir={out}", "episodes_per_combo=2", "chunk_size=2", "tasks=[lift]",
                                 "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)


@pytest.fixture(scope="module")
def gen_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    r = _gen(out)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


# ------------------------------------------------------------------ 1) generate
def test_generate_files_and_schema(gen_dir):
    names = sorted(p.name for p in gen_dir.iterdir())
    assert names == ["lift_blue_0000.h5", "lift_red_0000.h5", "manifest.json", "stats.json",
                     "subset_10.json", "subset_100.json", "subset_30.json"]
    man = json.loads((gen_dir / "manifest.json").read_text())
    assert man["episodes"] == 4 and man["per_task"] == {"lift": 4} and sorted(man["shards"]) == names[:2]
    st = json.loads((gen_dir / "stats.json").read_text())
    assert len(st["proprio_mean"]) == 11 and len(st["proprio_std"]) == 11 and min(st["proprio_std"]) > 0
    seeds = []
    for color in ("red", "blue"):
        with h5py.File(gen_dir / f"lift_{color}_0000.h5", "r") as f:
            n = f["actions"].shape[0]
            assert f["images"].shape == (n, 128, 128, 3) and f["images"].dtype == np.uint8
            assert f["proprio"].shape == (n, 11) and f["proprio"].dtype == np.float32
            assert f["actions"].shape == (n, 4) and f["actions"].dtype == np.float32
            assert np.abs(f["actions"][:]).max() <= 1
            lens, starts = f["ep_len"][:], f["ep_start"][:]
            assert len(lens) == 2 and lens.sum() == n and starts[0] == 0 and list(starts[1:]) == list(np.cumsum(lens)[:-1])
            assert (lens <= 100).all() and (lens >= 1).all()
            for ds in ("task", "color", "instruction"):
                assert len(f[ds]) == 2
            assert {x.decode() for x in f["color"][:]} == {color}
            assert {x.decode() for x in f["task"][:]} == {"lift"}
            for ins in f["instruction"][:]:
                assert ins.decode() == f"lift the {color} cube"            # instruction colour == shard colour
            assert f["images"][:].std() > 5 and f.attrs["split"] == "train"
            seeds += list(f["seed"][:])
    assert len(set(seeds)) == len(seeds)                                  # disjoint seeds


def test_generate_replays_from_seed_and_only_successes(gen_dir):
    """First frame of an episode equals env.reset(seed, target colour) -> data is reproducible; episode ends in success."""
    with h5py.File(gen_dir / "lift_blue_0000.h5", "r") as f:
        seed, img0, pro0 = int(f["seed"][0]), f["images"][0], f["proprio"][0]
        n0 = int(f["ep_len"][0])
    env = TabletopEnv("lift", "train")
    try:
        obs, info = env.reset(seed=seed, options={"target_idx": CUBE_COLORS.index("blue")})
        assert info["instruction"] == "lift the blue cube"
        assert np.array_equal(obs["image"], img0) and np.allclose(obs["proprio"], pro0)
        from vla.sim.expert import ScriptedExpert
        ex = ScriptedExpert("lift", np.random.default_rng(10_000_000 + seed), 0.05)
        done, steps = False, 0
        while not done:
            _, _, term, trunc, info = env.step(ex.act(env)); steps += 1; done = term or trunc
        assert info["success"] and steps == n0                            # stored episode == successful replay
    finally:
        env.close()


def test_generate_resumable(gen_dir):
    before = {p.name: p.stat().st_mtime_ns for p in gen_dir.glob("*.h5")}
    r = _gen(gen_dir)
    assert r.returncode == 0 and "0 jobs (2 total)" in r.stdout, r.stdout
    assert {p.name: p.stat().st_mtime_ns for p in gen_dir.glob("*.h5")} == before
    assert not list(gen_dir.glob("*.tmp"))


def test_generate_partial_resume(tmp_path):
    r = _gen(tmp_path); assert r.returncode == 0
    (tmp_path / "lift_red_0000.h5").unlink()
    r = _gen(tmp_path)
    assert "1 jobs (2 total)" in r.stdout and (tmp_path / "lift_red_0000.h5").exists()


# ------------------------------------------------------------------ 2) dataset
def _synth(root: Path, n_shards=3, eps_per=9, seed=0):
    """Tiny synthetic shards (fast): 27 episodes of variable length with known actions."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    all_pro = []
    for s in range(n_shards):
        lens = rng.integers(1, 7, eps_per)
        n = int(lens.sum())
        pro = rng.normal(5.0, 3.0, (n, 11)).astype(np.float32)
        all_pro.append(pro)
        with h5py.File(root / f"lift_red_{s:04d}.h5", "w") as f:
            f["images"] = rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8)
            f["proprio"], f["actions"] = pro, rng.uniform(-1, 1, (n, 4)).astype(np.float32)
            f["ep_start"] = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
            f["ep_len"], f["seed"] = lens.astype(np.int64), np.arange(eps_per, dtype=np.int64)
            dt = h5py.string_dtype()
            f.create_dataset("instruction", data=np.array([f"lift the red cube"] * eps_per, dtype=object), dtype=dt)
    p = np.concatenate(all_pro)
    json.dump({"proprio_mean": p.mean(0).tolist(), "proprio_std": (p.std(0) + 1e-6).tolist(), "n_frames": len(p)},
              open(root / "stats.json", "w"))
    return root


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return _synth(tmp_path_factory.mktemp("synth"))


def test_dataset_len_and_items(gen_dir):
    ds = VLADataset(str(gen_dir), chunk=4)
    with h5py.File(gen_dir / "lift_blue_0000.h5", "r") as a, h5py.File(gen_dir / "lift_red_0000.h5", "r") as b:
        assert len(ds) == a["actions"].shape[0] + b["actions"].shape[0]
    it = ds[0]
    assert it["image"].shape == (128, 128, 3) and it["image"].dtype == torch.uint8
    assert it["proprio"].shape == (11,) and it["actions"].shape == (4, 4) and it["action_mask"].shape == (4,)
    assert "cube" in it["instruction"]


def test_dataset_chunk_padding_and_mask(synth):
    ds = VLADataset(str(synth), chunk=4)
    seen_pad = 0
    for e in range(len(ds.ep_len)):
        L, first = int(ds.ep_len[e]), int(ds.ep_first[e])
        with h5py.File(synth / ds.shards[int(ds.ep_shard[e])], "r") as f:
            acts = f["actions"][ds.ep_start[e]:ds.ep_start[e] + L]
        for t in range(L):
            it = ds[first + t]
            k = min(4, L - t)
            assert it["action_mask"].tolist() == [1.0] * k + [0.0] * (4 - k), (e, t)
            np.testing.assert_array_equal(it["actions"][:k].numpy(), acts[t:t + k])
            if k < 4:
                seen_pad += 1
                assert torch.equal(it["actions"][k:], it["actions"][k - 1:k].expand(4 - k, 4))   # padded with last action
        last = ds[first + L - 1]
        assert last["action_mask"].tolist() == [1, 0, 0, 0]
        assert torch.equal(last["actions"], torch.tensor(acts[-1]).expand(4, 4))
    assert seen_pad > 0
    assert len(ds) == int(ds.ep_len.sum())


def test_dataset_proprio_normalised(synth):
    ds = VLADataset(str(synth))
    x = torch.stack([ds[i]["proprio"] for i in range(len(ds))])
    assert x.mean(0).abs().max() < 1e-4 and (x.std(0, unbiased=False) - 1).abs().max() < 1e-3


def test_dataset_train_val_split(synth):
    allds, tr, va = (VLADataset(str(synth), part=p) for p in ("all", "train", "val"))
    n_eps = len(allds.ep_len)
    assert n_eps == 27 and len(va.ep_len) == len(range(0, n_eps, 20)) == 2
    assert len(tr.ep_len) + len(va.ep_len) == n_eps
    key = lambda d: {(int(d.ep_shard[i]), int(d.ep_start[i])) for i in range(len(d.ep_len))}   # noqa: E731
    assert not key(tr) & key(va) and key(tr) | key(va) == key(allds)
    assert len(tr) + len(va) == len(allds)
    # val is exactly the 0th and 20th episodes (global order over sorted shards)
    order = [(si, int(allds.ep_start[i])) for i, si in enumerate(allds.ep_shard)]
    assert key(va) == {order[0], order[20]}


def test_dataset_subset_restricts(gen_dir):
    sub = gen_dir / "sub.json"
    sub.write_text(json.dumps([["lift_red_0000.h5", 1]]))
    ds = VLADataset(str(gen_dir), subset=str(sub))
    assert len(ds.ep_len) == 1 and ds.shards[int(ds.ep_shard[0])] == "lift_red_0000.h5" and ds.instr == ["lift the red cube"]
    with h5py.File(gen_dir / "lift_red_0000.h5", "r") as f:
        lens = f["ep_len"][:]
    assert len(ds) == int(lens[1])
    s10 = VLADataset(str(gen_dir), subset=str(gen_dir / "subset_10.json"))
    full = VLADataset(str(gen_dir), subset=str(gen_dir / "subset_100.json"))
    assert len(s10) <= len(full) == len(VLADataset(str(gen_dir)))


def test_collate_feeds_tiny_vla(gen_dir):
    from vla.model.vla import VLA, VLAConfig
    from vla.utils.config import load_config

    torch.manual_seed(0)
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), []))
    ds = VLADataset(str(gen_dir), chunk=4)
    batch = make_collate(m.tokenizer)([ds[i] for i in (0, 5, len(ds) - 1)])
    assert batch["image"].shape == (3, 128, 128, 3) and batch["actions"].shape == (3, 4, 4)
    assert batch["input_ids"].shape[0] == 3 and batch["attention_mask"].shape == batch["input_ids"].shape
    out = m(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


# ------------------------------------------------------------------ 3) sampler
def _drain(s):
    return list(iter(s))


def test_sampler_resume_mid_epoch_identical():
    a = ResumableDistributedSampler(103, rank=1, world_size=4, seed=3, epoch=2)
    it = iter(a)
    head = [next(it) for _ in range(7)]
    sd = a.state_dict()
    assert sd["consumed"] == 7
    rest_a = list(it)
    b = ResumableDistributedSampler(103, rank=1, world_size=4, seed=3)
    b.load_state_dict(sd)
    assert _drain(b) == rest_a and len(head) + len(rest_a) == 103 // 4
    full = _drain(ResumableDistributedSampler(103, rank=1, world_size=4, seed=3, epoch=2))
    assert head + rest_a == full


def test_sampler_ranks_disjoint_and_cover():
    n, w = 103, 4
    outs = [_drain(ResumableDistributedSampler(n, r, w, seed=1)) for r in range(w)]
    flat = [i for o in outs for i in o]
    assert len(flat) == len(set(flat)) == n // w * w and all(0 <= i < n for i in flat)
    assert all(len(o) == n // w for o in outs)
    # same global permutation each rank draws from: union equals the first n//w*w of that permutation
    g = torch.Generator(); g.manual_seed(1 * 1_000_003 + 0)
    assert set(flat) == set(torch.randperm(n, generator=g).tolist()[: n // w * w])


def test_sampler_epoch_and_seed_change_order():
    s = ResumableDistributedSampler(50, 0, 2, seed=0)
    e0 = _drain(s)
    s.set_epoch(1)
    assert s.consumed == 0
    e1 = _drain(s)
    assert e0 != e1
    assert _drain(ResumableDistributedSampler(50, 0, 2, seed=0, epoch=0)) == e0          # deterministic
    assert _drain(ResumableDistributedSampler(50, 0, 2, seed=1, epoch=0)) != e0
    assert len(ResumableDistributedSampler(50, 0, 2)) == 25


def test_sampler_state_dict_consumed_override_and_len():
    s = ResumableDistributedSampler(40, 0, 2, seed=5)
    it = iter(s)
    for _ in range(10):
        next(it)                                    # prefetch ran ahead: 10 pulled, trainer only used 6
    sd = s.state_dict(consumed=6)
    assert sd["consumed"] == 6 and s.consumed == 10  # override must not mutate the sampler
    r = ResumableDistributedSampler(40, 0, 2, seed=5)
    r.load_state_dict(sd)
    assert len(r) == 20 - 6
    full = _drain(ResumableDistributedSampler(40, 0, 2, seed=5))
    assert _drain(r) == full[6:]


def test_sampler_load_asserts_on_mismatch():
    sd = ResumableDistributedSampler(40, 0, 2).state_dict()
    with pytest.raises(AssertionError):
        ResumableDistributedSampler(41, 0, 2).load_state_dict(sd)
    with pytest.raises(AssertionError):
        ResumableDistributedSampler(40, 0, 4).load_state_dict(sd)


def test_sampler_through_dataloader_resume(synth):
    ds = VLADataset(str(synth), chunk=2)
    ref = ResumableDistributedSampler(len(ds), 0, 1, seed=2)
    full = _drain(ref)
    s = ResumableDistributedSampler(len(ds), 0, 1, seed=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=5, sampler=s, num_workers=0, collate_fn=lambda x: x)
    it = iter(dl); next(it); next(it)
    sd = s.state_dict(consumed=10)
    s2 = ResumableDistributedSampler(len(ds), 0, 1, seed=2); s2.load_state_dict(sd)
    assert _drain(s2) == full[10:]


# ------------------------------------------------------------------ 4) DR ablation groups
def _fields(p):
    return {k: np.array(v) for k, v in p.to_dict().items()}


def _same(a, b, keys):
    return all(np.array_equal(a[k], b[k]) for k in keys)


CAM = ["cam_pos_offset", "cam_look_offset", "cam_fov_delta"]
VIS = ["n_lights", "light_az", "light_el", "light_dir", "light_diffuse", "table_mat", "floor_mat", "cube_rgb"]
PHYS = ["friction", "cube_mass", "kp_scale"]


@pytest.mark.parametrize("split", ["train", "heldout"])
@pytest.mark.parametrize("group,fixed,others", [("cam", CAM, VIS + PHYS), ("vis", VIS, CAM + PHYS),
                                                ("phys", PHYS, CAM + VIS)])
def test_nominalize_groups(split, group, fixed, others):
    for s in range(20):
        p = sample_dr(split, np.random.default_rng(s))
        q = nominalize(p, [group])
        a, b = _fields(p), _fields(q)
        assert _same(a, b, others), (group, s, [k for k in others if not np.array_equal(a[k], b[k])])
        if group == "cam":
            assert not np.any(b["cam_pos_offset"]) and not np.any(b["cam_look_offset"]) and b["cam_fov_delta"] == 0
        if group == "phys":
            assert b["friction"] == 0.9 and b["cube_mass"] == 0.1 and np.all(b["kp_scale"] == 1)
        if group == "vis":
            assert b["table_mat"] == 1 and b["floor_mat"] == 0
            assert np.allclose(b["cube_rgb"], [CUBE_BASE_RGB[c] for c in CUBE_COLORS])
            assert np.all(b["light_diffuse"] == b["light_diffuse"][0]) and np.allclose(np.linalg.norm(b["light_dir"], axis=1), 1)
    # nominal values are identical across seeds (ablation really removes the variation)
    qs = [_fields(nominalize(sample_dr(split, np.random.default_rng(s)), [group])) for s in range(5)]
    assert all(_same(qs[0], q, fixed) for q in qs)


def test_nominalize_input_not_mutated_and_combined():
    p = sample_dr("train", np.random.default_rng(0)); before = _fields(p)
    q = nominalize(p, ["cam", "vis", "phys"])
    assert _same(before, _fields(p), before.keys())
    assert _same(_fields(q), _fields(nominalize(sample_dr("train", np.random.default_rng(9)), NOMINAL_GROUPS)), CAM + PHYS + VIS)
    assert _same(_fields(nominalize(p, [])), before, before.keys())


def test_nominalize_unknown_group_raises():
    with pytest.raises(ValueError):
        nominalize(sample_dr("train", np.random.default_rng(0)), ["lighting"])


def _cubes(e):
    return np.array([e.data.xpos[b] for b in e.cube_body])


@pytest.mark.parametrize("group", ["cam", "vis"])
def test_env_dr_nominal_keeps_layout(group):
    a, b = TabletopEnv("push", "train", render_obs=False), TabletopEnv("push", "train", render_obs=False, dr_nominal=(group,))
    try:
        for s in range(5):
            a.reset(seed=s); b.reset(seed=s)
            np.testing.assert_allclose(_cubes(a), _cubes(b), atol=1e-9)
            assert a.target_idx == b.target_idx and np.array_equal(a.goal_xy, b.goal_xy)
        # and the nominal group is really applied to the model
        if group == "cam":
            assert np.allclose(b.model.cam_fovy[b.cam], 50.0) and not np.allclose(a.model.cam_fovy[a.cam], 50.0)
    finally:
        a.close(); b.close()


def test_env_dr_nominal_images_differ_from_full_dr():
    a, b = TabletopEnv("lift", "train"), TabletopEnv("lift", "train", dr_nominal=("cam",))
    try:
        ia, ib = a.reset(seed=3)[0]["image"], b.reset(seed=3)[0]["image"]
        assert not np.array_equal(ia, ib)
        assert np.array_equal(b.reset(seed=3)[0]["image"], ib)          # deterministic
    finally:
        a.close(); b.close()


def test_env_target_idx_option():
    e = TabletopEnv("lift", "train", render_obs=False)
    try:
        for s in range(8):
            _, info0 = e.reset(seed=s)
            nat = e.target_idx
            same = e.reset(seed=s, options={"target_idx": nat})[1]
            assert same["instruction"] == info0["instruction"]
            for k in range(3):
                _, info = e.reset(seed=s, options={"target_idx": k})
                assert e.target_idx == k and e.target_color == CUBE_COLORS[k]
                assert CUBE_COLORS[k] in info["instruction"]
                rgb = e.model.geom_rgba[e.cube_geom[k], :3]
                assert CUBE_COLORS[int(np.argmax(rgb))] == CUBE_COLORS[k]
                # layout valid for the forced target: goal inside bounds, cubes separated
                c = _cubes(e)[:, :2]
                assert all(np.linalg.norm(c[i] - c[j]) >= 0.07 for i in range(3) for j in range(i + 1, 3))
                assert 0.35 <= e.goal_xy[0] <= 0.72 and abs(e.goal_xy[1]) <= 0.30
    finally:
        e.close()


def test_env_target_idx_same_rng_stream():
    """Forcing the target must not shift the RNG: the DR draw (camera, friction, ...) is identical."""
    e = TabletopEnv("lift", "train", render_obs=False)
    try:
        e.reset(seed=4); d0 = e.dr.to_dict()
        e.reset(seed=4, options={"target_idx": (e.target_idx + 1) % 3}); d1 = e.dr.to_dict()
        assert d0.keys() == d1.keys() and all(np.array_equal(d0[k], d1[k]) for k in d0)
    finally:
        e.close()


# ------------------------------------------------------------------ 5) egl helpers
def test_assert_nvidia_renderer_off_linux():
    from vla.sim.egl import assert_nvidia_renderer

    if sys.platform.startswith("linux"):
        pytest.skip("linux requires NVIDIA renderer")
    e = TabletopEnv("lift", "train")
    try:
        r = assert_nvidia_renderer(e)
        assert isinstance(r, str) and r
    finally:
        e.close()


def test_set_egl_device(monkeypatch):
    import os

    from vla.sim.env import set_egl_device

    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    set_egl_device(5, [3, 1, 2])
    if sys.platform.startswith("linux"):
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == str(2)           # 5 % 3 -> index 2 -> id 2
        set_egl_device(0, [7, 8]); assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "7"
    else:
        assert "MUJOCO_EGL_DEVICE_ID" not in os.environ              # no-op off Linux
