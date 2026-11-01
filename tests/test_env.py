"""Phase 0/1 acceptance tests: determinism, DR, split disjointness, obs/info contract, hygiene."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import mujoco

from vla.sim.assets import CUBE_COLORS, HELDOUT_MATS, HELDOUT_PHOTO_MATS, TRAIN_MATS
from vla.sim.env import MAX_STEPS, TabletopEnv
from vla.sim.randomization import DRConfig, sample_dr

ROOT = Path(__file__).resolve().parents[1]
ACTS = [np.array(a, np.float32) for a in
        ([0.3, -0.2, 0.1, 0.0], [0.5, 0.5, -0.5, 1.0], [-0.1, 0.0, 0.2, -1.0], [0.0, 0.3, 0.3, 0.5], [0.2, 0.2, 0.0, 0.0])]


@pytest.fixture(scope="module")
def env():
    e = TabletopEnv("lift", "train")
    yield e
    e.close()


def _rollout(e, seed, acts=ACTS):
    obs, _ = e.reset(seed=seed)
    out = [obs]
    for a in acts:
        out.append(e.step(a)[0])
    return out


# ------------------------------------------------------------ determinism
def test_same_seed_identical_across_envs():
    e1, e2 = TabletopEnv("lift", "train"), TabletopEnv("lift", "train")
    try:
        r1, r2 = _rollout(e1, 7), _rollout(e2, 7)
    finally:
        e1.close(); e2.close()
    for o1, o2 in zip(r1, r2):
        assert np.array_equal(o1["image"], o2["image"])
        assert np.array_equal(o1["proprio"], o2["proprio"])


def test_same_seed_identical_after_other_seed_in_same_env(env):
    a = _rollout(env, 3)
    _rollout(env, 99)          # pollute model/data with another episode
    b = _rollout(env, 3)       # DR must be fully re-applied, no leakage
    for o1, o2 in zip(a, b):
        assert np.array_equal(o1["image"], o2["image"])
        assert np.array_equal(o1["proprio"], o2["proprio"])


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_determinism_both_splits(split):
    e = TabletopEnv("push", split)
    try:
        a, b = _rollout(e, 11), _rollout(e, 11)
    finally:
        e.close()
    assert all(np.array_equal(x["image"], y["image"]) and np.array_equal(x["proprio"], y["proprio"])
               for x, y in zip(a, b))


# ------------------------------------------------------------ DR varies per reset
def test_different_seeds_differ(env):
    seen = []
    for s in range(6):
        obs, info = env.reset(seed=s)
        m = env.model
        seen.append(dict(cam=m.cam_pos[env.cam].copy(), quat=m.cam_quat[env.cam].copy(), fov=float(m.cam_fovy[env.cam]),
                         mat=(int(m.geom_matid[env.table_geom]), int(m.geom_matid[env.floor_geom])),
                         fric=float(m.geom_friction[env.cube_geom[0], 0]),
                         mass=float(m.body_mass[env.cube_body[0]]), img=obs["image"].copy()))
    for k in ("fov", "fric", "mass"):
        assert len({round(s[k], 9) for s in seen}) == len(seen), k
    assert len({tuple(np.round(s["cam"], 9)) for s in seen}) == len(seen)
    assert len({tuple(np.round(s["quat"], 9)) for s in seen}) == len(seen)
    assert len({s["mat"] for s in seen}) > 1
    assert len({s["img"].tobytes() for s in seen}) == len(seen)


def _matname(m, geom):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[geom]))


def test_dr_is_applied_to_model_at_every_reset(env):
    """info['dr'] must match the actual mjModel after reset (not only at construction)."""
    for s in (1, 2, 3):
        _, info = env.reset(seed=s)
        p, m = env.dr, env.model
        assert np.allclose(m.cam_fovy[env.cam], 50.0 + p.cam_fov_delta)
        assert np.allclose(m.cam_pos[env.cam], np.array([1.0, 0.0, 0.45]) + p.cam_pos_offset)
        assert _matname(m, env.table_geom) == f"mat{p.table_mat}" and _matname(m, env.floor_geom) == f"mat{p.floor_mat}"
        for i in range(3):
            assert m.body_mass[env.cube_body[i]] == pytest.approx(p.cube_mass)
            assert m.geom_friction[env.cube_geom[i], 0] == pytest.approx(p.friction)
            assert np.allclose(m.geom_rgba[env.cube_geom[i], :3], p.cube_rgb[i])
        assert int(m.light_active[env.lights].sum()) == p.n_lights
        assert info["dr"]["friction"] == pytest.approx(p.friction)


def test_camera_change_changes_image(env):
    env.reset(seed=5)
    base = env.render()
    env.model.cam_fovy[env.cam] += 6
    moved = env.render()
    env.model.cam_fovy[env.cam] -= 6
    assert not np.array_equal(base, moved)


# ------------------------------------------------------------ split ranges
def _bands(v, train, held):
    return train[0] <= v <= train[1], any(a < v <= b for a, b in held)


def _light_ok_train(az, el):
    return abs(az) <= 60 and 45 <= el <= 75


def _light_ok_held(az, el):
    side = 90 < abs(az) <= 180 and 45 <= el <= 75
    low = abs(az) <= 60 and 20 <= el <= 40
    return side or low


def test_train_samples_inside_train_range():
    cfg = DRConfig()
    for s in range(200):
        p = sample_dr("train", np.random.default_rng(s), cfg)
        assert cfg.in_train_range(p), s
        assert np.all(np.abs(p.cam_pos_offset) <= cfg.cam_pos) and np.all(np.abs(p.cam_look_offset) <= cfg.cam_look)
        assert abs(p.cam_fov_delta) <= cfg.cam_fov
        assert np.all((p.light_diffuse >= 0.4) & (p.light_diffuse <= 1.0))
        assert 0.6 <= p.friction <= 1.2 and 0.05 <= p.cube_mass <= 0.2
        assert p.table_mat in TRAIN_MATS and p.floor_mat in TRAIN_MATS
        assert p.table_mat < 42 and p.floor_mat < 42
        assert 1 <= p.n_lights <= 3
        assert np.all(np.abs(p.kp_scale - 1) <= 0.1)
        for az, el in zip(np.degrees(p.light_az), np.degrees(p.light_el)):  # all 3 lights
            assert _light_ok_train(az, el), (s, az, el)


@pytest.mark.parametrize("split", ["heldout", "heldout_photo"])
def test_heldout_samples_outside_train_range(split):
    cfg = DRConfig()
    for s in range(200):
        p = sample_dr(split, np.random.default_rng(s), cfg)
        assert not cfg.in_train_range(p)
        assert np.all((np.abs(p.cam_pos_offset) > 0.03) & (np.abs(p.cam_pos_offset) <= 0.06)), s
        assert np.all((np.abs(p.cam_look_offset) > 0.02) & (np.abs(p.cam_look_offset) <= 0.04)), s
        assert 4 < abs(p.cam_fov_delta) <= 8, s
        for d in p.light_diffuse:
            assert (0.15 <= d < 0.4) or (1.0 < d <= 1.4), (s, d)
        assert (0.3 <= p.friction < 0.6) or (1.2 < p.friction <= 1.8), s
        assert 0.2 < p.cube_mass <= 0.4, s
        # kp: all 7 strictly outside train, inside wider band
        dk = np.abs(p.kp_scale - 1)
        assert np.all((dk > 0.10) & (dk <= 0.25)), (s, dk)
        # light direction: first n_lights AND all 3
        az, el = np.degrees(p.light_az), np.degrees(p.light_el)
        assert az.shape == (3,) and el.shape == (3,)
        for i in range(3):
            assert _light_ok_held(az[i], el[i]), (s, i, az[i], el[i])
            assert not _light_ok_train(az[i], el[i]), (s, i, az[i], el[i])
        # light_dir unit, downward
        assert np.allclose(np.linalg.norm(p.light_dir, axis=1), 1) and np.all(p.light_dir[:, 2] < 0)
        assert p.table_mat != p.floor_mat
        if split == "heldout":
            assert p.table_mat in HELDOUT_MATS and p.floor_mat in HELDOUT_MATS
        else:
            assert p.table_mat in HELDOUT_PHOTO_MATS and p.floor_mat in HELDOUT_PHOTO_MATS
            assert p.table_mat >= 42 and p.floor_mat >= 42


def test_material_id_sets_disjoint():
    assert set(HELDOUT_MATS) == set(range(8, 12))
    assert set(HELDOUT_PHOTO_MATS) == set(range(42, 59))
    assert set(TRAIN_MATS) == set(range(8)) | set(range(12, 42))
    assert not set(TRAIN_MATS) & set(HELDOUT_MATS)
    assert not set(TRAIN_MATS) & set(HELDOUT_PHOTO_MATS)
    assert not set(HELDOUT_MATS) & set(HELDOUT_PHOTO_MATS)


def test_heldout_photo_never_uses_train_or_procedural_ids():
    for s in range(300):
        p = sample_dr("heldout_photo", np.random.default_rng(s))
        assert min(p.table_mat, p.floor_mat) >= 42
        q = sample_dr("train", np.random.default_rng(s))
        assert max(q.table_mat, q.floor_mat) < 42
        h = sample_dr("heldout", np.random.default_rng(s))
        assert max(h.table_mat, h.floor_mat) < 12 and min(h.table_mat, h.floor_mat) >= 8


@pytest.mark.parametrize("split", ["heldout", "heldout_photo"])
def test_heldout_env_applies_heldout_params(split):
    e = TabletopEnv("lift", split)
    try:
        for s in range(5):
            e.reset(seed=s)
            m, p = e.model, e.dr
            assert not DRConfig().in_train_range(p)
            assert _matname(m, e.table_geom) == f"mat{p.table_mat}"
            assert _matname(m, e.floor_geom) == f"mat{p.floor_mat}"
            assert m.body_mass[e.cube_body[0]] > 0.2
            # kp actually written to model
            assert np.allclose(m.actuator_gainprm[:7, 0], e._orig["actuator_gainprm"][:7, 0] * p.kp_scale)
            # light dir actually written to model
            for i in range(3):
                assert np.allclose(m.light_dir[e.lights[i]], p.light_dir[i], atol=1e-6)
    finally:
        e.close()


def test_light_dir_changes_image_and_model_per_seed(env):
    dirs = []
    for s in range(4):
        env.reset(seed=s)
        dirs.append(env.model.light_dir[env.lights[0]].copy())
    assert len({tuple(np.round(d, 6)) for d in dirs}) == 4


def test_expert_ceiling_json():
    p = ROOT / "results" / "expert_ceiling.json"
    assert p.is_file()
    c = json.loads(p.read_text())["ceiling"]
    assert set(c) >= {"train", "heldout", "heldout_photo"}
    for split in ("train", "heldout", "heldout_photo"):
        for t in ("reach", "lift", "push"):
            r = c[split][t]
            assert 0 <= r["success"] <= 1 and len(r["ci95"]) == 2
            assert r["ci95"][0] <= r["success"] <= r["ci95"][1] + 1e-9


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        sample_dr("bogus", np.random.default_rng(0))


# ------------------------------------------------------------ obs / action / info contract
def test_obs_shapes_dtypes(env):
    obs, info = env.reset(seed=0)
    assert obs["image"].shape == (128, 128, 3) and obs["image"].dtype == np.uint8
    assert obs["proprio"].shape == (11,) and obs["proprio"].dtype == np.float32
    assert env.observation_space["image"].contains(obs["image"])
    assert env.observation_space["proprio"].contains(obs["proprio"])
    # proprio = EE pos (3), gripper width (1), 7 joint q
    assert np.allclose(obs["proprio"][:3], info["ee_pos"], atol=1e-6)
    assert np.allclose(obs["proprio"][4:], info["qpos"], atol=1e-6)
    assert obs["proprio"][3] == pytest.approx(env.data.qpos[7] + env.data.qpos[8], abs=1e-6)
    assert 0.0 <= obs["proprio"][3] <= 0.09
    assert obs["image"].std() > 5  # not a blank frame


def test_actions_clipped(env):
    env.reset(seed=0)
    env.step(np.array([50, -50, 50, 50], np.float32))
    ee_big = env.data.site_xpos[env.site].copy()
    env.reset(seed=0)
    env.step(np.array([1, -1, 1, 1], np.float32))
    ee_one = env.data.site_xpos[env.site].copy()
    assert np.allclose(ee_big, ee_one, atol=1e-9)
    assert env.action_space.low.min() == -1 and env.action_space.high.max() == 1 and env.action_space.shape == (4,)


def test_episode_truncates_at_100_steps():
    e = TabletopEnv("lift", "train", render_obs=False)
    try:
        e.reset(seed=0)
        n, done = 0, False
        while not done:
            _, _, term, trunc, info = e.step(np.zeros(4, np.float32))
            n += 1
            done = term or trunc
            assert n <= 100
        assert n == MAX_STEPS == 100 and trunc and not term and info["step"] == 100
    finally:
        e.close()


def test_info_keys(env):
    _, info = env.reset(seed=0)
    for k in ("success", "qacc", "qvel", "contacts", "ee_pos"):
        assert k in info
    _, _, _, _, info = env.step(np.zeros(4, np.float32))
    for k in ("success", "qacc", "qvel", "contacts", "ee_pos"):
        assert k in info
    assert info["qacc"].shape == (7,) and info["qvel"].shape == (7,) and info["ee_pos"].shape == (3,)
    assert isinstance(info["success"], (bool, np.bool_))
    assert {"arm_table", "arm_distractor"} <= set(info["contacts"])


@pytest.mark.parametrize("task", ["reach", "lift", "push"])
def test_instruction_names_target_and_cube_color_matches(task):
    e = TabletopEnv(task, "train", render_obs=False)
    try:
        seen = set()
        for s in range(30):
            _, info = e.reset(seed=s)
            c = e.target_color
            seen.add(c)
            assert c in info["instruction"]
            assert not any(o in info["instruction"] for o in CUBE_COLORS if o != c)
            rgb = e.model.geom_rgba[e.cube_geom[e.target_idx], :3]
            assert ("red", "green", "blue")[int(np.argmax(rgb))] == c
            assert e.model.geom_rgba[e.cube_geom[e.target_idx], :3].max() >= 0.65
        assert seen == set(CUBE_COLORS)
    finally:
        e.close()


# ------------------------------------------------------------ utils
def test_config_overrides(tmp_path):
    from vla.utils.config import load_config

    @dataclass
    class Opt:
        lr: float = 1e-4
        wd: float = 0.0

    @dataclass
    class Cfg:
        steps: int = 10
        name: str = "x"
        flag: bool = False
        opt: Opt = None

    p = tmp_path / "c.yaml"
    p.write_text("steps: 5\nopt: {lr: 0.1}\n")
    c = load_config(Cfg, str(p), ["steps=7", "flag=true", "opt.wd=3e-5", "name=abc"])
    assert c.steps == 7 and c.flag is True and c.name == "abc"
    assert c.opt.lr == 0.1 and c.opt.wd == pytest.approx(3e-5)
    with pytest.raises(KeyError):
        load_config(Cfg, None, ["nope=1"])
    with pytest.raises(ValueError):
        load_config(Cfg, None, ["bad"])


def test_metrics_logger_jsonl(tmp_path):
    from vla.utils.logging import MetricsLogger

    lg = MetricsLogger(tmp_path / "r")
    lg.log(1, {"loss": np.float32(0.5), "n": 3})
    lg.log(2, {"loss": 0.25})
    lg.close()
    rows = [json.loads(l) for l in (tmp_path / "r" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2] and rows[0]["loss"] == 0.5
    assert (tmp_path / "r" / "tb").exists()
    off = MetricsLogger(tmp_path / "off", enabled=False)
    off.log(1, {"a": 1}); off.close()
    assert not (tmp_path / "off").exists()


# ------------------------------------------------------------ static checks on remote scripts / config
def _scripts():
    return sorted((ROOT / "scripts" / "remote").glob("*.sh"))


def test_no_sudo_or_apt():
    assert _scripts()
    pat = re.compile(r"(^|[\s;&|(`])(sudo|apt|apt-get|dpkg|yum|dnf)(\s|$)")
    for f in _scripts():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            assert not pat.search(code), f"{f.name}:{i}: {line}"


def test_scripts_syntax():
    import subprocess
    for f in _scripts():
        assert subprocess.run(["bash", "-n", str(f)]).returncode == 0, f.name


def test_env_example_gpu_pinning():
    t = (ROOT / ".env.example").read_text().splitlines()
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in t
    assert "CUDA_VISIBLE_DEVICES=0,1,2,4" in t


def test_gpu_snapshot_exists():
    assert (ROOT / "scripts/remote/gpu_snapshot.sh").is_file()


def test_tensorboard_not_wandb():
    py = (ROOT / "pyproject.toml").read_text()
    assert "tensorboard" in py and "wandb" not in py.lower()
    for f in list((ROOT / "src").rglob("*.py")) + _scripts():
        assert "wandb" not in f.read_text().lower(), f


def test_cu130_index_pinned_for_linux():
    py = (ROOT / "pyproject.toml").read_text()
    assert "download.pytorch.org/whl/cu130" in py
    assert re.search(r"torch\s*=\s*\[\{[^}]*pytorch-cu130[^}]*sys_platform == 'linux'", py)


# ------------------------------------------------------------ semantic material checks (resolve names, not ids)
def _dtd_class(name):
    """'mat{i}' -> DTD class index (i-12) for photo mats, None for procedural mats 0-11."""
    assert re.fullmatch(r"mat\d+", name), name
    i = int(name[3:])
    return i - 12 if i >= 12 else None


@pytest.mark.parametrize("split", ["train", "heldout", "heldout_photo"])
def test_material_names_per_split(split):
    e = TabletopEnv("lift", split, render_obs=False)
    try:
        for s in range(40):
            e.reset(seed=s)
            for g in (e.table_geom, e.floor_geom):
                name = _matname(e.model, g)
                assert name is not None and re.fullmatch(r"mat\d+", name), name  # never a Panda material
                i, k = int(name[3:]), _dtd_class(name)
                if split == "train":
                    assert i < 8 or (k is not None and 0 <= k <= 29), name
                elif split == "heldout":
                    assert 8 <= i <= 11, name
                else:
                    assert k is not None and 30 <= k <= 46, name
    finally:
        e.close()


def test_panda_materials_never_reassigned():
    e = TabletopEnv("lift", "train", render_obs=False)
    try:
        base = e.model.geom_matid.copy()
        free = {e.table_geom, e.floor_geom}
        for s in range(10):
            e.reset(seed=s)
            diff = {int(g) for g in np.nonzero(e.model.geom_matid != base)[0]}
            assert diff <= free, diff
        panda = {mujoco.mj_id2name(e.model, mujoco.mjtObj.mjOBJ_MATERIAL, int(i)) for i in base if i >= 0} - {None}
        assert panda - {f"mat{i}" for i in range(59)}, "expected Panda-own materials to exist"
        for g in range(e.model.ngeom):
            if g not in free and base[g] >= 0:
                n = mujoco.mj_id2name(e.model, mujoco.mjtObj.mjOBJ_MATERIAL, int(base[g]))
                assert not re.fullmatch(r"mat\d+", n or ""), (g, n)  # DR mats only on table/floor
    finally:
        e.close()
