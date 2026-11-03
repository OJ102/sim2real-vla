"""load_weights dropping ground_head.*, RL init from a grounded txt_first BC ckpt, rollout_eval --reward-components, episode_terms."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vla.infer.policy import load_weights
from vla.model.vla import VLA, VLAConfig
from vla.sim.rewards import DEFAULT_WEIGHTS, JOINT_HI, JOINT_LO, QD_MAX, episode_return, episode_terms
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}


def tiny(gw, seed=0, order=None):
    torch.manual_seed(seed)
    ov = [f"ground_weight={gw}"] + ([f"prefix_order={order}"] if order else [])
    return VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ov))


# ------------------------------------------------------------------ (1) load_weights
def test_load_weights_drops_ground_head_for_headless_model(tmp_path, capsys):
    src = tiny(0.2, seed=3)
    p = tmp_path / "g.pt"; torch.save(src.state_dict(), p)
    n_gh = sum(1 for k in src.state_dict() if k.startswith("ground_head."))
    assert n_gh > 0
    dst = tiny(0.0, seed=9)
    load_weights(dst, str(p))
    assert f"load_weights: dropped {n_gh} ground_head.* keys" in capsys.readouterr().out
    assert not any(k.startswith("ground_head.") for k in dst.state_dict())
    for k, v in dst.state_dict().items():
        assert torch.equal(v, src.state_dict()[k]), k


def test_load_weights_keeps_ground_head_when_model_has_one(tmp_path, capsys):
    src = tiny(0.2, seed=3); p = tmp_path / "g.pt"; torch.save(src.state_dict(), p)
    dst = tiny(0.2, seed=9)
    load_weights(dst, str(p))
    assert "dropped" not in capsys.readouterr().out
    assert all(torch.equal(v, dst.state_dict()[k]) for k, v in src.state_dict().items())


def test_load_weights_still_raises_on_genuinely_wrong_keys(tmp_path):
    src = tiny(0.0, seed=1)
    sd = src.state_dict()
    bad_extra = dict(sd); bad_extra["bogus.weight"] = torch.zeros(2)
    bad_missing = {k: v for k, v in sd.items() if k != next(k for k in sd if k.startswith("llm.model.layers.0"))}
    only_ground_plus_extra = {**tiny(0.2).state_dict(), "bogus.weight": torch.zeros(2)}
    for name, d in (("extra", bad_extra), ("missing", bad_missing), ("ground+extra", only_ground_plus_extra)):
        p = tmp_path / f"{name.replace('+', '_')}.pt"; torch.save(d, p)
        with pytest.raises(RuntimeError, match="checkpoint mismatch"):
            load_weights(tiny(0.0), str(p))


# ------------------------------------------------------------------ end to end: grpo init from a grounded txt_first bc ckpt
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def torchrun(module, *args, timeout=600):
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master_addr=127.0.0.1",
                        f"--master_port={port()}", "-m", module, *args], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2500:]
    return r


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("rp_demos")
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={out}", "episodes_per_combo=4", "chunk_size=2", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0
    return out


def test_grpo_init_ckpt_from_grounded_txt_first_bc(demos, tmp_path):
    cfg = tmp_path / "model_tiny_txt.yaml"
    cfg.write_text((ROOT / "configs/model_tiny.yaml").read_text() + "\nprefix_order: txt_first\n")
    bc = torchrun("vla.train.bc", "--config", "configs/bc_tiny.yaml", f"model_config={cfg}", f"data_dir={demos}",
                  f"runs_dir={tmp_path / 'bc'}", "name=g", "steps=4", "ground_weight=0.2", "ckpt_every=100")
    final = tmp_path / "bc" / "g" / "final.pt"
    sd = torch.load(final, weights_only=True)
    n_gh = sum(1 for k in sd if k.startswith("ground_head."))
    assert n_gh > 0
    r = torchrun("vla.rl.grpo", "--config", "configs/rl_tiny.yaml", f"model_config={cfg}", f"init_ckpt={final}",
                 f"runs_dir={tmp_path / 'rl'}", f"shm_dir={tmp_path / 'shm'}", "iterations=2")
    assert r.stdout.count(f"dropped {n_gh} ground_head.* keys") >= 2, r.stdout[-1500:]     # both ranks (learner + rollout)
    rec = [json.loads(l) for l in (tmp_path / "rl" / "rl_tiny" / "metrics.jsonl").read_text().splitlines()]
    assert len(rec) == 2 and all(np.isfinite(x["loss"]) and x["engine_mismatch_equal_version"] < 1e-4 for x in rec)


# ------------------------------------------------------------------ episode_terms / episode_return
def infos(T, ee=None, q=None, qd=None, col=None, success=False):
    ee = np.zeros((T, 3)) if ee is None else ee
    q = np.tile((JOINT_LO + JOINT_HI) / 2, (T, 1)) if q is None else q
    qd = np.zeros((T, 7)) if qd is None else qd
    col = np.zeros(T, int) if col is None else col
    return [{"ee_pos": ee[i], "qpos": q[i], "qvel": qd[i], "success": success and i == T - 1,
             "contacts": {"arm_table": int(col[i]), "arm_distractor": 0}} for i in range(T)]


def test_episode_terms_consistent_with_episode_return():
    rng = np.random.default_rng(0)
    for T in (1, 3, 4, 12):
        a = rng.uniform(-1, 1, (T, 4)); ee = rng.normal(size=(T, 3)) * 0.01
        q = rng.uniform(JOINT_LO, JOINT_HI, (T, 7)); qd = rng.normal(size=(T, 7)) * 2; col = rng.integers(0, 2, T)
        inf = infos(T, ee, q, qd, col, success=bool(T % 2))
        t = episode_terms(a, inf)
        assert set(t) == {"success", "jerk_act", "jerk_ee", "limit", "vel", "collision"} and all(v >= 0 for v in t.values())
        R, comp = episode_return(a, inf)
        assert comp["success"] == DEFAULT_WEIGHTS["success"] * t["success"]
        for k in ("jerk_act", "jerk_ee", "limit", "vel", "collision"):
            assert comp[k] == pytest.approx(-DEFAULT_WEIGHTS[k] * t[k])
        assert R == pytest.approx(sum(comp.values()))
        w = {"limit": 0.5, "success": 1.0}
        assert episode_return(a, inf, w)[1]["limit"] == pytest.approx(-0.5 * t["limit"])
    # the raw sums follow their definitions
    T = 6; a = np.zeros((T, 4)); a[::2, 0] = 1.0
    d2 = a[2:] - 2 * a[1:-1] + a[:-2]
    assert episode_terms(a, infos(T))["jerk_act"] == pytest.approx(float((d2**2).sum()))
    qd = np.zeros((T, 7)); qd[2, 0] = 0.8 * QD_MAX[0] + 0.7
    assert episode_terms(np.zeros((T, 4)), infos(T, qd=qd))["vel"] == pytest.approx(0.7)


# ------------------------------------------------------------------ (2) rollout_eval --reward-components
def reval(tmp_path, tag, *extra):
    out = tmp_path / f"{tag}.json"
    r = subprocess.run([sys.executable, "-m", "vla.eval.rollout_eval", "--policy", "expert", "--splits", "train", "--tasks", "lift,push",
                        "--episodes", "4", "--n-envs", "2", "--out", str(out), *extra], cwd=ROOT, env=ENV, capture_output=True,
                       text=True, timeout=420)
    assert r.returncode == 0, r.stdout[-1000:] + r.stderr[-2000:]
    return json.loads(out.read_text())


def test_reward_components_in_rollout_eval(tmp_path):
    on, off = reval(tmp_path, "on", "--reward-components"), reval(tmp_path, "off")
    keys5 = ["jerk_act", "jerk_ee", "limit", "vel", "collision"]
    for task in ("lift", "push"):
        rc = on["results"]["train"][task]["reward_components"]
        assert set(rc) == {"total", "components", "raw_sums"}
        assert set(rc["components"]) == {"success", *keys5} and set(rc["raw_sums"]) == set(keys5)
        assert {"mean", "std"} <= set(rc["total"]) and all({"mean", "std"} <= set(v) for v in rc["components"].values())
        assert all({"mean", "std"} <= set(v) for v in rc["raw_sums"].values())
        c = rc["components"]
        assert c["success"]["mean"] >= 0 and all(c[k]["mean"] <= 1e-12 for k in keys5)          # signed
        assert sum(v["mean"] for v in c.values()) == pytest.approx(rc["total"]["mean"], abs=1e-9)
        for k in keys5:                                                                        # -comp = w * raw (means)
            assert -c[k]["mean"] == pytest.approx(DEFAULT_WEIGHTS[k] * rc["raw_sums"][k]["mean"], rel=1e-9, abs=1e-12)
            assert c[k]["std"] == pytest.approx(DEFAULT_WEIGHTS[k] * rc["raw_sums"][k]["std"], rel=1e-9, abs=1e-12)
        assert c["success"]["mean"] == pytest.approx(DEFAULT_WEIGHTS["success"] * on["results"]["train"][task]["success"])
        # without the flag: no reward_components, every other field identical
        a, b = dict(on["results"]["train"][task]), dict(off["results"]["train"][task])
        assert "reward_components" not in b
        a.pop("reward_components")
        a.pop("eval_secs", None); b.pop("eval_secs", None)
        assert a == b
    assert on["meta"]["seed0"] == off["meta"]["seed0"]
