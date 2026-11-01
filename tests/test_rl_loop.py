"""P6 loop tests (CPU gloo, tiny): GRPO runs S=0/S=1, episode_return, load_batches, grpo_loss gradient."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vla.rl.algo import grpo_loss
from vla.rl.learner import load_batches
from vla.sim.rewards import DEFAULT_WEIGHTS, JOINT_HI, JOINT_LO, QD_MAX, episode_return

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}
ENV.pop("VLA_RUNS", None)


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def grpo(tmp, name, *extra, timeout=600):
    runs, shm = tmp / "runs", tmp / "shm"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={_port()}", "-m", "vla.rl.grpo", "--config", "configs/rl_tiny.yaml", f"runs_dir={runs}",
           f"shm_dir={shm}", f"name={name}", *extra]
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return r, runs / name, shm / name


def recs(run):
    return [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------ 1) S=0
@pytest.fixture(scope="module")
def s0(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rl_s0")
    r, run, shm = grpo(tmp, "rl_tiny")
    return r, run, shm


def test_s0_artifacts_and_metrics(s0):
    r, run, shm = s0
    assert (run / "final.pt").is_file() and (run / "config.yaml").is_file() and (run / "metrics.jsonl").is_file()
    rec = recs(run)
    assert [x["iter"] for x in rec] == [0, 1]
    need = ["return_mean", "success", "loss", "kl", "tis_w_mean", "tis_w_max", "ratio_clipfrac", "grad_norm",
            "engine_mismatch_equal_version", "sync_latency_ms", "learner_idle_frac", "env_steps_per_s",
            "learner_tokens_per_s", "staleness_hist", "iter_time_s", "n_samples"]
    for x in rec:
        for k in need:
            assert k in x, k
        for k in ("return_mean", "success", "loss", "kl", "grad_norm", "env_steps_per_s", "learner_tokens_per_s"):
            assert np.isfinite(x[k]), (k, x[k])
        assert x["kl"] >= -1e-9 and x["env_steps_per_s"] > 0 and 0 <= x["learner_idle_frac"] <= 1
        assert {"R_success", "R_jerk_act", "R_jerk_ee", "R_limit", "R_vel", "R_collision"} <= set(x)
        # S=0: everything on-policy -> importance weight exactly 1 and all samples at gap 0
        assert x["tis_w_mean"] == pytest.approx(1.0, abs=1e-3) and x["tis_w_max"] < 1.01
        assert x["staleness_hist"] == {"0": x["n_samples"]}
        assert x["engine_mismatch_equal_version"] < 1e-4, x["engine_mismatch_equal_version"]
    assert all(x["sync_latency_ms"] > 0 for x in rec)                       # S=0: K-S = 2 syncs, after every update
    # return = sum of components
    for x in rec:
        comp = sum(x[k] for k in x if k.startswith("R_"))
        assert x["return_mean"] == pytest.approx(comp, rel=1e-4, abs=1e-4)


def test_s0_initial_sync_and_log_lines(s0):
    r, run, shm = s0
    assert "[learner] initial sync done" in r.stdout and "[rollout 1] initial sync v0" in r.stdout
    assert "[rollout 1] batch 0 v0" in r.stdout and "[rollout 1] batch 1 v1" in r.stdout    # S=0: v_k for batch k
    assert "[learner] done: 2 iterations" in r.stdout


def test_batch_files_deleted_after_consumption(s0):
    _, run, shm = s0
    assert shm.exists() and not list(shm.glob("*batch*")), list(shm.iterdir())


# ------------------------------------------------------------------ 2) S=1
def test_s1_staleness_never_exceeds_one(tmp_path):
    r, run, shm = grpo(tmp_path, "rl_s1", "staleness=1", "iterations=4")
    rec = recs(run)
    assert [x["iter"] for x in rec] == [0, 1, 2, 3]
    n = rec[0]["n_samples"]
    assert rec[0]["staleness_hist"] == {"0": n}
    for x in rec[1:]:
        assert x["staleness_hist"] == {"1": n}, x["staleness_hist"]          # exactly one-step off-policy, never >1
    # batches were produced with versions v0, v0, v1, v2
    got = [l.split("batch ")[1].split(" ")[:2] for l in r.stdout.splitlines() if l.startswith("[rollout 1] batch")]
    assert [g[1] for g in got] == ["v0", "v0", "v1", "v2"], got
    # off-policy: TIS actually engaged, finite
    assert all(np.isfinite(x["tis_w_mean"]) and x["tis_w_max"] <= 2.0 + 1e-6 for x in rec)
    assert not list(shm.glob("*batch*")) and (run / "final.pt").is_file()
    # sync count: K-S = 3 syncs (after updates 0,1,2), none after the last
    assert [x["sync_latency_ms"] > 0 for x in rec] == [True, True, True, False]


# ------------------------------------------------------------------ 4) learning happens
def test_learner_weights_move(tmp_path):
    from vla.model.vla import VLA, VLAConfig
    from vla.utils.config import load_config

    _, run, _ = grpo(tmp_path, "rl_lr", "lr=3e-3", "iterations=2")
    sd = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(0)                                                # RLConfig.seed=0 -> same init as the roles
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ["head=discrete"]))
    init = {k: v.detach().to(torch.bfloat16) for k, v in m.state_dict().items()}
    moved = [k for k, v in sd.items() if v.is_floating_point() and not torch.equal(v, init[k])
             and not k.startswith("vision.")]
    assert len(moved) > 5, moved
    vis = [k for k, v in sd.items() if k.startswith("vision.") and v.is_floating_point() and not torch.equal(v, init[k])]
    assert not vis, f"vision encoder must stay frozen, changed: {vis[:3]}"
    rec = recs(run)
    assert rec[0]["engine_mismatch_equal_version"] < 1e-4              # rollout started from the learner's weights


# ------------------------------------------------------------------ 3) episode_return
def _infos(T, ee=None, q=None, qd=None, col=None, success=False):
    ee = np.zeros((T, 3)) if ee is None else ee
    q = np.tile((JOINT_LO + JOINT_HI) / 2, (T, 1)) if q is None else q
    qd = np.zeros((T, 7)) if qd is None else qd
    col = np.zeros(T, int) if col is None else col
    return [{"ee_pos": ee[i], "qpos": q[i], "qvel": qd[i], "success": success and i == T - 1,
             "contacts": {"arm_table": int(col[i]), "arm_distractor": 0, "total": 0}} for i in range(T)]


def test_return_zero_penalties_when_smooth():
    T = 10
    R, c = episode_return(np.tile([0.2, 0.1, -0.3, 0.5], (T, 1)), _infos(T, ee=np.tile([0.4, 0, 0.2], (T, 1))))
    assert R == 0.0 and all(v == 0.0 for v in c.values()) and set(c) == set(DEFAULT_WEIGHTS)
    R, c = episode_return(np.zeros((T, 4)), _infos(T, success=True))
    assert R == pytest.approx(10.0) and c["success"] == 10.0


def test_return_components_sum_and_each_term():
    T = 8
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (T, 4))
    ee = rng.normal(size=(T, 3)) * 0.01
    q = np.tile((JOINT_LO + JOINT_HI) / 2, (T, 1))
    q[2, 0] = JOINT_LO[0] + 0.01 * (JOINT_HI[0] - JOINT_LO[0]); q[5, 3] = JOINT_HI[3]; q[6, 1] = JOINT_LO[1] + 0.2
    qd = np.zeros((T, 7)); qd[1, 0] = 0.8 * QD_MAX[0] + 0.5; qd[3, 6] = -(0.8 * QD_MAX[6] + 0.25)
    col = np.array([0, 1, 0, 0, 1, 1, 0, 0])
    R, c = episode_return(a, _infos(T, ee, q, qd, col, success=True))
    w = DEFAULT_WEIGHTS
    assert R == pytest.approx(sum(c.values()))
    assert c["success"] == w["success"]
    d2 = a[2:] - 2 * a[1:-1] + a[:-2]
    assert c["jerk_act"] == pytest.approx(-w["jerk_act"] * (d2**2).sum())
    d3 = np.diff(ee, n=3, axis=0)
    assert c["jerk_ee"] == pytest.approx(-w["jerk_ee"] * (d3**2).sum())
    assert c["limit"] == pytest.approx(-w["limit"] * 2)                  # two steps within 5% of a limit
    assert c["vel"] == pytest.approx(-w["vel"] * (0.5 + 0.25))
    assert c["collision"] == pytest.approx(-w["collision"] * 3)
    # custom weights override, others keep defaults
    R2, c2 = episode_return(a, _infos(T, ee, q, qd, col, success=True), {"limit": 1.0, "success": 0.0})
    assert c2["limit"] == -2.0 and c2["success"] == 0.0 and c2["collision"] == c["collision"]


def test_return_consistent_with_eval_metrics():
    from vla.eval.metrics import episode_metrics

    rng = np.random.default_rng(1)
    for _ in range(5):
        T = 12
        q = rng.uniform(JOINT_LO, JOINT_HI, (T, 7)); qd = rng.normal(size=(T, 7)) * 2
        col = rng.integers(0, 2, T)
        infos = _infos(T, rng.normal(size=(T, 3)), q, qd, col, True)
        a = rng.uniform(-1, 1, (T, 4))
        _, c = episode_return(a, infos); m = episode_metrics(a, infos)
        assert -c["limit"] / DEFAULT_WEIGHTS["limit"] == pytest.approx(m["joint_limit_viol"])
        assert -c["collision"] / DEFAULT_WEIGHTS["collision"] == pytest.approx(m["collision_viol"])


@pytest.mark.parametrize("T", [1, 2, 3])
def test_return_short_episodes_no_crash(T):
    R, c = episode_return(np.zeros((T, 4)), _infos(T, success=True))
    assert np.isfinite(R) and c["jerk_ee"] == 0.0 and c["jerk_act"] == 0.0 and c["success"] == 10.0


# ------------------------------------------------------------------ 7) load_batches advantage grouping
def _write_batch(path, returns, G, version, n_tok=16, seed=0):
    rng = np.random.default_rng(seed)
    E = len(returns)
    lens = rng.integers(2, 5, E)
    ep = np.repeat(np.arange(E), lens)
    N = len(ep)
    np.savez(path, image=np.zeros((N, 8, 8, 3), np.uint8), proprio=np.zeros((N, 11), np.float32),
             bins=np.zeros((N, n_tok), np.int64), logp=np.zeros((N, n_tok), np.float32), ep=ep,
             ep_return=np.array(returns, np.float32), ep_success=np.zeros(E, np.float32),
             instr=np.array([f"i{e}" for e in range(E)]), version=np.int64(version),
             comp_success=np.zeros(E, np.float32), meta=np.array(json.dumps({"env_steps": 1, "gen_time": 1.0, "idle_s": 0.0})))
    return ep


def test_load_batches_group_advantages(tmp_path):
    G = 4
    r0 = [1, 2, 3, 6,  5, 5, 5, 5]                       # group 0 varied, group 1 constant -> adv 0
    r1 = [0, 0, 10, 10,  -1, 0, 1, 2]
    e0, e1 = _write_batch(tmp_path / "a.npz", r0, G, 3, seed=1), _write_batch(tmp_path / "b.npz", r1, G, 4, seed=2)
    data, es, metas = load_batches([tmp_path / "a.npz", tmp_path / "b.npz"], G)
    assert len(data["adv"]) == len(e0) + len(e1) == len(data["bins"]) == len(data["instr"])
    adv = data["adv"]
    n0 = len(e0)
    for sl, ep, rets in ((slice(0, n0), e0, r0), (slice(n0, None), e1, r1)):
        a, e = adv[sl], ep
        for g in range(len(rets) // G):
            eps_in_g = [g * G + k for k in range(G)]
            per_ep = np.array([a[e == x][0] for x in eps_in_g])
            assert all(np.allclose(a[e == x], a[e == x][0]) for x in eps_in_g)         # one advantage per episode
            assert per_ep.mean() == pytest.approx(0, abs=1e-5)
            rr = np.array([rets[x] for x in eps_in_g], float)
            if rr.std() > 0:
                assert per_ep.std() == pytest.approx(1.0, abs=1e-3)
                assert np.array_equal(np.argsort(per_ep), np.argsort(rr))              # order preserved
            else:
                assert np.abs(per_ep).max() == 0 and np.isfinite(per_ep).all()
    assert (data["version"][:n0] == 3).all() and (data["version"][n0:] == 4).all()
    assert np.concatenate(es["ret"]).tolist() == r0 + r1 and metas[0]["env_steps"] == 1
    assert data["instr"][0] == "i0" and data["instr"][n0] == "i0"


# ------------------------------------------------------------------ 6) grpo_loss gradient vs hand-computed PPO-clip
def _hand_grad(new, old, beh, ref, adv, eps=0.2, beta=0.02, cap=2.0):
    N = new.size
    r = np.exp(new - old)
    w = np.minimum(np.exp(old - beh), cap)
    active = np.where(adv >= 0, r <= 1 + eps, r >= 1 - eps)                 # pessimistic min picks r*A, else clipped (grad 0)
    g_pg = -w * adv * r * active
    d = ref - new
    g_kl = beta * (1 - np.exp(d))
    return (g_pg + g_kl) / N


def test_grpo_loss_gradient_matches_hand_computation():
    rng = np.random.default_rng(0)
    old = rng.normal(-2, 0.3, (3, 5))
    new = old + rng.normal(0, 0.3, (3, 5))                                  # ratios spread across the clip boundary
    new[0, 0], new[1, 1] = old[0, 0] + 0.5, old[1, 1] - 0.5
    beh = old + rng.normal(0, 0.5, (3, 5)); beh[2, 2] = old[2, 2] - 3       # one TIS-capped token
    ref = old + rng.normal(0, 0.2, (3, 5))
    adv = np.array([1.0, -1.0, 0.5])
    advb = np.broadcast_to(adv[:, None], new.shape)
    t = lambda x: torch.tensor(x, dtype=torch.float64)                     # noqa: E731
    tn = t(new).requires_grad_()
    loss, m = grpo_loss(tn, t(old), t(beh), t(ref), t(adv))
    loss.backward()
    np.testing.assert_allclose(tn.grad.numpy(), _hand_grad(new, old, beh, ref, advb), atol=1e-9)
    # value check against the hand formula too
    r = np.exp(new - old); w = np.minimum(np.exp(old - beh), 2.0)
    pg = np.minimum(r * advb, np.clip(r, 0.8, 1.2) * advb)
    kl = np.exp(ref - new) - (ref - new) - 1
    assert loss.item() == pytest.approx((-pg * w + 0.02 * kl).mean(), rel=1e-9)
    assert m["tis_w_max"] == pytest.approx(w.max())
    # clipped tokens really have zero policy gradient (kl-only gradient)
    clipped = ((advb >= 0) & (r > 1.2)) | ((advb < 0) & (r < 0.8))
    assert clipped.any() and np.allclose(tn.grad.numpy()[clipped], (0.02 * (1 - np.exp(ref - new)) / new.size)[clipped])


def test_s0_temperature_not_one_stays_on_policy(tmp_path):
    _, run, _ = grpo(tmp_path, "rl_t07", "temperature=0.7")
    for x in recs(run):
        assert x["engine_mismatch_equal_version"] < 1e-4, x["engine_mismatch_equal_version"]
        assert x["tis_w_mean"] == pytest.approx(1.0, abs=1e-3) and np.isfinite(x["loss"])
