"""P5 tests (CPU, tiny): eval metrics, rollout_eval CLI, GPU snapshot/quiet_wait (mocked nvidia-smi), SubprocVecEnv."""
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vla.eval.metrics import DT, JOINT_HI, JOINT_LO, QD_MAX, aggregate, episode_metrics, wilson
from vla.utils import gpu

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 1) metrics
def test_wilson_known_values():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson(0, 10); assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(10, 10); assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    lo, hi = wilson(98, 100); assert lo == pytest.approx(0.9300, abs=2e-3) and hi == pytest.approx(0.9945, abs=2e-3)


def test_wilson_monotone_and_ordered():
    prev = (-1, -1)
    for k in range(0, 51):
        lo, hi = wilson(k, 50)
        assert 0 <= lo <= k / 50 <= hi <= 1 and lo >= prev[0] and hi >= prev[1]
        prev = (lo, hi)
    assert wilson(50, 500)[1] - wilson(50, 500)[0] < wilson(5, 50)[1] - wilson(5, 50)[0]   # narrower with n


def _infos(T, ee=None, q=None, qd=None, arm_table=0, arm_dist=0, success=True):
    ee = np.zeros((T, 3)) if ee is None else ee
    q = np.tile((JOINT_LO + JOINT_HI) / 2, (T, 1)) if q is None else q
    qd = np.zeros((T, 7)) if qd is None else qd
    return [{"ee_pos": ee[i], "qpos": q[i], "qvel": qd[i], "success": success and i == T - 1,
             "contacts": {"arm_table": arm_table if np.isscalar(arm_table) else arm_table[i],
                          "arm_distractor": arm_dist if np.isscalar(arm_dist) else arm_dist[i], "total": 0}}
            for i in range(T)]


def test_episode_metrics_smooth_is_zero():
    T = 10
    const = np.tile([0.3, -0.2, 0.7, 0.5], (T, 1))
    m = episode_metrics(const, _infos(T, ee=np.tile([0.4, 0.0, 0.2], (T, 1))))
    assert m["action_2nd_diff"] == 0 and m["ee_jerk_rms"] == pytest.approx(0, abs=1e-9)
    t = np.arange(T)[:, None] * DT
    lin = np.hstack([t, 2 * t, -t])
    quad = np.hstack([t**2, t, t])
    assert episode_metrics(const, _infos(T, ee=lin))["ee_jerk_rms"] == pytest.approx(0, abs=1e-6)
    assert episode_metrics(const, _infos(T, ee=quad))["ee_jerk_rms"] == pytest.approx(0, abs=1e-6)
    # linearly ramping action also has zero 2nd difference
    ramp = np.outer(np.arange(T), [0.01, 0, 0, 0])
    assert episode_metrics(ramp, _infos(T))["action_2nd_diff"] == pytest.approx(0, abs=1e-12)


def test_episode_metrics_cubic_jerk_and_second_diff():
    T = 12
    t = np.arange(T) * DT
    ee = np.zeros((T, 3)); ee[:, 0] = t**3                         # d3x/dt3 = 6
    assert episode_metrics(np.zeros((T, 4)), _infos(T, ee=ee))["ee_jerk_rms"] == pytest.approx(6.0, rel=1e-6)
    ee3 = np.stack([t**3, t**3, t**3], 1)                           # 3 axes -> sqrt(3*36)
    assert episode_metrics(np.zeros((T, 4)), _infos(T, ee=ee3))["ee_jerk_rms"] == pytest.approx(6 * math.sqrt(3), rel=1e-6)
    # alternating action +-1 on one dim: |a[t+1]-2a[t]+a[t-1]| = 4 on that dim, mean over 4 dims = 1
    a = np.zeros((T, 4)); a[::2, 0], a[1::2, 0] = 1, -1
    assert episode_metrics(a, _infos(T))["action_2nd_diff"] == pytest.approx(1.0)


def test_episode_metrics_violations():
    T = 6
    q = np.tile((JOINT_LO + JOINT_HI) / 2, (T, 1))
    q[1, 0] = JOINT_LO[0] + 0.01 * (JOINT_HI[0] - JOINT_LO[0])      # inside 5% margin
    q[2, 3] = JOINT_HI[3] - 0.04 * (JOINT_HI[3] - JOINT_LO[3])      # inside
    q[3, 1] = JOINT_LO[1] + 0.06 * (JOINT_HI[1] - JOINT_LO[1])      # outside margin (6%) -> not counted
    qd = np.zeros((T, 7))
    qd[0, 2] = 0.81 * QD_MAX[2]; qd[4, 5] = -0.9 * QD_MAX[5]; qd[5, 0] = 0.79 * QD_MAX[0]
    col = np.array([0, 1, 0, 2, 0, 0])
    m = episode_metrics(np.zeros((T, 4)), _infos(T, q=q, qd=qd, arm_table=col, arm_dist=np.array([0, 0, 1, 0, 0, 0])))
    assert m["joint_limit_viol"] == 2 and m["vel_viol"] == 2 and m["collision_viol"] == 3
    assert m["steps"] == T and m["success"] is True
    clean = episode_metrics(np.zeros((T, 4)), _infos(T, success=False))
    assert clean["joint_limit_viol"] == clean["vel_viol"] == clean["collision_viol"] == 0 and clean["success"] is False


@pytest.mark.parametrize("T", [1, 2, 3])
def test_episode_metrics_short_episodes_nan_not_crash(T):
    m = episode_metrics(np.zeros((T, 4)), _infos(T))
    assert math.isnan(m["ee_jerk_rms"])                              # needs >=4 steps
    assert math.isnan(m["action_2nd_diff"]) == (T < 3) and m["steps"] == T
    ag = aggregate([m, episode_metrics(np.zeros((8, 4)), _infos(8))])  # nanmean must skip the NaNs
    assert not math.isnan(ag["ee_jerk_rms"]) and ag["n"] == 2


def test_aggregate_ceiling_and_empty():
    eps = [{"success": s, "steps": 10, "ee_jerk_rms": 1.0, "action_2nd_diff": 0.1, "joint_limit_viol": 0,
            "vel_viol": 1, "collision_viol": 2} for s in (True, True, False, False)]
    a = aggregate(eps, 0.8)
    assert a["n"] == 4 and a["successes"] == 2 and a["success"] == 0.5 and a["success_vs_ceiling"] == pytest.approx(0.625)
    assert a["expert_ceiling"] == 0.8 and a["ci95"] == wilson(2, 4)
    assert a["vel_viol_per_ep"] == 1 and a["collision_viol_per_ep"] == 2
    for c in (None, 0):
        b = aggregate(eps, c)
        assert "success_vs_ceiling" not in b and b["success"] == 0.5
    e = aggregate([], 1.0)
    assert e["n"] == 0 and e["success"] == 0 and e["ci95"] == (0.0, 1.0)


# ------------------------------------------------------------------ 2) rollout_eval CLI
def _env():
    e = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}
    e.pop("CUDA_VISIBLE_DEVICES", None)
    return e


def _eval(out, *args, timeout=420):
    r = subprocess.run([sys.executable, "-m", "vla.eval.rollout_eval", *args, "--out", str(out)], cwd=ROOT, env=_env(),
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return json.loads(Path(out).read_text())


KEYS = {"n", "successes", "success", "ci95", "success_vs_ceiling", "ee_jerk_rms", "action_2nd_diff",
        "joint_limit_viol_per_ep", "vel_viol_per_ep", "collision_viol_per_ep"}


def test_expert_cli_schema_and_values(tmp_path):
    d = _eval(tmp_path / "e.json", "--policy", "expert", "--splits", "train,heldout_photo", "--tasks", "lift",
              "--episodes", "4", "--n-envs", "2")
    meta = d["meta"]
    for k in ("gpu_snapshot_before", "gpu_snapshot_after", "contended", "seed0"):
        assert k in meta
    assert meta["contended"] is False and meta["gpu_snapshot_before"]["available"] is False or sys.platform != "darwin"
    assert set(d["results"]) == {"train", "heldout_photo"}
    for split in d["results"]:
        r = d["results"][split]["lift"]
        assert KEYS <= set(r), KEYS - set(r)
        assert r["n"] == 4 and 0 <= r["ci95"][0] <= r["success"] <= r["ci95"][1] <= 1
        assert r["ee_jerk_rms"] > 0 and r["action_2nd_diff"] >= 0 and r["mean_steps"] > 0
    assert d["results"]["train"]["lift"]["success"] >= 0.75              # expert ~1.0 (n=4)
    ceil = json.loads((ROOT / "results/expert_ceiling.json").read_text())["ceiling"]["train"]["lift"]["success"]
    assert d["results"]["train"]["lift"]["success_vs_ceiling"] == pytest.approx(d["results"]["train"]["lift"]["success"] / ceil)


def test_expert_deterministic_and_seeds(tmp_path):
    a = _eval(tmp_path / "a.json", "--policy", "expert", "--splits", "heldout", "--tasks", "push", "--episodes", "4",
              "--n-envs", "2", "--seed0", "1234")
    b = _eval(tmp_path / "b.json", "--policy", "expert", "--splits", "heldout", "--tasks", "push", "--episodes", "4",
              "--n-envs", "4", "--seed0", "1234")                       # different worker count, same results
    strip = lambda r: {k: v for k, v in r.items() if k != "eval_secs"}   # noqa: E731
    assert strip(a["results"]["heldout"]["push"]) == strip(b["results"]["heldout"]["push"])
    assert a["meta"]["seed0"] == 1234
    c = _eval(tmp_path / "c.json", "--policy", "expert", "--splits", "heldout", "--tasks", "push", "--episodes", "4",
              "--n-envs", "2", "--seed0", "5000")
    assert strip(c["results"]["heldout"]["push"]) != strip(a["results"]["heldout"]["push"])


def test_eval_seeds_below_training_seed_base():
    from vla.eval.rollout_eval import EVAL_SEED0
    assert 0 <= EVAL_SEED0 and EVAL_SEED0 + 5000 < 100000                 # demo seeds start at 100000


def test_vec_expert_matches_direct_run():
    from vla.eval.rollout_eval import EVAL_SEED0, run_expert
    from vla.sim.env import TabletopEnv
    from vla.sim.expert import run_episode
    from vla.sim.vec_env import SubprocVecEnv

    vec = SubprocVecEnv(2, require_nvidia=False)
    try:
        eps = run_expert(vec, "lift", "train", list(range(EVAL_SEED0, EVAL_SEED0 + 3)))
    finally:
        vec.close()
    env = TabletopEnv("lift", "train", render_obs=False)
    for i, e in enumerate(eps):
        ref = run_episode(env, EVAL_SEED0 + i)
        assert (e["success"], e["steps"]) == (ref["success"], ref["steps"]), i
    env.close()


# --- VLA policy through the CLI: needs demos (stats.json) and a trained tiny ckpt
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("vla_eval")
    demos, runs = root / "demos", root / "runs"
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={demos}", "episodes_per_combo=2", "chunk_size=2", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    for head in ("discrete", "flow"):
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
                            "--master_addr=127.0.0.1", "--master_port=" + str(_free_port()), "-m", "vla.train.bc",
                            "--config", "configs/bc_tiny.yaml", f"data_dir={demos}", f"runs_dir={runs}", f"name={head}",
                            f"head={head}", "steps=10", "global_batch=4", "micro_batch=2", "ckpt_every=10"],
                           cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=400)
        assert r.returncode == 0, r.stderr[-2500:]
    return demos, runs


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


VLA_ARGS = ["--policy", "vla", "--model-config", "configs/model_tiny.yaml", "--splits", "train", "--tasks", "lift",
            "--episodes", "2", "--n-envs", "2"]


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_vla_cli_random_weights(tmp_path, trained, head):
    demos, _ = trained
    d = _eval(tmp_path / "v.json", *VLA_ARGS, "--head", head, "--data-dir", str(demos))
    r = d["results"]["train"]["lift"]
    assert KEYS <= set(r) and r["n"] == 2 and 0 <= r["success"] <= 1
    assert d["meta"]["policy"] == "vla" and d["meta"]["head"] == head


@pytest.mark.parametrize("head", ["discrete", "flow"])
@pytest.mark.parametrize("kind", ["final", "ckptdir"])
def test_vla_cli_with_checkpoints(tmp_path, trained, head, kind):
    demos, runs = trained
    ck = runs / head / ("final.pt" if kind == "final" else "ckpt")
    assert ck.exists()
    d = _eval(tmp_path / "v.json", *VLA_ARGS, "--head", head, "--data-dir", str(demos), "--ckpt", str(ck))
    assert d["results"]["train"]["lift"]["n"] == 2 and d["meta"]["ckpt"] == str(ck)


def test_vla_cli_ckpt_loading_changes_weights_and_is_deterministic(trained):
    from vla.infer.policy import VLAPolicy
    demos, runs = trained
    st = str(demos / "stats.json")
    rnd = VLAPolicy("configs/model_tiny.yaml", None, st, "discrete")
    ld = VLAPolicy("configs/model_tiny.yaml", str(runs / "discrete/final.pt"), st, "discrete")
    ld2 = VLAPolicy("configs/model_tiny.yaml", str(runs / "discrete/ckpt"), st, "discrete")     # DCP dir
    import torch
    assert any(not torch.equal(p, q) for p, q in zip(rnd.model.parameters(), ld.model.parameters()))
    # final.pt is bf16-exported; DCP is fp32: must agree to bf16 precision
    for (n, p), q in zip(ld.model.named_parameters(), ld2.model.parameters()):
        assert torch.allclose(p, q, atol=2e-2, rtol=2e-2), n
    img = np.random.default_rng(0).integers(0, 255, (2, 128, 128, 3), dtype=np.uint8)
    pro = np.random.default_rng(1).normal(size=(2, 11)).astype(np.float32)
    a1, a2 = ld.act(img, pro, ["lift the red cube", "lift the blue cube"]), ld.act(img, pro, ["lift the red cube", "lift the blue cube"])
    assert a1.shape == (2, 4, 4) and np.array_equal(a1, a2) and np.abs(a1).max() <= 1


def test_vla_bf16_flow(tmp_path, trained):
    demos, runs = trained
    d = _eval(tmp_path / "v.json", *VLA_ARGS, "--head", "flow", "--data-dir", str(demos), "--bf16",
              "--ckpt", str(runs / "flow/final.pt"))
    assert d["results"]["train"]["lift"]["n"] == 2


def test_vla_bf16_discrete_policy_in_process(trained):
    import torch
    from vla.infer.policy import VLAPolicy
    demos, _ = trained
    for head in ("flow", "discrete"):
        p = VLAPolicy("configs/model_tiny.yaml", None, str(demos / "stats.json"), head, "cpu", torch.bfloat16)
        img = np.zeros((1, 128, 128, 3), np.uint8); pro = np.zeros((1, 11), np.float32)
        a = p.act(img, pro, ["lift the red cube"])
        assert a.dtype == np.float32 and np.isfinite(a).all() and np.abs(a).max() <= 1


# ------------------------------------------------------------------ 4) utils.gpu with mocked nvidia-smi
def _mock_smi(tmp_path, util=5.0, apps=(), gpu_mem=100):
    d = tmp_path / "bin"; d.mkdir(exist_ok=True)
    (tmp_path / "util").write_text(str(util))
    (tmp_path / "apps").write_text("\n".join(f"{p}, GPU-0, {m}, python" for p, m in apps))
    s = d / "nvidia-smi"
    s.write_text(f"""#!/bin/bash
case "$*" in
  *query-compute-apps*) cat {tmp_path}/apps ;;
  *"query-gpu=index,utilization.gpu"*) echo "0, $(cat {tmp_path}/util)" ;;
  *query-gpu*) echo "0, GPU-0, NVIDIA A100-SXM4-40GB, {gpu_mem}, 40000, $(cat {tmp_path}/util), 40" ;;
esac
""")
    s.chmod(s.stat().st_mode | stat.S_IEXEC)
    return d


def _path(monkeypatch, bindir):
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin")


def test_snapshot_without_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    s = gpu.snapshot()
    assert s["available"] is False and "time" in s and len(s["host_load"]) == 3 and "contended" not in s


def test_quiet_wait_without_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    r = gpu.quiet_wait([0, 1], hold=0.1, timeout=1)
    assert r["quiet"] is None and r["waited_s"] == 0.0


def test_snapshot_contention_logic(monkeypatch, tmp_path):
    foreign_pid = 2_000_000_000 - 1
    _path(monkeypatch, _mock_smi(tmp_path, util=50, apps=[(foreign_pid, 1000)]))
    s = gpu.snapshot(gpus=[0])
    assert s["available"] and s["contended"] is True
    g = s["gpus"][0]
    assert g["util_pct"] == 50 and g["foreign_apps"][0]["pid"] == foreign_pid and g["own_mem_mib"] == 0
    # same util but the only app is ourselves -> not attributed as foreign
    _path(monkeypatch, _mock_smi(tmp_path, util=50, apps=[(os.getpid(), 1000)]))
    s = gpu.snapshot(gpus=[0])
    assert s["contended"] is False and s["gpus"][0]["own_mem_mib"] == 1000 and s["gpus"][0]["foreign_apps"] == []
    # low util with foreign app present -> not contended
    _path(monkeypatch, _mock_smi(tmp_path, util=5, apps=[(foreign_pid, 1000)]))
    assert gpu.snapshot(gpus=[0])["contended"] is False
    # contention on a GPU we don't use is ignored
    _path(monkeypatch, _mock_smi(tmp_path, util=90, apps=[(foreign_pid, 1000)]))
    assert gpu.snapshot(gpus=[3])["contended"] is False
    assert gpu.snapshot(foreign_util_max=95, gpus=[0])["contended"] is False


def test_snapshot_survives_smi_failure(monkeypatch, tmp_path):
    d = tmp_path / "bin"; d.mkdir()
    s = d / "nvidia-smi"; s.write_text("#!/bin/bash\necho boom >&2\nexit 9\n"); s.chmod(0o755)
    _path(monkeypatch, d)
    snap = gpu.snapshot()
    assert snap["available"] is False and "boom" in snap["error"]


def test_quiet_wait_timeout_and_quiet_window(monkeypatch, tmp_path):
    _path(monkeypatch, _mock_smi(tmp_path, util=80))
    r = gpu.quiet_wait([0], util_max=15, hold=0.3, timeout=0.6, poll=0.05, verbose=False)
    assert r["quiet"] is False and r["waited_s"] >= 0.6 and r["util_at_timeout"] == {0: 80.0}
    _path(monkeypatch, _mock_smi(tmp_path, util=3))
    r = gpu.quiet_wait([0], util_max=15, hold=0.3, timeout=5, poll=0.05)
    assert r["quiet"] is True and 0.3 <= r["waited_s"] < 5
    # a GPU id that is not reported is treated as busy
    r = gpu.quiet_wait([7], util_max=15, hold=0.1, timeout=0.3, poll=0.05, verbose=False)
    assert r["quiet"] is False


def test_quiet_wait_busy_burst_resets_hold(monkeypatch, tmp_path):
    import threading, time
    _path(monkeypatch, _mock_smi(tmp_path, util=3))
    def burst():
        time.sleep(0.4); (tmp_path / "util").write_text("90"); time.sleep(0.2); (tmp_path / "util").write_text("3")
    threading.Thread(target=burst, daemon=True).start()
    r = gpu.quiet_wait([0], util_max=15, hold=0.6, timeout=10, poll=0.05)
    assert r["quiet"] is True and r["waited_s"] >= 0.4 + 0.2 + 0.6 - 0.1      # hold restarted after the burst


def test_cli_contended_flag_from_quiet_wait_timeout(monkeypatch, tmp_path):
    """meta.contended must be True when --quiet-wait times out (mock smi busy)."""
    bindir = _mock_smi(tmp_path, util=90)
    env = _env(); env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    # shrink the hard-coded quiet_wait (hold=60,timeout=1200) via a sitecustomize-free wrapper
    code = ("import sys, vla.utils.gpu as g, vla.eval.rollout_eval as r; "
            "orig=g.quiet_wait; r.quiet_wait=lambda gp, **k: orig(gp, util_max=15, hold=0.2, timeout=0.5, poll=0.05, verbose=False); "
            f"sys.argv=['x','--policy','expert','--splits','train','--tasks','reach','--episodes','1','--n-envs','1','--quiet-wait','--out',r'{tmp_path}/q.json']; r.main()")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads((tmp_path / "q.json").read_text())
    assert d["meta"]["quiet_wait"]["quiet"] is False and d["meta"]["contended"] is True
    assert d["meta"]["gpu_snapshot_before"]["available"] is True


# ------------------------------------------------------------------ 5) SubprocVecEnv
def test_vec_env_shapes_finished_not_stepped_and_close():
    from vla.sim.vec_env import SubprocVecEnv

    vec = SubprocVecEnv(2, require_nvidia=False)
    try:
        res = vec.reset([0, 1], "lift", "train", [3, 4])
        assert len(res) == 2
        for obs, info in res:
            assert obs["image"].shape == (128, 128, 3) and obs["image"].dtype == np.uint8 and obs["proprio"].shape == (11,)
            assert "lift the" in info["instruction"] and "ee_pos" in info
        out = vec.step([0, 1], np.zeros((2, 4), np.float32))
        assert len(out) == 2 and all(len(o) == 5 and o[4]["step"] == 1 for o in out)
        out = vec.step([1], np.zeros((1, 4), np.float32))                        # env 0 "finished": not stepped
        assert out[0][4]["step"] == 2
        out = vec.step([0], np.zeros((1, 4), np.float32))
        assert out[0][4]["step"] == 2                                             # env 0 had only been stepped once
        # reset subset with a target override
        (o, i), = vec.reset([0], "lift", "train", [3], targets=[2])
        assert "blue" in i["instruction"]
        procs = list(vec.procs)
    finally:
        vec.close()
    assert all(not p.is_alive() for p in procs)


def test_vec_env_worker_error_surfaces():
    from vla.sim.vec_env import SubprocVecEnv

    vec = SubprocVecEnv(1, require_nvidia=False)
    try:
        with pytest.raises(RuntimeError, match="worker 0"):
            vec.step([0], np.zeros((1, 4), np.float32))                           # step before reset
        # worker survives the error and keeps serving
        assert vec.reset([0], "reach", "train", [1])[0][0]["image"].shape == (128, 128, 3)
        with pytest.raises(RuntimeError, match="worker 0"):
            vec.reset([0], "bogus", "train", [1])
    finally:
        vec.close()


def test_vec_env_close_is_idempotent_enough():
    from vla.sim.vec_env import SubprocVecEnv

    vec = SubprocVecEnv(1, require_nvidia=False)
    vec.close()
    assert not vec.procs[0].is_alive()
    vec.close()                                                                   # second close must not raise/hang
