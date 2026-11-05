"""rollout_eval --temperature (commit 90ea0fb): sampled discrete-head eval next to greedy. CPU, tiny trained fixture."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TINY = "configs/model_tiny.yaml"


def _env():
    e = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}
    e.pop("CUDA_VISIBLE_DEVICES", None)
    return e


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def _run(*args, out=None, timeout=420):
    cmd = [sys.executable, "-m", "vla.eval.rollout_eval", *args] + (["--out", str(out)] if out else [])
    return subprocess.run(cmd, cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=timeout)


def _eval(out, *args):
    r = _run(*args, out=out)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return json.loads(Path(out).read_text()), r


def _strip(d):
    """results without wall-clock fields."""
    return {s: {t: {k: v for k, v in r.items() if k != "eval_secs"} for t, r in ts.items()} for s, ts in d["results"].items()}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("vla_temp")
    demos, runs = root / "demos", root / "runs"
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={demos}", "episodes_per_combo=2", "chunk_size=2", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
                        "--master_addr=127.0.0.1", "--master_port=" + str(_free_port()), "-m", "vla.train.bc",
                        "--config", "configs/bc_tiny.yaml", f"data_dir={demos}", f"runs_dir={runs}", "name=discrete",
                        "head=discrete", "steps=10", "global_batch=4", "micro_batch=2", "ckpt_every=10"],
                       cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=400)
    assert r.returncode == 0, r.stderr[-2500:]
    return demos, runs / "discrete" / "final.pt"


def _args(trained, *extra, seed0="0"):
    demos, ck = trained
    return ["--policy", "vla", "--model-config", TINY, "--head", "discrete", "--splits", "train", "--tasks", "lift",
            "--episodes", "2", "--n-envs", "2", "--data-dir", str(demos), "--ckpt", str(ck), "--seed0", seed0, *extra]


@pytest.fixture(scope="module")
def runs(trained, tmp_path_factory):
    """CLI runs shared by the comparison tests (each costs a subprocess + rollouts)."""
    d = tmp_path_factory.mktemp("temp_runs")
    out = {}
    for name, extra, seed0 in [("greedy", [], "0"), ("t1a", ["--temperature", "1"], "0"),
                               ("t1b", ["--temperature", "1"], "0"), ("t1_seed", ["--temperature", "1"], "77"),
                               ("t05", ["--temperature", "0.5"], "0")]:
        out[name] = _eval(d / f"{name}.json", *_args(trained, *extra, seed0=seed0))[0]
    return out


# ---------------------------------------------------------------- 1) no flag = greedy path
def test_no_flag_is_greedy_and_meta_temperature_none(trained, runs):
    from vla.eval.rollout_eval import evaluate
    from vla.infer.policy import VLAPolicy
    from vla.sim.vec_env import SubprocVecEnv
    demos, ck = trained
    g = runs["greedy"]
    assert "temperature" in g["meta"] and g["meta"]["temperature"] is None
    pol = VLAPolicy(TINY, str(ck), str(demos / "stats.json"), "discrete")
    assert pol.engine is None                                            # untouched model.predict path
    vec = SubprocVecEnv(2, require_nvidia=False)
    try:
        res = evaluate(pol, vec, ["train"], ["lift"], 2, 4, {}, seed0=0, log=lambda *a, **k: None)
    finally:
        vec.close()
    ref = {k: v for k, v in res["train"]["lift"].items() if k not in ("eval_secs", "success_vs_ceiling", "expert_ceiling")}
    cli = {k: v for k, v in _strip(g)["train"]["lift"].items() if k not in ("success_vs_ceiling", "expert_ceiling")}
    assert json.loads(json.dumps(ref)) == cli


# ---------------------------------------------------------------- 2-4) reproducibility and sensitivity
def test_t1_twice_identical(runs):
    assert runs["t1a"]["meta"]["temperature"] == 1.0
    assert _strip(runs["t1a"]) == _strip(runs["t1b"])


def test_t1_differs_from_greedy(runs):
    assert _strip(runs["t1a"]) != _strip(runs["greedy"])


def test_seed0_and_temperature_change_results(runs):
    assert _strip(runs["t1_seed"]) != _strip(runs["t1a"])
    assert runs["t05"]["meta"]["temperature"] == 0.5
    assert _strip(runs["t05"]) != _strip(runs["t1a"])


def test_sample_seed0_alone_changes_rollouts(trained):
    """Same env seeds, different generator seed -> different episodes (isolates the sampler from env seeding)."""
    from vla.eval.rollout_eval import run_vla
    from vla.infer.policy import VLAPolicy
    from vla.sim.vec_env import SubprocVecEnv
    demos, ck = trained
    pol = VLAPolicy(TINY, str(ck), str(demos / "stats.json"), "discrete", temperature=1.0)
    vec = SubprocVecEnv(2, require_nvidia=False)
    try:
        go = lambda s: run_vla(vec, pol, "lift", "train", [0, 1], 4, max_steps=12, sample_seed0=s)  # noqa: E731
        a, b, c = go(0), go(0), go(1)
    finally:
        vec.close()
    key = lambda eps: [(e["steps"], e["action_2nd_diff"]) for e in eps]  # noqa: E731
    assert key(a) == key(b) and key(a) != key(c)


# ---------------------------------------------------------------- 5) invalid T
@pytest.mark.parametrize("t", ["0", "-1", "nan", "inf"])
def test_invalid_temperature_exit_2(t):
    r = _run("--policy", "expert", "--splits", "train", "--tasks", "lift", "--episodes", "1", "--n-envs", "1",
             f"--temperature={t}", timeout=120)
    assert r.returncode == 2, (r.returncode, r.stderr[-800:])
    assert "--temperature must be > 0" in r.stderr


# ---------------------------------------------------------------- 6) ignored for flow / expert
def test_flow_head_with_temperature_warns_greedy(tmp_path, trained):
    demos, _ = trained
    d, r = _eval(tmp_path / "f.json", "--policy", "vla", "--model-config", TINY, "--head", "flow", "--splits", "train",
                 "--tasks", "lift", "--episodes", "2", "--n-envs", "2", "--data-dir", str(demos), "--temperature", "1")
    assert "--temperature ignored" in r.stderr
    assert d["meta"]["temperature"] is None and d["results"]["train"]["lift"]["n"] == 2


def test_flow_from_model_config_with_temperature_warns_greedy(tmp_path, trained):
    """head not on CLI, flow comes from the model config -> second warning branch."""
    demos, _ = trained
    cfg = tmp_path / "flow_tiny.yaml"
    cfg.write_text((ROOT / TINY).read_text().replace("head: discrete", "head: flow"))
    d, r = _eval(tmp_path / "f.json", "--policy", "vla", "--model-config", str(cfg), "--splits", "train",
                 "--tasks", "lift", "--episodes", "1", "--n-envs", "1", "--data-dir", str(demos), "--temperature", "1")
    assert "--temperature ignored" in r.stderr
    assert d["meta"]["temperature"] is None


def test_expert_with_temperature_warns_greedy(tmp_path):
    d, r = _eval(tmp_path / "e.json", "--policy", "expert", "--splits", "train", "--tasks", "lift", "--episodes", "2",
                 "--n-envs", "2", "--temperature", "1")
    assert "--temperature ignored" in r.stderr
    assert d["meta"]["temperature"] is None and d["results"]["train"]["lift"]["n"] == 2


# ---------------------------------------------------------------- 7) sample_seed
def test_sample_seed_stable_and_distinct():
    from vla.eval.rollout_eval import sample_seed
    s = sample_seed(0, "train", "lift", 0)
    assert s == sample_seed(0, "train", "lift", 0) and 0 <= s < 2**63
    # stable across processes (no PYTHONHASHSEED dependence)
    code = "from vla.eval.rollout_eval import sample_seed; print(sample_seed(0, 'train', 'lift', 0))"
    for hs in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**_env(), "PYTHONHASHSEED": hs},
                             capture_output=True, text=True, timeout=120)
        assert int(out.stdout.strip()) == s, out.stderr[-500:]
    others = [sample_seed(1, "train", "lift", 0), sample_seed(0, "heldout", "lift", 0),
              sample_seed(0, "train", "push", 0), sample_seed(0, "train", "lift", 1)]
    assert len({s, *others}) == 5


# ---------------------------------------------------------------- 8) in process VLAPolicy
def test_policy_temperature_generator_reproducible(trained):
    from vla.infer.policy import VLAPolicy
    demos, ck = trained
    p = VLAPolicy(TINY, str(ck), str(demos / "stats.json"), "discrete", temperature=1.0)
    assert p.engine is not None
    img = np.random.default_rng(0).integers(0, 255, (2, 128, 128, 3), dtype=np.uint8)
    pro = np.random.default_rng(1).normal(size=(2, 11)).astype(np.float32)
    ins = ["lift the red cube", "lift the blue cube"]
    a1 = p.act(img, pro, ins, generator=p.make_generator(123))
    a2 = p.act(img, pro, ins, generator=p.make_generator(123))
    a3 = p.act(img, pro, ins, generator=p.make_generator(124))
    assert a1.shape == (2, 4, 4) and a1.dtype == np.float32
    assert np.array_equal(a1, a2) and np.abs(a1).max() <= 1
    assert not np.array_equal(a1, a3)
    greedy = VLAPolicy(TINY, str(ck), str(demos / "stats.json"), "discrete").act(img, pro, ins)
    assert not np.array_equal(a1, greedy)


def test_policy_temperature_invalid_raises():
    from vla.infer.policy import VLAPolicy
    for t in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            VLAPolicy(TINY, None, None, "discrete", temperature=t)
    assert VLAPolicy(TINY, None, None, "flow", temperature=1.0).engine is None   # flow: ignored, greedy
