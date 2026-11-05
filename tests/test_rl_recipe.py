"""RL recipe amendment (docs/PLAN_v2_rl.md): episode-normalised loss, std floor, beta controller, entropy, new metrics."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.rl.algo import adapt_beta, group_advantages, grpo_loss
from vla.rl.learner import load_batches
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}


# ------------------------------------------------------------------ (1) episode-normalised loss
def toy(B=4, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(B, T, generator=g, dtype=torch.float64) - 3
    return old, old.clone()


def test_episode_norm_equal_weight_and_zero_loss():
    old, new = toy()
    sw = torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)       # ep A: 1 chunk, ep B: 3 chunks
    adv = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
    new = new.clone().requires_grad_()
    loss, _ = grpo_loss(new, old, old, old, adv, beta=0.0, sample_w=sw, w_total=sw.sum())
    assert abs(loss.item()) < 1e-12                                          # +1 and -1 episodes cancel
    loss.backward()
    per = new.grad.abs().sum(1)
    assert per[0].item() == pytest.approx(0.5) and per[1:].sum().item() == pytest.approx(0.5)   # equal total per episode
    # token norm (default) for comparison: 0.25 / 0.75 (length-biased)
    new2 = old.clone().requires_grad_()
    grpo_loss(new2, old, old, old, adv, beta=0.0)[0].backward()
    p2 = new2.grad.abs().sum(1)
    assert p2[0].item() == pytest.approx(0.25) and p2[1:].sum().item() == pytest.approx(0.75)


def hand(new, old, beh, ref, adv, sw, wt, beta, eps=0.2, cap=2.0):
    r = np.exp(new - old); w = np.minimum(np.exp(old - beh), cap)
    advb = adv[:, None]
    pg = np.minimum(r * advb, np.clip(r, 1 - eps, 1 + eps) * advb)
    d = ref - new; kl = np.exp(d) - d - 1
    return float((((-pg * w + beta * kl).mean(1)) * sw).sum() / wt)


def test_episode_norm_matches_hand_computation_including_kl_weights():
    g = torch.Generator().manual_seed(1)
    B, T = 5, 8
    old = torch.randn(B, T, generator=g, dtype=torch.float64) - 3
    new = old + 0.3 * torch.randn(B, T, generator=g, dtype=torch.float64)
    beh = old + 0.2 * torch.randn(B, T, generator=g, dtype=torch.float64)
    ref = old + torch.randn(B, T, generator=g, dtype=torch.float64) * torch.tensor([0.1, 1, 2, 0.5, 3], dtype=torch.float64)[:, None]
    adv = torch.tensor([1.0, -0.5, 0.2, -1.0, 0.7], dtype=torch.float64)
    sw = torch.tensor([1.0, 0.5, 0.25, 1 / 3, 0.125], dtype=torch.float64)
    loss, m = grpo_loss(new, old, beh, ref, adv, beta=0.3, sample_w=sw, w_total=sw.sum())
    want = hand(*(x.numpy() for x in (new, old, beh, ref, adv, sw)), float(sw.sum()), 0.3)
    assert loss.item() == pytest.approx(want, rel=1e-9)
    # the KL term really carries the sample weights: with beta only (adv=0) loss == weighted mean of per-sample mean KL
    l2, _ = grpo_loss(new, old, beh, ref, torch.zeros(B, dtype=torch.float64), beta=1.0, sample_w=sw, w_total=sw.sum())
    d = ref - new; per = (torch.exp(d) - d - 1).mean(1)
    assert l2.item() == pytest.approx(((per * sw).sum() / sw.sum()).item(), rel=1e-9)
    l3, _ = grpo_loss(new, old, beh, ref, torch.zeros(B, dtype=torch.float64), beta=1.0)
    assert l3.item() != pytest.approx(l2.item(), rel=1e-3)                    # unweighted token mean differs


def test_micro_batching_invariance_with_shared_w_total():
    g = torch.Generator().manual_seed(2)
    B, T = 6, 5
    old = torch.randn(B, T, generator=g, dtype=torch.float64) - 3
    new0 = old + 0.2 * torch.randn(B, T, generator=g, dtype=torch.float64)
    beh = old + 0.1 * torch.randn(B, T, generator=g, dtype=torch.float64)
    ref = old + 0.4 * torch.randn(B, T, generator=g, dtype=torch.float64)
    adv = torch.randn(B, generator=g, dtype=torch.float64)
    sw = torch.tensor([1, 1, 0.5, 0.5, 0.2, 0.2], dtype=torch.float64)
    a = new0.clone().requires_grad_()
    full, _ = grpo_loss(a, old, beh, ref, adv, beta=0.1, sample_w=sw, w_total=sw.sum()); full.backward()
    b = new0.clone().requires_grad_()
    tot = 0.0
    for sl in (slice(0, 2), slice(2, 5), slice(5, 6)):                       # uneven micro-batches
        l, _ = grpo_loss(b[sl], old[sl], beh[sl], ref[sl], adv[sl], beta=0.1, sample_w=sw[sl], w_total=sw.sum())
        l.backward(); tot += l.item()                                         # NO rescale by micro-batch size
    assert tot == pytest.approx(full.item(), rel=1e-10)
    assert torch.allclose(a.grad, b.grad, atol=1e-12)


def test_token_path_unchanged_when_sample_w_none():
    old, new = toy(); new = new + 0.1
    adv = torch.tensor([1.0, -1, 0.5, 0], dtype=torch.float64)
    l, m = grpo_loss(new, old, old, old + 0.2, adv)
    r = np.exp((new - old).numpy()); advb = adv.numpy()[:, None]
    pg = np.minimum(r * advb, np.clip(r, .8, 1.2) * advb); d = (old + 0.2 - new).numpy(); kl = np.exp(d) - d - 1
    assert l.item() == pytest.approx((-pg + 0.02 * kl).mean(), rel=1e-9) and m["kl"] == pytest.approx(kl.mean())


# ------------------------------------------------------------------ (2) std floor
def test_std_floor():
    r = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [10.0, 0.0, 10.0, 0.0], [5.0, 5.0, 5.0, 5.001]])
    a0 = group_advantages(r)
    af = group_advantages(r, std_floor=1.0)
    assert torch.isfinite(af).all() and af[0].abs().max() == 0                               # all-equal -> exactly 0
    assert a0[0].abs().max() == 0                                                              # (mean-subtraction already zero)
    assert a0[3].abs().max() > 0.5 and af[3].abs().max() < 1e-2                               # tiny differences: unit scale vs ~0
    assert af[1].abs().max() == pytest.approx(0.5)                                             # std 0.5 < floor 1 -> divided by floor
    assert torch.allclose(af[2], a0[2], atol=1e-5)                                             # std 5 > floor: unchanged
    assert torch.equal(group_advantages(r, std_floor=0.0), a0)                                 # 0 = off, identical


# ------------------------------------------------------------------ (3) beta controller
def test_adapt_beta_rule_and_clamps():
    assert adapt_beta(0.1, 0.2, 0.1) == pytest.approx(0.15)                    # kl > 1.5*target -> *1.5
    assert adapt_beta(0.1, 0.05, 0.1) == pytest.approx(0.1 / 1.5)              # kl < target/1.5 -> /1.5
    assert adapt_beta(0.1, 0.1, 0.1) == 0.1 and adapt_beta(0.1, 0.149, 0.1) == 0.1 and adapt_beta(0.1, 0.0667, 0.1) == 0.1
    assert adapt_beta(0.1, 0.1500001, 0.1) == pytest.approx(0.15) and adapt_beta(0.1, 0.0666, 0.1) == pytest.approx(0.1 / 1.5)
    assert adapt_beta(0.9, 5.0, 0.1) == 1.0 and adapt_beta(1.0, 5.0, 0.1) == 1.0            # upper clamp
    assert adapt_beta(0.006, 0.0, 0.1) == 0.005 and adapt_beta(0.005, 0.0, 0.1) == 0.005    # lower clamp
    assert adapt_beta(0.5, 3.0, 1.0, lo=0.1, hi=0.6) == 0.6
    b = 0.05
    for _ in range(3):
        b = adapt_beta(b, 0.0, 0.05)
    assert b == pytest.approx(0.05 / 1.5**3)


# ------------------------------------------------------------------ (4) entropy
def tiny(seed=0):
    torch.manual_seed(seed)
    return VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), [])).eval()


def tb(m, B=3):
    g = torch.Generator().manual_seed(1)
    return {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g), "proprio": torch.randn(B, 11, generator=g),
            **encode_instructions(m.tokenizer, ["lift the red cube", "push the blue cube", "reach it"][:B])}


def test_forward_bins_returns_logp_and_entropy():
    m = tiny(); b = tb(m)
    bins = torch.randint(0, 256, (3, 16), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        o = m({**b, "bins": bins, "temperature": 1.0})
        lp = m.token_logprobs(b, bins)
    assert set(o) >= {"logp", "entropy"} and o["entropy"].shape == (3, 16) and not o["entropy"].requires_grad
    assert torch.equal(o["logp"], lp)
    assert o["entropy"].max() <= np.log(256) + 1e-4 and (o["entropy"] > 0).all()
    assert o["entropy"].mean().item() == pytest.approx(np.log(256), abs=0.05)                # untrained tiny: ~uniform logits
    # peaked logits -> clearly lower entropy
    with torch.no_grad():
        w = m.llm.get_input_embeddings().weight
        w[m.act_base:] = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(5)) * 3
        o2 = m({**b, "bins": bins, "temperature": 1.0})
        o3 = m({**b, "bins": bins, "temperature": 0.3})
    assert o2["entropy"].mean() < o["entropy"].mean() - 0.5 and o3["entropy"].mean() < o2["entropy"].mean()   # temperature sharpens
    # grad mode: entropy stays detached while logp is differentiable
    m.train()
    o4 = m({**b, "bins": bins, "temperature": 1.0})
    assert o4["logp"].requires_grad and not o4["entropy"].requires_grad


# ------------------------------------------------------------------ (5) load_batches
def write_batch(path, returns, success, G, version, lens, n_tok=16):
    E = len(returns)
    ep = np.repeat(np.arange(E), lens)
    N = len(ep)
    np.savez(path, image=np.zeros((N, 8, 8, 3), np.uint8), proprio=np.zeros((N, 11), np.float32), bins=np.zeros((N, n_tok), np.int64),
             logp=np.zeros((N, n_tok), np.float32), ep=ep, ep_return=np.array(returns, np.float32),
             ep_success=np.array(success, np.float32), instr=np.array([f"i{e}" for e in range(E)]), version=np.int64(version),
             comp_success=np.zeros(E, np.float32), meta=np.array(json.dumps({"env_steps": 1, "gen_time": 1.0, "idle_s": 0.0})))
    return ep


def test_load_batches_ep_id_n_chunks_uniform_and_floor(tmp_path):
    G = 4
    l0, l1 = [1, 3, 2, 5, 4, 1, 2, 2], [3, 3, 1, 1, 2, 6, 1, 1]
    r0 = [1, 1, 1, 1, 0, 1, 0, 2]; s0 = [1, 1, 1, 1, 0, 0, 1, 0]      # group0 all-success, group1 mixed
    r1 = [0, 0, 0, 0, 5, 5, 5, 5]; s1 = [0, 0, 0, 0, 1, 1, 1, 1]      # group0 all-fail, group1 all-success
    e0, e1 = write_batch(tmp_path / "a.npz", r0, s0, G, 1, l0), write_batch(tmp_path / "b.npz", r1, s1, G, 2, l1)
    data, es, metas = load_batches([tmp_path / "a.npz", tmp_path / "b.npz"], G, 1.0)
    n0 = len(e0)
    assert len(data["ep_id"]) == len(data["n_chunks"]) == len(data["adv"]) == n0 + len(e1)
    ids = data["ep_id"]
    assert len(set(ids[:n0]) & set(ids[n0:])) == 0 and len(set(ids)) == 16                      # unique across batches
    assert np.array_equal(ids[:n0], e0) and np.array_equal(ids[n0:], e1 + 8)
    for e in range(16):
        assert set(data["n_chunks"][ids == e]) == {(l0 + l1)[e]}                                # chunks of that sample's episode
    assert [list(u) for u in es["uniform"]] == [[True, False], [True, True]]
    assert all(u.dtype == bool for u in es["uniform"])
    # std floor 1.0: group with returns [1,1,1,1] -> 0; group [1,0,1,2]... adv = (R-mean)/max(std,1)
    for e in range(4):
        assert np.all(data["adv"][ids == e] == 0)
    R = np.array(r0[4:], np.float32); want = (R - R.mean()) / max(R.std(), 1.0)
    for k, e in enumerate(range(4, 8)):
        assert data["adv"][ids == e][0] == pytest.approx(want[k], abs=1e-6)
    d0, *_ = load_batches([tmp_path / "a.npz"], G)                                               # default floor = 0 -> unchanged API
    assert "ep_id" in d0 and np.isfinite(d0["adv"]).all()


# ------------------------------------------------------------------ (6)-(8) end to end + configs
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def grpo(tmp, name, *extra, script=None, timeout=800):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={port()}", *([str(script)] if script else ["-m", "vla.rl.grpo"]), "--config", "configs/rl_tiny.yaml",
           f"runs_dir={tmp / 'runs'}", f"shm_dir={tmp / 'shm'}", f"name={name}", "iterations=3", "staleness=1", *extra]
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2500:]
    return [json.loads(l) for l in (tmp / "runs" / name / "metrics.jsonl").read_text().splitlines()]


RECIPE = ["opt_steps_per_iter=4", "loss_norm=episode", "adv_std_floor=1.0", "kl_target=0.05", "beta_adapt=true", "beta=0.05"]


def test_recipe_end_to_end_metrics_and_beta_controller(tmp_path):
    rec = grpo(tmp_path, "recipe", *RECIPE)
    assert len(rec) == 3
    for x in rec:
        for k in ("entropy", "frac_uniform_groups", "neg_weight_frac", "beta", "loss", "kl", "grad_norm"):
            assert k in x and np.isfinite(x[k]), k
        assert 0 <= x["frac_uniform_groups"] <= 1 and 0 <= x["neg_weight_frac"] <= 1
        assert 0 < x["entropy"] <= np.log(256) + 1e-3
        assert x["staleness_hist"] and max(int(k) for k in x["staleness_hist"]) <= 1
    betas = [x["beta"] for x in rec]
    assert betas[0] == pytest.approx(0.05) and betas[1] == pytest.approx(betas[0] / 1.5) and betas[2] == pytest.approx(betas[1] / 1.5)
    # token path: beta stays fixed without beta_adapt
    base = grpo(tmp_path, "nofix", "beta=0.05")
    assert all(x["beta"] == 0.05 for x in base)


COUNT = '''
import atexit, os, runpy, torch
n = [0]
_orig = torch.optim.AdamW.step
def step(self, *a, **k):
    n[0] += 1
    return _orig(self, *a, **k)
torch.optim.AdamW.step = step
if __name__ == "__main__":  # spawn workers re-import this file as __mp_main__; without the guard they re-run grpo and hang
    atexit.register(lambda: open(os.environ["STEP_COUNT_FILE"] + "." + os.environ.get("RANK", "0"), "w").write(str(n[0])))
    runpy.run_module("vla.rl.grpo", run_name="__main__")
'''


@pytest.mark.parametrize("k", [1, 4])
def test_opt_steps_per_iter_exact(tmp_path, k, monkeypatch):
    script = tmp_path / "count.py"; script.write_text(COUNT)
    monkeypatch.setitem(ENV, "STEP_COUNT_FILE", str(tmp_path / "steps"))
    try:
        grpo(tmp_path, f"k{k}", f"opt_steps_per_iter={k}", script=script)
    finally:
        ENV.pop("STEP_COUNT_FILE", None)
    assert int((tmp_path / "steps.0").read_text()) == 3 * k                                  # learner = rank 0: iterations x k


def test_defaults_are_deterministic_and_token_norm(tmp_path):
    a, b = grpo(tmp_path, "d1"), grpo(tmp_path, "d2")
    # rollouts are bitwise reproducible; learner-side floats drift ~2e-6 rel from multi-process CPU reduction order
    # (bitwise run-to-run determinism of training is not a requirement)
    # measured over 4x2 runs: loss max abs 1.5e-9 (|loss|~1e-4 is a cancellation of O(1) terms, so noise is absolute,
    # ~1.2e-5 rel), grad_norm 1.2e-8 rel, tis_w_mean 3e-9 rel, engine_mismatch 3e-10 abs; kl/clipfrac/return exact
    exact = ("kl", "ratio_clipfrac", "return_mean")
    close = {"loss": dict(rel=1e-5, abs=1e-8), "grad_norm": dict(rel=1e-5), "tis_w_mean": dict(rel=1e-6),
             "engine_mismatch": dict(rel=1e-5, abs=1e-7)}
    for x, y in zip(a, b):
        assert all(x[k] == y[k] for k in exact), [(k, x[k], y[k]) for k in exact if x[k] != y[k]]
        for k, tol in close.items():
            assert x[k] == pytest.approx(y[k], **tol), (k, x[k], y[k])
    assert all(x["beta"] == 0.02 and x["entropy"] > 0 for x in a)                              # default beta unchanged


def test_pilot_configs_ladder():
    from vla.rl.grpo import RLConfig

    c = {n: load_config(RLConfig, str(ROOT / f"configs/rl_pilot_{n}.yaml")) for n in "abc"}
    for n, x in c.items():
        assert x.iterations == 35 and x.engine_mode == "graph" and x.lr == 2e-6 and x.opt_steps_per_iter == 4, n
        assert x.seeds_per_iter * x.group_size == 64 and x.temperature == 1.0
    a, b, cc = c["a"], c["b"], c["c"]
    assert (a.loss_norm, a.adv_std_floor, a.beta_adapt) == ("token", 0.0, False)
    assert (b.loss_norm, b.adv_std_floor, b.beta_adapt) == ("episode", 1.0, False)
    assert (cc.loss_norm, cc.adv_std_floor, cc.beta_adapt, cc.kl_target, cc.beta) == ("episode", 1.0, True, 0.05, 0.05)
    assert b.beta == a.beta                                                                    # B only adds loss_norm + floor
