"""P6b Flow-GRPO tests: SDE sampler math on an exact-velocity toy, model-level log-probs, RL loop, DiT-only sync."""
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vla.model.flow_head import (euler_sample, flow_sde_sample, gaussian_kl, gaussian_logprob, sde_grid, sde_sigma,
                                 sde_step_stats)

ROOT = Path(__file__).resolve().parents[1]
M0, S0 = 0.4, 0.15


class ToyDiT:
    """Exact velocity field of x_t = (1-t) x0 + t eps with x0 ~ N(M0, S0^2), eps ~ N(0,1) (1-D, convention t=1 noise)."""
    chunk, act_dim = 1, 1

    def __init__(self):
        self.t_seen = []

    def __call__(self, x, t, proprio, ctx, ctx_mask):
        tt = float(t[0])
        self.t_seen.append(tt)
        var = (1 - tt) ** 2 * S0**2 + tt**2
        c = x - (1 - tt) * M0
        ex0 = M0 + (1 - tt) * S0**2 / var * c
        eeps = tt / var * c
        return eeps - ex0


def run_toy(K, a, n=200_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pro = torch.zeros(n, dtype=torch.float64)
    dit = ToyDiT()
    out = flow_sde_sample(dit, pro, None, None, K=K, a=a, generator=g)
    return dit, out, pro


# ------------------------------------------------------------------ toy exactness
def test_a0_sde_equals_ode_euler_on_same_grid():
    K = 20
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(5000, 1, 1, generator=g, dtype=torch.float64)
    pro = torch.zeros(5000, dtype=torch.float64)
    ode = euler_sample(ToyDiT(), pro, None, None, noise=noise, ts=sde_grid(K))
    sde = flow_sde_sample(ToyDiT(), pro, None, None, K=K, a=0.0, noise=noise)
    assert (ode - sde["actions"]).abs().max() < 1e-9
    assert torch.equal(sde["logp"], torch.zeros_like(sde["logp"]))      # deterministic: logp is a placeholder 0
    assert all(s == 0 for s in sde["std"])


@pytest.mark.parametrize("K,tol", [(200, 0.02), (50, 0.05)])
def test_sde_sampler_recovers_target_marginal(K, tol):
    _, out, _ = run_toy(K, 0.7)
    x = out["x_next"][:, -1, 0, 0]
    assert abs(x.mean().item() - M0) < 0.02 * S0 * (1 if K >= 200 else 3)
    assert abs(x.std().item() / S0 - 1) < tol, x.std().item()


def test_sde_logp_recompute_equals_returned():
    dit, out, pro = run_toy(30, 0.7, n=2000, seed=3)
    for i, (t, std) in enumerate(zip(out["t"], out["std"])):
        h = t - (out["t"][i + 1] if i + 1 < len(out["t"]) else 0.0)
        v = ToyDiT()(out["x"][:, i], torch.full((2000,), t, dtype=torch.float64), pro, None, None)
        mean, s = sde_step_stats(out["x"][:, i], v, t, h, 0.7)
        assert s == pytest.approx(std)
        lp = gaussian_logprob(out["x_next"][:, i], mean, s)
        assert torch.allclose(lp, out["logp"][:, i], atol=1e-9), i
    # chain: x_{i+1} of step i is x_i of step i+1
    assert torch.equal(out["x_next"][:, :-1], out["x"][:, 1:])
    # independent closed form of the Gaussian density
    mean = torch.zeros(4, 1, 1, dtype=torch.float64); x = torch.tensor([0.0, 0.3, -0.5, 1.0], dtype=torch.float64).view(4, 1, 1)
    ref = torch.distributions.Normal(0.0, 0.2).log_prob(x).sum((-2, -1))
    assert torch.allclose(gaussian_logprob(x, mean, 0.2), ref, atol=1e-9)


def test_gaussian_kl_closed_form():
    m = torch.randn(6, 2, 3, dtype=torch.float64)
    assert gaussian_kl(m, m.clone(), 0.3).abs().max() == 0
    q = m + torch.randn_like(m) * 0.1
    kl = gaussian_kl(m, q, 0.3)
    assert (kl > 0).all()
    assert torch.allclose(kl, ((m - q) ** 2).sum((-2, -1)) / (2 * 0.3**2))
    # matches torch's analytic Gaussian KL
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(m, 0.3), torch.distributions.Normal(q, 0.3)).sum((-2, -1))
    assert torch.allclose(kl, ref)


@pytest.mark.parametrize("K", [3, 5, 10, 50])
def test_grid_never_touches_endpoints(K):
    ts = sde_grid(K)
    assert len(ts) == K + 1 and ts[0] == pytest.approx(1 - 1 / (2 * K)) and ts[-1] == 0.0
    assert all(a > b for a, b in zip(ts, ts[1:])) and all(0 < t < 1 for t in ts[:-1])
    dit, out, _ = run_toy(K, 0.7, n=100)
    assert 0 < min(dit.t_seen) and max(dit.t_seen) < 1 and len(dit.t_seen) == K       # t=0 and t=1 never evaluated
    assert all(math.isfinite(s) and s > 0 for s in out["std"])
    assert torch.isfinite(out["x_next"]).all() and torch.isfinite(out["logp"]).all()
    assert sde_sigma(0.5, 0.7) == pytest.approx(0.7)
    with pytest.raises(ZeroDivisionError):
        sde_sigma(1.0, 0.7)                                                                 # why the grid must start below 1


def test_sampler_seed_reproducible():
    _, a, _ = run_toy(10, 0.7, n=64, seed=5); _, b, _ = run_toy(10, 0.7, n=64, seed=5); _, c, _ = run_toy(10, 0.7, n=64, seed=6)
    assert torch.equal(a["x_next"], b["x_next"]) and torch.equal(a["logp"], b["logp"])
    assert not torch.equal(a["x_next"], c["x_next"])


# ------------------------------------------------------------------ model level (tiny flow head)
def flow_model(seed=0):
    from vla.model.vla import VLA, VLAConfig
    from vla.utils.config import load_config

    torch.manual_seed(seed)
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ["head=flow"])).eval()
    with torch.no_grad():                                      # zero-init output layer would make everything trivial
        m.dit.out.weight.normal_(0, 0.2)
        m.dit.out.bias.normal_(0, 0.05)
    return m


def fbatch(m, B=3, seed=1):
    from vla.model.prompts import encode_instructions

    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(B, 11, generator=g),
            **encode_instructions(m.tokenizer, ["lift the red cube", "push the blue cube", "reach it"][:B])}


def rl_batch(m, out, a, **kw):
    b = fbatch(m)
    b.update({"x": out["x"], "x_next": out["x_next"], "t": out["t"], "sde_a": a, **kw})
    return b


def test_flow_rl_logp_matches_sampler_behaviour_logp():
    from vla.infer.engine import FlowEngine

    m = flow_model(); b = fbatch(m)
    out = FlowEngine(m, K=4, sde_a=0.7).generate(b, generator=torch.Generator().manual_seed(0))
    assert out["x"].shape == (3, 4, 4, 4) and out["logp"].shape == (3, 4) and len(out["t"]) == 4
    with torch.no_grad():
        r = m.flow_rl_logp(rl_batch(m, out, 0.7))
    assert torch.allclose(r["logp"], out["logp"], atol=1e-4), (r["logp"] - out["logp"]).abs().max()
    assert r["mean"].shape == out["x"].shape and torch.isfinite(r["logp"]).all()
    # forward() routes the 'x_next' batch to the same function
    with torch.no_grad():
        f = m(rl_batch(m, out, 0.7))
    assert torch.equal(f["logp"], r["logp"])
    # after a DiT weight change the recomputed logp moves (it really depends on the current policy)
    with torch.no_grad():
        m.dit.out.weight.add_(0.3 * torch.randn_like(m.dit.out.weight))
        r2 = m.flow_rl_logp(rl_batch(m, out, 0.7))
    assert not torch.allclose(r2["logp"], out["logp"], atol=1e-3)


def test_flow_rl_gradients_only_through_dit_unless_train_backbone():
    from vla.infer.engine import FlowEngine

    m = flow_model(); m.train()
    out = FlowEngine(m, K=3, sde_a=0.7).generate(fbatch(m), generator=torch.Generator().manual_seed(1))
    m.zero_grad()
    m.flow_rl_logp(rl_batch(m, out, 0.7))["logp"].sum().backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    assert all(g[n] is not None and torch.isfinite(g[n]).all() for n in g if n.startswith("dit."))
    assert any(g[n].abs().sum() > 0 for n in g if n.startswith("dit."))
    assert all(g[n] is None for n in g if not n.startswith("dit.")), [n for n in g if not n.startswith("dit.") and g[n] is not None][:3]
    m.zero_grad()
    m.flow_rl_logp(rl_batch(m, out, 0.7, train_backbone=True))["logp"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("llm.model"))


def test_flow_engine_greedy_is_deterministic_ode():
    from vla.infer.engine import FlowEngine, make_engine

    m = flow_model(); b = fbatch(m)
    e = make_engine(m, flow_K=3, sde_a=0.7)
    assert isinstance(e, FlowEngine)
    a1 = e.generate(b, greedy=True)["actions"]
    assert a1.shape == (3, 4, 4) and a1.abs().max() <= 1
    # greedy = ODE from *random* noise (not seeded): same noise -> same actions through the sampler used by greedy
    from vla.model.flow_head import euler_sample
    noise = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        prefix, mask = m.embed_prefix(b)
        h = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask)).last_hidden_state
        o1 = euler_sample(m.dit, b["proprio"], h, mask, m.cfg.flow.steps, noise=noise)
        o2 = euler_sample(m.dit, b["proprio"], h, mask, m.cfg.flow.steps, noise=noise)
    assert torch.equal(o1, o2)
    s1 = e.generate(b, generator=torch.Generator().manual_seed(1)); s2 = e.generate(b, generator=torch.Generator().manual_seed(1))
    assert torch.equal(s1["x_next"], s2["x_next"])


def test_grpo_loss_kl_override_replaces_k3():
    from vla.rl.algo import grpo_loss

    g = torch.Generator().manual_seed(0)
    old = torch.randn(4, 3, generator=g); new = (old + 0.1 * torch.randn(4, 3, generator=g)).requires_grad_()
    ref = old + 0.5; adv = torch.tensor([1.0, -1.0, 0.5, 0.0])
    l_k3, m_k3 = grpo_loss(new, old, old, ref, adv)
    kl = torch.full((4, 3), 0.25)
    l_ov, m_ov = grpo_loss(new, old, old, ref, adv, kl=kl)
    assert m_ov["kl"] == pytest.approx(0.25) and m_k3["kl"] != pytest.approx(0.25)
    assert not torch.isclose(l_k3, l_ov)
    l_none, m_none = grpo_loss(new, old, old, ref, adv, beta=0.0, kl=kl)
    l_none2, _ = grpo_loss(new, old, old, ref, adv, beta=0.0)
    assert torch.isclose(l_none, l_none2)                              # with beta=0 the KL term is irrelevant
    klv = (torch.randn(4, 3, generator=g) ** 2).requires_grad_()
    grpo_loss(new, old, old, ref, adv, kl=klv)[0].backward()
    assert klv.grad is not None and klv.grad.abs().sum() > 0           # gradient flows through the supplied KL


# ------------------------------------------------------------------ DiT-only weight sync (2-proc gloo)
def _sync_worker(rank, world, init_file, prefix):
    import torch.distributed as dist

    from vla.model.vla import VLA, VLAConfig
    from vla.rl.roles import init_sync_group, make_roles
    from vla.rl.weight_sync import sync_weights
    from vla.utils.config import load_config

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        roles = make_roles(1, 1); pg = init_sync_group(roles)
        torch.manual_seed(100 + rank)
        m = VLA(load_config(VLAConfig, "configs/model_tiny.yaml", ["head=flow"]))
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        info = sync_weights(m, pg, roles.sync_src, 5, prefix=prefix)
        dit_names = [n for n in before if n.startswith("dit.")]
        flat = lambda sel: torch.cat([dict(m.named_parameters())[n].detach().reshape(-1) for n in sel])   # noqa: E731
        both = [torch.zeros_like(flat(dit_names)) for _ in range(2)]
        dist.all_gather(both, flat(dit_names))
        assert torch.equal(both[0], both[1]), "dit params must be synced"
        other = [n for n in before if not n.startswith("dit.")]
        if prefix == "dit." and rank == 1:
            assert all(torch.equal(dict(m.named_parameters())[n], before[n]) for n in other), "non-dit must be untouched"
            assert info["nbytes"] == sum(before[n].numel() for n in dit_names) * 4
        if prefix is None and rank == 1:
            assert any(not torch.equal(dict(m.named_parameters())[n], before[n]) for n in other)
        assert info["version"] == 5
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("prefix", ["dit.", None])
def test_sync_prefix_moves_only_matching_params(tmp_path, prefix):
    import torch.multiprocessing as mp
    mp.spawn(_sync_worker, args=(2, str(tmp_path / "rdzv"), prefix), nprocs=2, join=True)


# ------------------------------------------------------------------ RL loop
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}


def grpo(tmp, name, *extra, timeout=900):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                        "--master_addr=127.0.0.1", f"--master_port={port}", "-m", "vla.rl.grpo", "--config",
                        "configs/rl_flow_tiny.yaml", f"runs_dir={tmp / 'runs'}", f"shm_dir={tmp / 'shm'}", f"name={name}",
                        *extra], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    run = tmp / "runs" / name
    return r, run, [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]


def test_flow_rl_loop_s0(tmp_path):
    r, run, rec = grpo(tmp_path, "flow0", "iterations=3", "lr=3e-3")
    assert [x["iter"] for x in rec] == [0, 1, 2] and (run / "final.pt").is_file()
    for x in rec:
        assert all(np.isfinite(x[k]) for k in ("loss", "kl", "return_mean", "grad_norm"))
        assert x["kl"] >= -1e-9
        assert x["engine_mismatch_equal_version"] < 1e-4, x["engine_mismatch_equal_version"]
        assert x["tis_w_mean"] == pytest.approx(1.0, abs=1e-3)
        assert x["staleness_hist"] == {"0": x["n_samples"]}
    # learner trained only the DiT (backbone frozen): non-dit weights equal init, dit weights moved
    from vla.model.vla import VLA, VLAConfig
    from vla.utils.config import load_config

    sd = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(0)
    init = {k: v.detach().to(torch.bfloat16) for k, v in VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"),
                                                                           ["head=flow"])).state_dict().items()}
    dit_moved = [k for k, v in sd.items() if k.startswith("dit.") and v.is_floating_point() and not torch.equal(v, init[k])]
    other_moved = [k for k, v in sd.items() if not k.startswith("dit.") and v.is_floating_point() and not torch.equal(v, init[k])]
    assert dit_moved and not other_moved, (dit_moved[:2], other_moved[:3])


def test_flow_rl_loop_s1_staleness(tmp_path):
    _, run, rec = grpo(tmp_path, "flow1", "iterations=3", "staleness=1")
    n = rec[0]["n_samples"]
    assert rec[0]["staleness_hist"] == {"0": n}
    assert all(x["staleness_hist"] == {"1": n} for x in rec[1:]), [x["staleness_hist"] for x in rec]
    assert all(np.isfinite(x["loss"]) and x["kl"] >= -1e-9 and x["tis_w_max"] <= 2.0 + 1e-6 for x in rec)


# ------------------------------------------------------------------ (A) ckpt stall study smoke
def test_ckpt_stall_study_smoke(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    out, shm, nfs = tmp_path / "o.json", tmp_path / "shm", tmp_path / "nfs"
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                        "--master_addr=127.0.0.1", f"--master_port={port}", "scripts/ckpt_stall_study.py",
                        "--model-config", "configs/model_tiny.yaml", "--micro-batch", "4", "--shm-dir", str(shm),
                        "--nfs-dir", str(nfs), "--out", str(out)], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2500:]
    d = json.loads(out.read_text())
    legs = d["legs"]
    assert set(legs) == {"sync_shm", "async_shm", "sync_nfs", "async_nfs"}
    assert legs["sync_shm"]["saves"] == legs["async_shm"]["saves"] == 5
    assert legs["sync_nfs"]["saves"] == legs["async_nfs"]["saves"] == 2
    for leg in legs.values():
        assert leg["stall_s_min"] <= leg["stall_s_median"] <= leg["stall_s_max"] and leg["publish_s_median"] > 0
        assert all(rec["failed"] == 0 and rec["stall_s"] >= 0 for rec in leg["records"])
    for loc in ("shm", "nfs"):
        assert d[f"async_stall_over_sync_{loc}"] > 0 and math.isfinite(d[f"async_stall_over_sync_{loc}"])
    assert {"step_s_idle_median", "contended", "gpu_snapshot_before", "gpu_snapshot_after", "world"} <= set(d) and d["world"] == 2
    assert not (nfs / "sync").exists() and not (nfs / "async").exists()          # NFS legs deleted afterwards
