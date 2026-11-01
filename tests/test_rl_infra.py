"""P6 skeleton tests (CPU, tiny model, gloo): weight sync, GRPO/TIS math, inference engine."""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vla.infer.engine import InferenceEngine
from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.rl.algo import group_advantages, grpo_loss, kl_k3, staleness_hist, tis_weight
from vla.rl.roles import init_sync_group, make_roles
from vla.rl.weight_sync import plan_buckets, sync_weights
from vla.utils.config import load_config


def tiny(seed, head="discrete"):
    torch.manual_seed(seed)
    return VLA(load_config(VLAConfig, "configs/model_tiny.yaml", [f"head={head}"]))


def flat(m):
    return torch.cat([p.detach().reshape(-1).float() for p in m.parameters()])


# =============================================================== A) weight sync, 2-proc gloo
def _worker(rank, world, init_file, mode):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        roles = make_roles(1, 1)
        assert roles.role_of(rank) == ("learner" if rank == 0 else "rollout")
        pg = init_sync_group(roles)
        m = tiny(100 + rank)                                  # DIFFERENT seeds
        ptrs = [p.data_ptr() for p in m.parameters()]
        mine = flat(m)
        both = [torch.zeros_like(mine) for _ in range(2)]
        dist.all_gather(both, mine)
        assert not torch.equal(both[0], both[1]), "models should start different"

        sd = torch.bfloat16 if mode == "bf16" else None
        info = sync_weights(m, pg, roles.sync_src, version=7, bucket_bytes=64 << 10, sync_dtype=sd)
        assert info["version"] == 7, info
        assert info["n_buckets"] > 1 and info["latency_ms"] >= 0 and info["nbytes"] > 0, info
        assert [p.data_ptr() for p in m.parameters()] == ptrs, "param storage must not change"

        mine = flat(m)
        dist.all_gather(both, mine)
        if sd is None:
            assert torch.equal(both[0], both[1]), "rollout params not bitwise equal"
        else:
            assert torch.equal(both[1], both[0].to(torch.bfloat16).float()), "bf16 sync mismatch"
            assert not torch.equal(both[0], both[1])  # lossy, as expected

        # second sync after learner params change
        if rank == 0:
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.5)
        info2 = sync_weights(m, pg, roles.sync_src, version=8, bucket_bytes=64 << 10, sync_dtype=sd)
        assert info2["version"] == 8
        mine = flat(m)
        dist.all_gather(both, mine)
        ref = both[0] if sd is None else both[0].to(torch.bfloat16).float()
        assert torch.equal(both[1], ref)
        assert [p.data_ptr() for p in m.parameters()] == ptrs
        assert not torch.equal(both[1], both[1] * 0)  # sanity
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_weight_sync_2proc_gloo(tmp_path, mode):
    mp.spawn(_worker, args=(2, str(tmp_path / "rdzv"), mode), nprocs=2, join=True)


def test_roles():
    r = make_roles(2, 2)
    assert r.learner_ranks == (0, 1) and r.rollout_ranks == (2, 3) and r.world_size == 4
    assert r.sync_src == 0 and r.sync_ranks == (0, 2, 3)
    r13 = make_roles(1, 3)
    assert r13.sync_ranks == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        make_roles(0, 1)
    with pytest.raises(ValueError):
        make_roles(1, 0)
    with pytest.raises(ValueError):
        r.role_of(9)


def test_plan_buckets_partition_and_size():
    m = tiny(0)
    items = [(n, p, p.dtype) for n, p in m.named_parameters()]
    for bb in (1 << 10, 64 << 10, 256 << 20):
        plan = plan_buckets(items, bb)
        assert [i for b in plan for i in b] == list(range(len(items)))   # exact ordered partition
        for b in plan:
            nb = sum(items[i][1].numel() * 4 for i in b)
            assert len(b) == 1 or nb <= bb
    assert len(plan_buckets(items, 256 << 20)) == 1
    # mixed dtype never shares a bucket
    mixed = [("a", torch.zeros(4), torch.float32), ("b", torch.zeros(4), torch.bfloat16)]
    assert len(plan_buckets(mixed, 1 << 20)) == 2


# =============================================================== B) algo math
def test_tis_weight():
    lp = torch.randn(4, 16)
    assert torch.equal(tis_weight(lp, lp.clone()), torch.ones(4, 16))
    w = tis_weight(torch.tensor([5.0, 0.0, -1.0]), torch.tensor([0.0, 0.0, 0.0]))
    assert w.tolist() == pytest.approx([2.0, 1.0, float(torch.exp(torch.tensor(-1.0)))])
    x = torch.randn(5, requires_grad=True)
    assert not tis_weight(x, torch.zeros(5)).requires_grad                    # detached
    assert tis_weight(torch.tensor([3.0]), torch.tensor([0.0]), cap=1.5).item() == 1.5


def test_group_advantages():
    g = torch.Generator().manual_seed(0)
    r = torch.rand(6, 8, generator=g)
    a = group_advantages(r)
    assert a.shape == r.shape
    assert a.mean(1).abs().max() < 1e-5
    assert (a.std(1, unbiased=False) - 1).abs().max() < 1e-3
    r[2] = 0.7                                         # all-equal group
    a = group_advantages(r)
    assert torch.isfinite(a).all() and a[2].abs().max() == 0
    assert group_advantages(torch.zeros(2, 4)).abs().max() == 0


def test_kl_k3():
    a, b = torch.randn(1000), torch.randn(1000)
    assert (kl_k3(a, b) >= 0).all()
    assert kl_k3(a, a.clone()).abs().max() == 0


def _lp(b=3, t=16, seed=0):
    return torch.randn(b, t, generator=torch.Generator().manual_seed(seed)) - 3


def test_grpo_loss_identity_metrics():
    lp = _lp().requires_grad_()
    loss, m = grpo_loss(lp, lp.detach().clone(), lp.detach().clone(), lp.detach().clone(), torch.tensor([1.0, -1.0, 0.5]))
    assert m["ratio_clipfrac"] == 0 and m["tis_w_mean"] == pytest.approx(1) and m["tis_w_max"] == pytest.approx(1)
    assert m["kl"] == 0 and m["engine_mismatch"] == 0 and m["tis_clipfrac"] == 0
    loss.backward()
    assert torch.isfinite(lp.grad).all() and lp.grad.abs().sum() > 0


def test_grpo_clip_zero_grad_outside_trust_region():
    old = _lp(1, 4)
    for adv_sign, shift in ((+1.0, 0.5), (-1.0, -0.5)):      # adv>0 & ratio>1+eps ; adv<0 & ratio<1-eps
        new = (old + shift).clone().requires_grad_()
        loss, m = grpo_loss(new, old, old, old.clone(), torch.tensor([adv_sign]), beta=0.0)
        loss.backward()
        assert new.grad.abs().max() == 0, (adv_sign, new.grad)
        assert m["ratio_clipfrac"] == 1.0
    # inside region: gradient flows and has the right sign (ascend on adv>0 => loss grad negative)
    new = (old + 0.05).clone().requires_grad_()
    grpo_loss(new, old, old, old.clone(), torch.tensor([1.0]), beta=0.0)[0].backward()
    assert (new.grad < 0).all()
    # opposite sides are NOT clipped (pessimistic min): adv>0 & ratio<1-eps still gets gradient
    new = (old - 0.5).clone().requires_grad_()
    grpo_loss(new, old, old, old.clone(), torch.tensor([1.0]), beta=0.0)[0].backward()
    assert new.grad.abs().min() > 0


def test_grpo_tis_weights_gradient_and_detached():
    old = _lp(1, 4)
    beh = old - 5.0                                           # pi_old/mu = e^5 -> capped at 2
    new = old.clone().requires_grad_()
    loss, m = grpo_loss(new, old, beh, old.clone(), torch.tensor([1.0]), beta=0.0)
    assert m["tis_w_mean"] == pytest.approx(2.0) and m["tis_w_max"] == pytest.approx(2.0) and m["tis_clipfrac"] == 1.0
    assert m["engine_mismatch"] == pytest.approx(5.0)
    loss.backward()
    g_tis = new.grad.clone()
    new2 = old.clone().requires_grad_()
    grpo_loss(new2, old, beh, old.clone(), torch.tensor([1.0]), beta=0.0, use_tis=False)[0].backward()
    assert torch.allclose(g_tis, 2 * new2.grad, atol=1e-6)    # exactly the weight scales the gradient


def test_grpo_mask_excludes_padding():
    new = _lp(2, 6).requires_grad_(); old = new.detach().clone()
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.float)
    ref = old - 100.0 * (1 - mask)                            # absurd KL only on padded positions
    adv = torch.tensor([1.0, -1.0])
    loss, m = grpo_loss(new, old, old, ref, adv, mask=mask, beta=0.5)
    ref2 = old.clone(); ref2[1, 3:] += 7                      # change padded positions only
    loss2, _ = grpo_loss(new, old, old, ref2, adv, mask=mask, beta=0.5)
    assert loss.item() == pytest.approx(loss2.item(), abs=1e-6)
    # padded positions get zero gradient
    loss.backward()
    assert new.grad[1, 3:].abs().max() == 0 and new.grad[1, :3].abs().max() > 0
    assert torch.isfinite(torch.tensor(m["loss"]))
    # engine_mismatch only counts valid tokens
    beh = old.clone(); beh[1, 3:] += 9
    _, m2 = grpo_loss(new.detach(), old, beh, old, adv, mask=mask)
    assert m2["engine_mismatch"] == 0 and m2["tis_w_max"] == pytest.approx(1)


def test_grpo_kl_term_positive_when_policy_moves():
    old = _lp(2, 4)
    _, m = grpo_loss(old + 0.3, old, old, old, torch.zeros(2), beta=0.02)
    assert m["kl"] > 0


def test_staleness_hist():
    assert staleness_hist(torch.tensor([5, 5, 4, 3, 5]), 5) == {0: 3, 1: 1, 2: 1}
    assert staleness_hist(torch.tensor([2]), 2) == {0: 1}


# =============================================================== C) inference engine
def _batch(m, instr=("lift the red cube", "push the blue cube to the yellow zone"), seed=1):
    g = torch.Generator().manual_seed(seed)
    b = len(instr)
    return {"image": torch.randint(0, 256, (b, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(b, 11, generator=g), **encode_instructions(m.tokenizer, list(instr))}


def test_engine_logp_matches_teacher_forced():
    m = tiny(0).eval(); batch = _batch(m)
    out = InferenceEngine(m).generate(batch, generator=torch.Generator().manual_seed(0))
    assert out["bins"].shape == (2, 16) and out["logp"].shape == (2, 16) and out["actions"].shape == (2, 4, 4)
    assert (out["bins"] >= 0).all() and (out["bins"] < 256).all() and (out["logp"] <= 0).all()
    with torch.no_grad():
        lp = m.token_logprobs(batch, out["bins"])
    assert torch.allclose(out["logp"], lp, atol=1e-4), (out["logp"] - lp).abs().max()
    assert torch.allclose(out["actions"], m.act_tok.decode_bins(out["bins"].float()).reshape(2, 4, 4))


def test_engine_greedy_matches_predict_with_untied_logits():
    m = tiny(0).eval(); batch = _batch(m)
    with torch.no_grad():
        m.llm.get_input_embeddings().weight[m.act_base:] *= 30
    out = InferenceEngine(m).generate(batch, greedy=True)
    assert torch.equal(out["actions"], m.predict(batch))
    with torch.no_grad():
        lp = m.token_logprobs(batch, out["bins"])
    assert torch.allclose(out["logp"], lp, atol=1e-4)


def test_engine_seeded_reproducible_and_stochastic():
    m = tiny(0).eval(); batch = _batch(m); eng = InferenceEngine(m)
    a = eng.generate(batch, generator=torch.Generator().manual_seed(3))
    b = eng.generate(batch, generator=torch.Generator().manual_seed(3))
    c = eng.generate(batch, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a["bins"], b["bins"]) and torch.equal(a["logp"], b["logp"])
    assert not torch.equal(a["bins"], c["bins"])


def test_engine_sees_updated_weights_in_place():
    """After an in-place weight copy the engine must sample from the new policy (no stale cached head)."""
    m, m2 = tiny(0).eval(), tiny(1).eval(); batch = _batch(m)
    with torch.no_grad():  # untrained logits are ~uniform (logp = -ln 256 everywhere): make them informative
        for mm in (m, m2):
            mm.llm.get_input_embeddings().weight[mm.act_base:] *= 30
    eng = InferenceEngine(m)
    before = eng.generate(batch, greedy=True)["logp"].clone()
    with torch.no_grad():
        for p, q in zip(m.parameters(), m2.parameters()):
            p.copy_(q)
    after = eng.generate(batch, greedy=True)
    ref = InferenceEngine(m2).generate(batch, greedy=True)
    assert torch.allclose(after["logp"], ref["logp"], atol=1e-5) and not torch.allclose(after["logp"], before)


def test_engine_rejects_flow_head():
    with pytest.raises(AssertionError):
        InferenceEngine(tiny(0, "flow"))


def test_engine_logp_matches_token_logprobs_at_temperature():
    """Behaviour log-probs are of softmax(logits/T): learner's token_logprobs must use the same T."""
    m = tiny(0).eval(); batch = _batch(m)
    with torch.no_grad():  # new action rows start near-identical (mean init): randomise so logits are informative
        w = m.llm.get_input_embeddings().weight
        w[m.act_base:] = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(5)) * 0.5
    out = InferenceEngine(m, temperature=0.7).generate(batch, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        lp07 = m.token_logprobs(batch, out["bins"], 0.7)
        lp1 = m.token_logprobs(batch, out["bins"])
        lp1b = m.token_logprobs(batch, out["bins"], temperature=1.0)
    assert torch.allclose(out["logp"], lp07, atol=1e-4), (out["logp"] - lp07).abs().max()
    assert not torch.allclose(out["logp"], lp1, atol=1e-3), (out["logp"] - lp1).abs().max()   # T=1 would be wrong
    assert torch.equal(lp1, lp1b) and (lp07 <= 0).all()
    # sharper than T=1 on the sampled tokens' average, and still a valid distribution
    assert lp07.mean() > lp1.mean()
    # forward() route with bins + temperature agrees
    fb = dict(batch, bins=out["bins"], temperature=0.7)
    with torch.no_grad():
        assert torch.allclose(m(fb)["logp"], lp07, atol=1e-5)
