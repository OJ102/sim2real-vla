"""P3 tests (CPU, tiny random-init model, no downloads): losses, overfit, tokenizer, sliced head, sampler, padding."""
import numpy as np
import pytest
import torch

from vla.model.action_tokenizer import ActionTokenizer
from vla.model.flow_head import flow_loss
from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, ActionLMHead, VLAConfig
from vla.utils.config import load_config

CFG = "configs/model_tiny.yaml"
INSTR = ["lift the red cube", "push the blue cube to the yellow zone"]


def make(head, seed=0):
    torch.manual_seed(seed)
    cfg = load_config(VLAConfig, CFG, [f"head={head}"])
    return VLA(cfg)


def make_batch(m, instr=INSTR, seed=1):
    g = torch.Generator().manual_seed(seed)
    b = len(instr)
    return {"image": torch.randint(0, 256, (b, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(b, 11, generator=g),
            "actions": torch.rand(b, 4, 4, generator=g) * 2 - 1,
            **encode_instructions(m.tokenizer, instr)}


@pytest.fixture(params=["discrete", "flow"])
def head(request):
    return request.param


# ---------------------------------------------------------------- (1) loss + grads
def test_config_head_and_dit(head):
    m = make(head)
    assert m.cfg.head == head and (m.dit is not None) == (head == "flow")
    assert m.cfg.tiny


def test_loss_finite_and_grads(head):
    m = make(head)
    out = m(make_batch(m))
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    missing = []
    for n, p in m.named_parameters():
        if p.grad is None:
            missing.append(n)
        else:
            assert torch.isfinite(p.grad).all(), n
    # trainable pieces that must receive a (non-None) gradient
    must = ["projector", "proprio_mlp", "llm.model.layers", "vision.encoder.layers"]
    if head == "flow":
        must += ["dit"]
    for key in must:
        named = [(n, p) for n, p in m.named_parameters() if key in n]
        assert named and all(p.grad is not None for _, p in named), (key, missing[:10])
    if head == "flow":
        # DiT out layer is zero-init -> backbone grads are exactly 0 at step 0; the DiT itself must learn
        assert any(p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("dit"))
        torch.optim.SGD(m.parameters(), lr=0.1).step()
        m.zero_grad(); torch.manual_seed(0); m(make_batch(m))["loss"].backward()
        for key in ("projector", "llm.model.layers", "vision.encoder.layers"):
            assert any(p.grad.abs().sum() > 0 for n, p in m.named_parameters() if key in n), key
    else:
        for key in must:
            assert any(p.grad.abs().sum() > 0 for n, p in m.named_parameters() if key in n), key
    # only the unused head side may lack grads: flow mode never touches act_head / tied embeddings rows
    assert not [n for n in missing if head == "discrete" and "dit" in n]
    if head == "discrete":
        assert m.llm.get_input_embeddings().weight.grad is not None  # tied embedding gets action-row grads
        assert "tok_acc" in out


# ---------------------------------------------------------------- (2) overfit one batch
def _train(m, batch, steps, lr, loss_fn):
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    first = last = None
    for i in range(steps):
        loss = loss_fn()
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    return first, last


def test_overfit_discrete():
    m = make("discrete"); batch = make_batch(m)
    first, last = _train(m, batch, 150, 3e-3, lambda: m(batch)["loss"])
    assert last < 0.1 * first, (first, last)
    m.eval()
    pred = m.predict(batch)
    assert (pred - batch["actions"]).abs().max() <= 1 / 255 + 1e-6  # memorised -> exact bins


def test_overfit_flow_fixed_t_noise():
    m = make("flow"); batch = make_batch(m)
    g = torch.Generator().manual_seed(5)
    t = torch.rand(2, generator=g) * 0.9 + 0.05
    eps = torch.randn(2, 4, 4, generator=g)

    def loss_fn():
        prefix, mask = m.embed_prefix(batch)
        h = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask)).last_hidden_state
        return flow_loss(m.dit, batch["actions"], batch["proprio"], h, mask, t=t, noise=eps)

    first, last = _train(m, batch, 300, 3e-3, loss_fn)
    assert last < 0.1 * first, (first, last)


def test_overfit_flow_forward_with_seeded_noise():
    """Public forward() path: seed every step so t/noise are fixed."""
    m = make("flow"); batch = make_batch(m)

    def loss_fn():
        torch.manual_seed(123)
        return m(batch)["loss"]

    first, last = _train(m, batch, 300, 3e-3, loss_fn)
    assert last < 0.1 * first, (first, last)


# ---------------------------------------------------------------- (3) tokenizer
def test_action_tokenizer_roundtrip():
    tk = ActionTokenizer(256, base_id=1000)
    a = np.random.default_rng(0).uniform(-1, 1, 5000)
    ids = tk.encode(a)
    assert ids.min() >= 1000 and ids.max() <= 1255
    assert np.abs(tk.decode(ids) - a).max() <= 1 / 255 + 1e-9
    at = torch.tensor(a)
    assert (tk.decode(tk.encode(at)) - at).abs().max() <= 1 / 255 + 1e-9
    e = tk.encode(np.array([-1.0, 1.0, 0.0, -5.0, 5.0]))
    assert list(e[:2]) == [1000, 1255] and e[3] == 1000 and e[4] == 1255  # clipped
    assert tk.decode(np.array([1000, 1255])).tolist() == [-1.0, 1.0]


# ---------------------------------------------------------------- (4) sliced head
def test_sliced_head_equals_full_head_gather():
    m = make("discrete")
    h = torch.randn(3, 5, m.llm.config.hidden_size)
    full = m.llm.lm_head(h)[..., m.act_base:m.act_base + 256]
    assert torch.allclose(m.act_head(h), full, atol=1e-5)
    cp = ActionLMHead(m.llm, m.act_base, copy=True)
    assert torch.allclose(cp(h), full, atol=1e-5)
    assert sum(p.numel() for p in cp.parameters()) + sum(b.numel() for b in cp.buffers()) == 256 * m.llm.config.hidden_size
    assert m.llm.lm_head.weight.shape[0] >= m.act_base + 256
    assert m.llm.lm_head.weight.data_ptr() == m.llm.get_input_embeddings().weight.data_ptr()  # tied
    assert cp.w.data_ptr() != m.llm.lm_head.weight.data_ptr()  # copy is independent storage


def test_sliced_head_argmax_matches_full_argmax():
    m = make("discrete")
    h = torch.randn(4, m.llm.config.hidden_size)
    assert (m.act_head(h).argmax(-1) == m.llm.lm_head(h)[..., m.act_base:m.act_base + 256].argmax(-1)).all()


# ---------------------------------------------------------------- (5) flow sampler determinism
def test_flow_sampler_deterministic_with_fixed_noise():
    m = make("flow").eval(); batch = make_batch(m)
    noise = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(9))
    a, b = m.predict(batch, noise), m.predict(batch, noise)
    assert torch.equal(a, b) and a.shape == (2, 4, 4)
    assert a.abs().max() <= 1
    assert not torch.equal(a, m.predict(batch, noise + 1))  # sampler really depends on noise
    n0 = noise.clone(); m.predict(batch, noise); assert torch.equal(n0, noise)  # input noise not mutated


# ---------------------------------------------------------------- (6) KV-cache decode consistency
@pytest.mark.parametrize("trained", [False, True])
def test_discrete_greedy_matches_teacher_forced_argmax(trained):
    m = make("discrete"); batch = make_batch(m)
    if trained:
        _train(m, batch, 40, 3e-3, lambda: m(batch)["loss"])
    m.eval()
    pred = m.predict(batch)
    assert pred.shape == (2, 4, 4) and pred.abs().max() <= 1
    fb = dict(batch, actions=pred)  # feed its own tokens, teacher-forced
    with torch.no_grad():
        out = m(fb)
    assert out["tok_acc"].item() == 1.0, out["tok_acc"]


# ---------------------------------------------------------------- (7) left padding
def _pad_more(batch, k):
    ids, mask = batch["input_ids"], batch["attention_mask"]
    b = ids.shape[0]
    return dict(batch, input_ids=torch.cat([torch.zeros(b, k, dtype=ids.dtype), ids], 1),
                attention_mask=torch.cat([torch.zeros(b, k, dtype=mask.dtype), mask], 1))


def test_left_padding_invariance_loss(head):
    m = make(head).eval(); one = make_batch(m, INSTR[:1])
    padded = _pad_more(one, 9)
    assert padded["input_ids"].shape[1] == one["input_ids"].shape[1] + 9
    torch.manual_seed(0); l1 = m(one)["loss"]
    torch.manual_seed(0); l2 = m(padded)["loss"]
    assert torch.allclose(l1, l2, atol=1e-4), (l1.item(), l2.item())


def test_left_padding_invariance_predict_batched_with_longer(head):
    m = make(head).eval()
    both = make_batch(m, INSTR)                       # shorter one is left-padded by the tokenizer
    assert both["attention_mask"][0].sum() < both["attention_mask"][1].sum() and both["attention_mask"][0, 0] == 0
    alone = {k: (v[:1] if k not in ("input_ids", "attention_mask") else None) for k, v in both.items()}
    alone.update(encode_instructions(m.tokenizer, INSTR[:1]))
    noise = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(2))
    pb = m.predict(both, noise if head == "flow" else None)
    pa = m.predict(alone, noise[:1] if head == "flow" else None)
    assert torch.allclose(pb[0], pa[0], atol=1e-4), (pb[0] - pa[0]).abs().max()
