"""Amendment B: prefix_order=txt_first -> prefix [txt (left-padded), img, proprio]; every path must honour it."""
from pathlib import Path

import pytest
import torch

from vla.infer.engine import FlowEngine, InferenceEngine
from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
INSTR = ["lift the red cube", "push the blue cube to the yellow zone", "reach it"]


def make(head="discrete", order="txt_first", gw=0.0, seed=0):
    torch.manual_seed(seed)
    ov = [f"head={head}", f"ground_weight={gw}"] + ([f"prefix_order={order}"] if order else [])
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ov)).eval()
    with torch.no_grad():      # informative logits (untrained action rows are near-identical)
        w = m.llm.get_input_embeddings().weight
        w[m.act_base:] = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(9)) * 0.5
        if head == "flow":
            m.dit.out.weight.normal_(0, 0.2)
    return m


def batch(m, instr=INSTR, seed=1):
    g = torch.Generator().manual_seed(seed)
    B = len(instr)
    return {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g), "proprio": torch.randn(B, 11, generator=g),
            "actions": torch.rand(B, 4, 4, generator=g) * 2 - 1, "ground_xy": torch.randn(B, 2, generator=g), "ground_mask": torch.ones(B),
            **encode_instructions(m.tokenizer, list(instr))}


# ------------------------------------------------------------------ layout
def test_default_order_is_img_first_and_layout_unchanged():
    m = make(order=None)
    assert m.cfg.prefix_order == "img_first"
    b = batch(m)
    prefix, mask = m.embed_prefix(b)
    n_img, T = 16, b["input_ids"].shape[1]
    assert prefix.shape[1] == n_img + 1 + T
    with torch.no_grad():
        assert torch.equal(prefix[:, n_img + 1:], m.llm.get_input_embeddings()(b["input_ids"]))      # [img, pro, txt]
        assert torch.equal(prefix[:, n_img], m.proprio_mlp(b["proprio"]))
    assert torch.equal(mask[:, n_img + 1:], b["attention_mask"]) and mask[:, : n_img + 1].all()


def test_txt_first_layout_matches_mask_and_positions():
    mi, mt = make(order="img_first"), make(order="txt_first")
    mt.load_state_dict(mi.state_dict())                              # same weights, different layout
    b = batch(mt)
    T, n_img = b["input_ids"].shape[1], 16
    with torch.no_grad():
        pi, ki = mi.embed_prefix(b)
        pt, kt = mt.embed_prefix(b)
        assert pt.shape == pi.shape == (3, n_img + 1 + T, pt.shape[-1])
        assert torch.equal(pt[:, :T], mt.llm.get_input_embeddings()(b["input_ids"]))                 # [txt ...
        assert torch.equal(pt[:, T:T + n_img], pi[:, :n_img])                                          # ... img ...
        assert torch.equal(pt[:, -1], pi[:, n_img])                                                    # ... proprio]
    assert torch.equal(kt[:, :T], b["attention_mask"]) and kt[:, T:].all()                       # mask permuted identically
    assert kt.sum(1).tolist() == ki.sum(1).tolist()
    # left padding stays at the sequence START (the shortest prompt is padded first)
    short = int(b["attention_mask"].sum(1).argmin())
    assert kt[short, 0] == 0 and kt[short, T - 1] == 1 and b["attention_mask"][int(b["attention_mask"].sum(1).argmax()), 0] == 1
    # position ids from the mask: padding does not advance positions, first real token is position 0
    pos = mt._pos(kt)
    assert (pos[:, -1] == kt.sum(1) - 1).all()
    for i in range(3):
        first = int(kt[i].nonzero()[0])
        assert pos[i, first] == 0 and (pos[i, first:].diff() == 1).all()


def test_text_first_changes_function_but_same_params():
    mi, mt = make(order="img_first"), make(order="txt_first")
    sd = mi.state_dict()
    mt.load_state_dict(sd, strict=True); mi.load_state_dict(mt.state_dict(), strict=True)       # checkpoints load under either order
    assert set(sd) == set(mt.state_dict())
    b = batch(mt)
    with torch.no_grad():
        assert not torch.allclose(mi(b)["loss"], mt(b)["loss"])                                 # the layout really changed


# ------------------------------------------------------------------ discrete paths agree with teacher forcing
def test_predict_matches_teacher_forced_argmax_txt_first():
    m = make(); b = batch(m)
    pred = m.predict(b)
    assert pred.shape == (3, 4, 4) and pred.abs().max() <= 1
    with torch.no_grad():
        out = m({**b, "actions": pred})
    assert out["tok_acc"].item() == 1.0, out["tok_acc"]


def test_padding_invariance_txt_first():
    """Left padding at the start must not change results: same prompt alone vs batched with a longer one."""
    m = make(); both = batch(m, INSTR[1:])
    alone = {k: (v[:1] if k not in ("input_ids", "attention_mask") else None) for k, v in both.items()}
    alone.update(encode_instructions(m.tokenizer, INSTR[1:2] if False else [INSTR[2]]))
    both = batch(m, [INSTR[2], INSTR[1]]); alone["image"], alone["proprio"] = both["image"][:1], both["proprio"][:1]
    with torch.no_grad():
        pb, pa = m.predict(both), m.predict(alone)
        lb = m.token_logprobs(both, m.act_tok.encode_bins(both["actions"].reshape(2, -1)))
        la = m.token_logprobs({**alone, "actions": both["actions"][:1]}, m.act_tok.encode_bins(both["actions"][:1].reshape(1, -1)))
    assert torch.allclose(pb[0], pa[0], atol=1e-4)
    assert torch.allclose(lb[0], la[0], atol=1e-4)


@pytest.mark.parametrize("mode", ["eager", "static"])
def test_engine_logp_and_tokens_match_teacher_forced_txt_first(mode):
    m = make(); b = batch(m)
    eng = InferenceEngine(m, 1.0, mode)
    out = eng.generate(b, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        lp = m.token_logprobs(b, out["bins"])
    assert torch.allclose(out["logp"], lp, atol=1e-4), (out["logp"] - lp).abs().max()
    g = eng.generate(b, greedy=True)
    assert torch.equal(g["actions"], m.predict(b))


def test_static_equals_eager_txt_first_with_varying_prompts():
    m = make(); st = InferenceEngine(m, 1.0, "static"); ea = InferenceEngine(m, 1.0, "eager")
    for i, ins in enumerate((INSTR, INSTR[:2], [INSTR[1]] * 3, INSTR)):
        b = batch(m, ins, seed=20 + i)
        a, s = ea.generate(b, greedy=True), st.generate(b, greedy=True)
        assert torch.equal(a["bins"], s["bins"]) and torch.allclose(a["logp"], s["logp"], atol=1e-5)
    # temperature path consistent too
    b = batch(m)
    o = InferenceEngine(m, 0.7, "static").generate(b, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.allclose(o["logp"], m.token_logprobs(b, o["bins"], 0.7), atol=1e-4)


def test_rl_bins_route_txt_first():
    m = make(); b = batch(m)
    bins = m.act_tok.encode_bins(b["actions"].reshape(3, -1))
    with torch.no_grad():
        assert torch.equal(m({**b, "bins": bins})["logp"], m.token_logprobs(b, bins))


# ------------------------------------------------------------------ flow
def test_flow_predict_and_rl_logp_txt_first():
    m = make("flow"); b = batch(m)
    with torch.no_grad():
        noise = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(0))
        a1, a2 = m.predict(b, noise), m.predict(b, noise)
        assert torch.equal(a1, a2) and a1.shape == (3, 4, 4) and torch.isfinite(a1).all()
        loss = m(b)["loss"]
    assert torch.isfinite(loss)
    r = FlowEngine(m, 3, 0.7).generate(b, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        o = m({**b, "x": r["x"], "x_next": r["x_next"], "t": r["t"], "sde_a": 0.7})
    assert torch.allclose(o["logp"], r["logp"], atol=1e-4), (o["logp"] - r["logp"]).abs().max()
    # the flow path differs from img_first with identical weights (it really sees the other layout)
    mi = make("flow", "img_first"); mi.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert not torch.allclose(mi.predict(b, noise), a1, atol=1e-4)


# ------------------------------------------------------------------ ground head reads P-1
@pytest.mark.parametrize("head", ["discrete", "flow"])
@pytest.mark.parametrize("order", ["img_first", "txt_first"])
def test_ground_head_reads_last_prefix_position(head, order):
    m = make(head, order, gw=0.2 if head == "discrete" else 0.01); b = batch(m)
    seen = {}
    def grab(mod, inp, out):                      # must return None (a returned value would replace the module output)
        seen.setdefault("x", inp[0].detach().clone())
    h = m.ground_head.register_forward_hook(grab)
    torch.manual_seed(0)
    with torch.no_grad():
        out = m(b)
        prefix, mask = m.embed_prefix(b)
        P = prefix.shape[1]
        ref = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask)).last_hidden_state[:, P - 1]
    h.remove()
    assert torch.allclose(seen["x"], ref, atol=1e-5), (seen["x"] - ref).abs().max()
    assert torch.isfinite(out["ground_loss"])
    # under txt_first position P-1 is the proprio token and it sees the whole prefix (colour word + image)
    if order == "txt_first":
        assert torch.equal(prefix[:, -1], m.proprio_mlp(b["proprio"]).to(prefix.dtype))
        b2 = dict(b, input_ids=b["input_ids"].clone()); b2["input_ids"][:, -1] = (b2["input_ids"][:, -1] + 1) % 250
        with torch.no_grad():
            p2, m2 = m.embed_prefix(b2)
            ref2 = m.llm.model(inputs_embeds=p2, attention_mask=m2, position_ids=m._pos(m2)).last_hidden_state[:, P - 1]
        assert not torch.allclose(ref, ref2, atol=1e-6)               # the read-out depends on the text
