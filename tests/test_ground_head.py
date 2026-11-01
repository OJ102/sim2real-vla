"""Grounding aux head (plan step 4): head presence, losses, grads, masking, strict old-ckpt loading."""
from pathlib import Path

import pytest
import torch

from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def make(head, gw, seed=0):
    torch.manual_seed(seed)
    return VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), [f"head={head}", f"ground_weight={gw}"]))


def batch(m, B=4, mask=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    b = {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g),
         "proprio": torch.randn(B, 11, generator=g), "actions": torch.rand(B, 4, 4, generator=g) * 2 - 1,
         "ground_xy": torch.randn(B, 2, generator=g), "ground_mask": torch.ones(B) if mask is None else torch.tensor(mask, dtype=torch.float),
         **encode_instructions(m.tokenizer, ["lift the red cube", "push the blue cube", "reach it", "lift the green cube"][:B])}
    return b


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_ground_outputs_and_head_shape(head):
    m = make(head, 0.2 if head == "discrete" else 0.01)
    assert getattr(m, "ground_head", None) is not None
    b = batch(m)
    out = m(b)
    for k in ("loss", "main_loss", "ground_loss", "ground_err_cm"):
        assert k in out and torch.isfinite(out[k]), k
    assert out["ground_loss"] > 0 and out["ground_err_cm"] > 0
    gw = m.cfg.ground_weight
    assert out["loss"].item() == pytest.approx(out["main_loss"].item() + gw * out["ground_loss"].item(), rel=1e-5)
    # head maps the d_llm feature at P-1 to [B, 2]
    d = m.llm.config.hidden_size
    assert m.ground_head(torch.zeros(5, d)).shape == (5, 2)
    assert m.ground_head[0].out_features == 256 and m.ground_head[-1].out_features == 2


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_ground_head_gets_nonzero_grads_and_aux_reaches_backbone(head):
    m = make(head, 0.2 if head == "discrete" else 0.01)
    m(batch(m))["loss"].backward()
    gg = [p.grad for p in m.ground_head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gg) and any(g.abs().sum() > 0 for g in gg)
    # aux loss alone must send gradient into the backbone (that is the point: force colour x image composition)
    m.zero_grad()
    out = m(batch(m))
    out["ground_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("llm.model"))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("vision."))


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_ground_weight_zero_means_no_head_and_old_ckpt_loads_strictly(head):
    m0 = make(head, 0.0)
    assert getattr(m0, "ground_head", None) is None
    assert not any("ground" in n for n in m0.state_dict())
    out = m0(batch(m0))
    assert "loss" in out and torch.isfinite(out["loss"])
    assert out.get("ground_loss") is None or out["ground_loss"].item() == 0
    old_sd = {k: v.clone() for k, v in m0.state_dict().items()}            # a checkpoint from before the feature
    m1 = make(head, 0.0, seed=5)
    m1.load_state_dict(old_sd, strict=True)
    # a grounded model has extra keys only for the head; the shared part loads from the old ckpt
    mg = make(head, 0.2)
    missing, unexpected = mg.load_state_dict(old_sd, strict=False)
    assert not unexpected and all("ground_head" in k for k in missing) and missing
    with pytest.raises(RuntimeError):
        mg.load_state_dict(old_sd, strict=True)


def test_ground_weight_zero_loss_equals_plain_main_loss():
    a, b_ = make("discrete", 0.0), make("discrete", 0.2)
    b_.load_state_dict(a.state_dict(), strict=False)
    ba = batch(a)
    la, lb = a(ba)["loss"], b_(ba)
    assert la.item() == pytest.approx(lb["main_loss"].item(), rel=1e-6)           # same backbone: main loss identical


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_ground_mask_semantics(head):
    m = make(head, 0.2 if head == "discrete" else 0.01)
    torch.manual_seed(0)
    # all mask 0 -> ground_loss == 0, no NaN, total == main
    b0 = batch(m, mask=[0, 0, 0, 0])
    torch.manual_seed(3); o0 = m(b0)
    assert o0["ground_loss"].item() == 0 and torch.isfinite(o0["loss"]) and torch.isfinite(o0["ground_err_cm"])
    assert o0["loss"].item() == pytest.approx(o0["main_loss"].item(), rel=1e-6)
    # mixed: only labelled rows contribute -> changing unlabelled targets changes nothing
    m.eval()
    bm = batch(m, mask=[1, 0, 1, 0])
    with torch.no_grad():
        l1 = m(bm)["ground_loss"]
        bm2 = dict(bm, ground_xy=bm["ground_xy"].clone()); bm2["ground_xy"][[1, 3]] += 100.0
        l2 = m(bm2)["ground_loss"]
        bm3 = dict(bm, ground_xy=bm["ground_xy"].clone()); bm3["ground_xy"][[0, 2]] += 1.0
        l3 = m(bm3)["ground_loss"]
    assert l1.item() == pytest.approx(l2.item(), rel=1e-6) and l1.item() != pytest.approx(l3.item(), rel=1e-3)
    # normalised by mask.sum(): equals the mean over labelled rows only
    full_a = batch(m, mask=[1, 1, 1, 1])
    with torch.no_grad():
        per = [m({k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in full_a.items()})["ground_loss"].item() for i in range(4)]
        sel = m({k: v for k, v in bm.items()})["ground_loss"].item()
    assert sel == pytest.approx((per[0] + per[2]) / 2, rel=1e-3)
    # masked-out rows get no gradient signal from the aux loss
    m.train(); m.zero_grad()
    m(batch(m, mask=[0, 0, 0, 0]))["ground_loss"]
