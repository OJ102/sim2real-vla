"""Regression: flow head with bf16 weights and fp32 inputs (crashed bc_flow on the server: fp32 x_t into bf16 DiT)."""
from pathlib import Path

import pytest
import torch

from vla.infer.engine import FlowEngine
from vla.model.flow_head import euler_sample
from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def m():
    torch.manual_seed(0)
    mm = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ["head=flow"])).to(torch.bfloat16)
    with torch.no_grad():
        mm.dit.out.weight.normal_(0, 0.2)
    return mm


def batch(m, B=3):
    g = torch.Generator().manual_seed(1)
    return {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(B, 11, generator=g, dtype=torch.float32),
            "actions": torch.rand(B, 4, 4, generator=g, dtype=torch.float32) * 2 - 1,
            **encode_instructions(m.tokenizer, ["lift the red cube", "push the blue cube", "reach it"][:B])}


def test_loss_float32_finite_and_grads(m):
    b = batch(m)
    assert b["proprio"].dtype == torch.float32 and next(m.parameters()).dtype == torch.bfloat16
    m.train(); m.zero_grad()
    loss = m(b)["loss"]
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    g = [p.grad for n, p in m.named_parameters() if n.startswith("dit.")]
    assert g and all(x is not None and torch.isfinite(x.float()).all() for x in g)
    assert any(x.float().abs().sum() > 0 for x in g)


def test_predict_shape_finite(m):
    m.eval()
    a = m.predict(batch(m))
    assert a.shape == (3, 4, 4) and torch.isfinite(a.float()).all() and a.float().abs().max() <= 1


def test_flow_engine_and_rl_logp_bf16(m):
    m.eval(); b = batch(m)
    r = FlowEngine(m, 3, 0.7).generate(b, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(r["logp"].float()).all() and r["x"].shape[:2] == (3, 3)
    with torch.no_grad():
        o = m({**b, "x": r["x"], "x_next": r["x_next"], "t": r["t"], "sde_a": 0.7})
    assert o["logp"].shape == (3, 3) and torch.isfinite(o["logp"]).all()
    # bf16 resolves ~3 digits and |logp| is ~5-25 here, so 1e-2 absolute is below one bf16 ulp (0.06-0.125)
    assert torch.allclose(o["logp"].float(), r["logp"].float(), atol=0.1, rtol=1e-2), (o["logp"] - r["logp"].float()).abs().max()


def test_euler_sample_fp32_noise_with_bf16_params(m):
    m.eval(); b = batch(m)
    with torch.no_grad():
        prefix, mask = m.embed_prefix(b)
        h = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask)).last_hidden_state
        noise = torch.randn(3, 4, 4, dtype=torch.float32)
        out = euler_sample(m.dit, b["proprio"].to(h.dtype), h, mask, 4, noise=noise)
        out2 = euler_sample(m.dit, b["proprio"], h, mask, 4, noise=noise)      # fp32 proprio too
    assert out.shape == (3, 4, 4) and torch.isfinite(out.float()).all() and torch.isfinite(out2.float()).all()
