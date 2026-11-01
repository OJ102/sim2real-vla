"""Guards the _discrete_logits refactor (grounding plan step 4): outputs must equal frozen pre-refactor references."""
from pathlib import Path

import pytest
import torch

from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
REF = Path(__file__).parent / "data" / "disc_logits_ref.pt"


def build():
    torch.manual_seed(1234)
    return VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ["head=discrete"])).eval()


def make_batch(m, B=3):
    g = torch.Generator().manual_seed(7)
    return {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(B, 11, generator=g), "actions": torch.rand(B, 4, 4, generator=g) * 2 - 1,
            **encode_instructions(m.tokenizer, ["lift the red cube", "push the blue cube", "reach it"][:B])}


@pytest.fixture(scope="module")
def ref():
    assert REF.is_file(), "reference captured from pre-refactor code is missing"
    return torch.load(REF, weights_only=True)


def test_discrete_logits_identical_to_pre_refactor(ref):
    m = build(); b = make_batch(m)
    with torch.no_grad():
        prefix, mask = m.embed_prefix(b)
        bins = m.act_tok.encode_bins(b["actions"].reshape(3, -1))
        out = m._discrete_logits(prefix, mask, bins)
    logits = out[0] if isinstance(out, tuple) else out          # refactor returns (logits, h[:, P-1])
    assert torch.equal(bins, ref["bins"])
    assert logits.shape == ref["logits"].shape == (3, 16, 256)
    assert torch.allclose(logits, ref["logits"], atol=0, rtol=0), (logits - ref["logits"]).abs().max()
    if isinstance(out, tuple):
        assert out[1].shape == (3, m.llm.config.hidden_size)


def test_token_logprobs_loss_and_predict_unchanged(ref):
    m = build(); b = make_batch(m)
    with torch.no_grad():
        bins = ref["bins"]
        assert torch.equal(m.token_logprobs(b, bins), ref["logp"])
        assert torch.equal(m.token_logprobs(b, bins, 0.7), ref["logp_t07"])
        assert torch.equal(m(b)["loss"], ref["loss"])               # with ground_weight=0 the loss is the plain CE
        assert torch.equal(m.predict(b), ref["pred"])
    assert not hasattr(m, "ground_head") or m.ground_head is None


def test_rl_bins_route_unchanged(ref):
    m = build(); b = make_batch(m)
    with torch.no_grad():
        assert torch.equal(m({**b, "bins": ref["bins"]})["logp"], ref["logp"])
