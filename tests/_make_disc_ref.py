"""Run ONCE against pre-refactor code to freeze reference outputs: uv run python tests/_make_disc_ref.py"""
import torch
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config
from test_disc_logits_unchanged import build, make_batch  # noqa

m, b = build(), None
b = make_batch(m)
with torch.no_grad():
    prefix, mask = m.embed_prefix(b)
    bins = m.act_tok.encode_bins(b["actions"].reshape(b["actions"].shape[0], -1))
    out = m._discrete_logits(prefix, mask, bins)
    logits = out[0] if isinstance(out, tuple) else out
    ref = {"logits": logits.clone(), "bins": bins, "logp": m.token_logprobs(b, bins), "logp_t07": m.token_logprobs(b, bins, 0.7),
           "loss": m(b)["loss"].clone(), "pred": m.predict(b)}
torch.save(ref, "tests/data/disc_logits_ref.pt")
print({k: tuple(v.shape) for k, v in ref.items()}, ref["loss"].item())
