"""Batched action-sampling engine shared by RL rollout workers (and, later, the P7 latency benchmark).

Current state (skeleton): eager decode with a dynamic KV cache, sliced action head, per-token behaviour log-probs.
TODO (GPU, P6/P7): static KV cache + CUDA-graph capture of the decode step; weight updates must `copy_` in place
(see rl/weight_sync.py) so captured graphs stay valid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vla.model.vla import VLA


class InferenceEngine:
    def __init__(self, model: VLA, temperature: float = 1.0):
        assert model.cfg.head == "discrete", "flow rollout engine comes with P6b"
        self.model, self.temperature = model, temperature
        self.version = 0

    @torch.no_grad()
    def generate(self, batch: dict, greedy: bool = False, generator: torch.Generator | None = None) -> dict:
        """-> {"bins" [B,16] long, "logp" [B,16] behaviour log-prob of each sampled token (under softmax(logits/T)),
        "actions" [B,H,act_dim] in [-1,1], "version"}."""
        m = self.model
        prefix, mask = m.embed_prefix(batch)
        b = prefix.shape[0]
        out = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask), use_cache=True)
        cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        emb = m.llm.get_input_embeddings()
        toks, logps = [], []
        for i in range(m.n_act_tokens):
            logits = m.act_head(h).float() / self.temperature   # live slice view: always current weights
            logp = F.log_softmax(logits, dim=-1)
            tok = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            toks.append(tok)
            logps.append(logp.gather(-1, tok[:, None]).squeeze(-1))
            if i == m.n_act_tokens - 1:
                break
            mask = torch.cat([mask, torch.ones(b, 1, dtype=mask.dtype, device=mask.device)], 1)
            out = m.llm.model(inputs_embeds=emb(tok + m.act_base)[:, None], attention_mask=mask,
                              position_ids=m._pos(mask)[:, -1:], past_key_values=cache, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        bins = torch.stack(toks, 1)
        actions = m.act_tok.decode_bins(bins.float()).reshape(b, m.cfg.chunk, m.cfg.act_dim)
        return {"bins": bins, "logp": torch.stack(logps, 1), "actions": actions, "version": self.version}
