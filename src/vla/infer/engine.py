"""Batched action-sampling engine shared by RL rollout workers and the P7 latency benchmark.

Modes
  eager  : HF dynamic KV cache, op-by-op (reference path, always available).
  static : preallocated StaticCache + fixed-shape decode step, still op-by-op.
  graph  : `static` with the decode step captured in a CUDA graph (CUDA only; falls back to `static` with
           `graph_status` explaining why if capture fails).
Sliced action head everywhere (256 rows). Weight updates must `copy_` in place (see rl/weight_sync.py): parameter
storage is unchanged, so captured graphs stay valid. `verify_matches_eager` checks that after a weight update.

Static layout: cache has `max_len = max_prefix + n_act` slots. Prefill (eager, variable prefix length P <= max_prefix)
writes slots [0, P); decode step i writes slot P+i. A static 2D mask [B, max_len] marks valid slots (prompt padding is 0);
position ids = (#valid prefix tokens) + i.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from vla.model.vla import VLA


class _StaticDecoder:
    def __init__(self, model: VLA, B: int, max_len: int, device, dtype):
        from transformers import StaticCache

        self.m, self.B, self.max_len = model, B, max_len
        llm = model.llm
        cfg = llm.config
        self.cache = StaticCache(config=cfg, max_cache_len=max_len)
        kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.cache.early_initialization(B, kv_heads, head_dim, dtype, device)  # allocate now: fixed addresses
        d = cfg.hidden_size
        self.embeds = torch.zeros(B, 1, d, device=device, dtype=dtype)
        self.pos = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.cache_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        self.logits = None          # static output of the (captured) step
        self.graph = None
        self.graph_status = "not captured"

    def reset(self):
        self.cache.reset()
        self.mask.zero_()

    def step(self) -> torch.Tensor:
        out = self.m.llm.model(inputs_embeds=self.embeds, attention_mask=self.mask, position_ids=self.pos,
                               past_key_values=self.cache, cache_position=self.cache_pos, use_cache=True)
        return self.m.act_head(out.last_hidden_state[:, -1]).float()

    def _counters(self):
        """HF StaticLayer.update ignores the cache_position we pass: it writes at its own device counter
        `cumulative_length` (arange(kv_len) + counter) and increments it in place on every call, also inside a captured
        graph. Warm-up steps therefore advance the counter; it must be restored or every later write lands at the wrong
        slot (the first replayed decode step would write slot P+3 instead of P)."""
        return [l.cumulative_length for l in self.cache.layers]

    def _warmup(self, n: int = 3) -> None:
        saved = [c.clone() for c in self._counters()]
        for _ in range(n):
            self.step()
        for c, v in zip(self._counters(), saved):
            c.copy_(v)

    def capture(self, fake: bool = False) -> str:
        """fake=True (CPU tests): run the warm-up + counter bookkeeping exactly as the CUDA path does, but capture nothing."""
        if not self.embeds.is_cuda and not fake:
            self.graph_status = "cpu: no CUDA graphs"
            return self.graph_status
        if fake:
            self._warmup()
            self.graph_status = "fake capture (CPU bookkeeping test)"
            return self.graph_status
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._warmup()  # allocator/kernel warm-up outside capture; restores the cache write counters
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            saved = [c.clone() for c in self._counters()]
            with torch.cuda.graph(g):
                self.logits = self.step()  # captured, not executed
            for c, v in zip(self._counters(), saved):  # nothing ran, but restore defensively
                c.copy_(v)
            self.graph, self.graph_status = g, "captured"
        except Exception as e:  # noqa: BLE001  (e.g. a host sync inside mask creation)
            self.graph, self.graph_status = None, f"capture failed: {type(e).__name__}: {str(e)[:200]}"
        return self.graph_status

    def run(self) -> torch.Tensor:
        if self.graph is not None:
            self.graph.replay()
            return self.logits
        return self.step()


class InferenceEngine:
    def __init__(self, model: VLA, temperature: float = 1.0, mode: str = "eager", max_prefix: int | None = None,
                 fake_capture: bool = False):
        assert model.cfg.head == "discrete", "flow rollout engine comes with P6b"
        assert mode in ("eager", "static", "graph")
        self.model, self.temperature, self.mode = model, temperature, mode
        self.fake_capture = fake_capture  # CPU test hook: exercise the capture bookkeeping without CUDA graphs
        self.version = 0
        self.max_prefix = max_prefix
        self._dec: dict[int, _StaticDecoder] = {}

    @property
    def graph_status(self) -> dict:
        return {B: d.graph_status for B, d in self._dec.items()}

    def _decoder(self, B: int, prefix_len: int, device, dtype) -> _StaticDecoder:
        mp = self.max_prefix or prefix_len
        n = self.model.n_act_tokens
        if B in self._dec and self._dec[B].max_len - n < prefix_len:
            # a prompt longer than the preallocated budget must never crash a rollout: reallocate (graph is recaptured)
            warnings.warn(f"prefix length {prefix_len} > max_prefix {self._dec[B].max_len - n}: reallocating the static "
                          f"cache for batch {B}" + (" and recapturing the CUDA graph" if self.mode == "graph" else ""))
            del self._dec[B]
        if B not in self._dec:
            self._dec[B] = _StaticDecoder(self.model, B, max(mp, prefix_len) + n, device, dtype)
        return self._dec[B]

    def _pick(self, logits: torch.Tensor, greedy: bool, generator):
        logp = F.log_softmax(logits / self.temperature, dim=-1)
        tok = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        return tok, logp.gather(-1, tok[:, None]).squeeze(-1)

    @torch.no_grad()
    def generate(self, batch: dict, greedy: bool = False, generator: torch.Generator | None = None) -> dict:
        """-> {"bins" [B,16], "logp" [B,16] behaviour log-prob under softmax(logits/T), "actions" [B,H,act_dim],
        "version"}."""
        m = self.model
        prefix, mask = m.embed_prefix(batch)
        B, P = prefix.shape[:2]
        emb = m.llm.get_input_embeddings()
        n = m.n_act_tokens
        toks, logps = [], []
        if self.mode == "eager":
            out = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask), use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[:, -1]
            for i in range(n):
                tok, lp = self._pick(m.act_head(h).float(), greedy, generator)
                toks.append(tok); logps.append(lp)
                if i == n - 1:
                    break
                mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=mask.device)], 1)
                out = m.llm.model(inputs_embeds=emb(tok + m.act_base)[:, None], attention_mask=mask,
                                  position_ids=m._pos(mask)[:, -1:], past_key_values=cache, use_cache=True)
                cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        else:
            dec = self._decoder(B, P, prefix.device, prefix.dtype)
            dec.reset()
            dec.mask[:, :P] = mask
            out = m.llm.model(inputs_embeds=prefix, attention_mask=dec.mask, position_ids=m._pos(mask),
                              past_key_values=dec.cache, cache_position=torch.arange(P, device=prefix.device),
                              use_cache=True)
            h = out.last_hidden_state[:, -1]
            if self.mode == "graph" and dec.graph is None and dec.graph_status == "not captured":
                # capture after prefill so the cache/mask buffers are in a valid state; step() warmup only touches slot P
                dec.cache_pos.fill_(P); dec.pos.fill_(0); dec.embeds.zero_(); dec.mask[:, P] = 1
                dec.capture(fake=self.fake_capture)
                dec.mask[:, P] = 0
            valid = mask.sum(1)
            logits = m.act_head(h).float()
            for i in range(n):
                tok, lp = self._pick(logits, greedy, generator)
                toks.append(tok); logps.append(lp)
                if i == n - 1:
                    break
                dec.embeds.copy_(emb(tok + m.act_base)[:, None])
                dec.pos.copy_((valid + i)[:, None])
                dec.cache_pos.fill_(P + i)
                dec.mask[:, P + i] = 1
                logits = dec.run()
        bins = torch.stack(toks, 1)
        actions = m.act_tok.decode_bins(bins.float()).reshape(B, m.cfg.chunk, m.cfg.act_dim)
        return {"bins": bins, "logp": torch.stack(logps, 1), "actions": actions, "version": self.version}

    @torch.no_grad()
    def verify_matches_eager(self, batch: dict) -> dict:
        """Greedy-decode `batch` with this engine's mode and with the eager reference on the CURRENT weights
        (call after a weight `copy_` sync). Returns token agreement and max |logp diff|."""
        ref = InferenceEngine(self.model, self.temperature, "eager").generate(batch, greedy=True)
        got = self.generate(batch, greedy=True)
        return {"token_agreement": float((ref["bins"] == got["bins"]).float().mean()),
                "max_logp_diff": float((ref["logp"] - got["logp"]).abs().max()), "graph_status": self.graph_status}


class FlowEngine:
    """Rollout engine for the flow head: backbone prefill (no KV cache needed: the DiT cross-attends to the prefix
    hidden states) + K-step Flow-GRPO SDE sampling. Stores per-step latents and behaviour log-probs for RL."""

    def __init__(self, model: VLA, K: int = 5, sde_a: float = 0.7):
        assert model.cfg.head == "flow"
        self.model, self.K, self.a, self.version = model, K, sde_a, 0

    @torch.no_grad()
    def generate(self, batch: dict, greedy: bool = False, generator: torch.Generator | None = None) -> dict:
        from vla.model.flow_head import euler_sample, flow_sde_sample

        m = self.model
        prefix, mask = m.embed_prefix(batch)
        h = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask),
                        use_cache=False).last_hidden_state
        pro = batch["proprio"].to(h.dtype)
        if greedy:  # eval: deterministic 10-step ODE
            act = euler_sample(m.dit, pro, h, mask, m.cfg.flow.steps)
            return {"actions": act, "version": self.version}
        out = flow_sde_sample(m.dit, pro, h, mask, self.K, self.a, generator)
        out["version"] = self.version
        return out


def make_engine(model: VLA, temperature: float = 1.0, mode: str = "eager", max_prefix: int | None = None,
                flow_K: int = 5, sde_a: float = 0.7):
    if model.cfg.head == "flow":
        return FlowEngine(model, flow_K, sde_a)
    return InferenceEngine(model, temperature, mode, max_prefix)
