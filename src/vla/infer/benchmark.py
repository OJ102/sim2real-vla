"""P7 inference benchmark (PLAN v2). Rows (discrete head unless noted), each at batch 1 and 64:
  1 fp32 eager, no cache, full lm_head      2 bf16 eager, no cache, full lm_head
  3 bf16 SDPA + KV cache, full lm_head      4 row 3 + sliced action head
  5 row 4 + static cache + CUDA graphs      6 flow head: 1 prefill + 10 DiT steps, CUDA graphs (needs --flow-ckpt/config)
Latency per query (16 action tokens / one 4-step chunk): p50/p95 over all timed iterations, median/min/max of the
per-repeat medians (5 repeats, interleaved row order per repeat). Token agreement vs row 2 on --n-obs fixed observations.
Roofline: bytes read per decoded token, HBM floor at 2.0 TB/s, measured/floor ratio, lm_head byte share before/after slicing.

    uv run python -m vla.infer.benchmark --ckpt $VLA_RUNS/bc_disc/final.pt --data-dir $VLA_DATA/demos \
        --out results/latency.json [--flow-ckpt $VLA_RUNS/bc_flow/final.pt] [--quiet-wait] [--trace results/traces]
Every row records gpu_snapshot before/after and a contended flag; contended rows are flagged, not hidden.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vla.infer.engine import InferenceEngine
from vla.model.prompts import encode_instructions
from vla.utils.gpu import quiet_wait, snapshot, visible_gpus

HBM_BPS = 2.0e12  # A100-80GB SXM peak HBM bandwidth (bytes/s)
PEAK_FLOPS = 312e12


# ----------------------------------------------------------------------------- decode variants (rows 1-4)
@torch.no_grad()
def decode_nocache(model, batch, sliced: bool = False) -> torch.Tensor:
    """Recompute the whole sequence for every token. Constrained to action ids; full lm_head unless `sliced`."""
    prefix, mask = model.embed_prefix(batch)
    emb = model.llm.get_input_embeddings()
    toks = []
    x, m = prefix, mask
    for i in range(model.n_act_tokens):
        h = model.llm.model(inputs_embeds=x, attention_mask=m, position_ids=model._pos(m), use_cache=False
                            ).last_hidden_state[:, -1]
        tok = _argmax_action(model, h, sliced)
        toks.append(tok)
        x = torch.cat([x, emb(tok + model.act_base)[:, None]], 1)
        m = torch.cat([m, torch.ones(m.shape[0], 1, dtype=m.dtype, device=m.device)], 1)
    return torch.stack(toks, 1)


@torch.no_grad()
def decode_cache(model, batch, sliced: bool) -> torch.Tensor:
    prefix, mask = model.embed_prefix(batch)
    emb = model.llm.get_input_embeddings()
    out = model.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=model._pos(mask), use_cache=True)
    cache, h = out.past_key_values, out.last_hidden_state[:, -1]
    toks = []
    for i in range(model.n_act_tokens):
        tok = _argmax_action(model, h, sliced)
        toks.append(tok)
        if i == model.n_act_tokens - 1:
            break
        mask = torch.cat([mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)], 1)
        out = model.llm.model(inputs_embeds=emb(tok + model.act_base)[:, None], attention_mask=mask,
                              position_ids=model._pos(mask)[:, -1:], past_key_values=cache, use_cache=True)
        cache, h = out.past_key_values, out.last_hidden_state[:, -1]
    return torch.stack(toks, 1)


def _argmax_action(model, h, sliced: bool) -> torch.Tensor:
    if sliced:
        return model.act_head(h).argmax(-1)
    full = model.llm.lm_head(h)                                  # [B, vocab(+256)]: the baseline cost
    return full[:, model.act_base:model.act_base + model.cfg.n_bins].argmax(-1)


# ----------------------------------------------------------------------------- flow with CUDA graphs (row 6)
class FlowGraphRunner:
    """Prefill eager, then the full 10-step Euler loop of the DiT in one CUDA graph (static ctx buffers)."""

    def __init__(self, model, batch, steps: int, use_graph: bool = True):
        self.m, self.steps = model, steps
        prefix, mask = model.embed_prefix(batch)
        B, P, D = prefix.shape
        self.B, self.P = B, P
        dt = prefix.dtype
        dev = prefix.device
        self.ctx = torch.zeros(B, P, D, device=dev, dtype=dt)
        self.ctx_mask = torch.zeros(B, P, device=dev, dtype=mask.dtype)
        self.proprio = torch.zeros(B, model.cfg.proprio_dim, device=dev, dtype=dt)
        self.noise = torch.zeros(B, model.dit.chunk, model.dit.act_dim, device=dev, dtype=dt)
        self.out = None
        self.graph = None
        self.status = "eager"
        if use_graph and dev.type == "cuda":
            self._capture()

    def _loop(self):
        from vla.model.flow_head import euler_sample

        return euler_sample(self.m.dit, self.proprio, self.ctx, self.ctx_mask, self.steps, self.noise)

    def _capture(self):
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._loop()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.out = self._loop()
            self.graph, self.status = g, "captured"
        except Exception as e:  # noqa: BLE001
            self.status = f"capture failed: {type(e).__name__}: {str(e)[:160]}"

    @torch.no_grad()
    def __call__(self, batch, noise=None):
        m = self.m
        prefix, mask = m.embed_prefix(batch)
        h = m.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=m._pos(mask),
                        use_cache=False).last_hidden_state
        self.ctx.copy_(h)
        self.ctx_mask.copy_(mask)
        self.proprio.copy_(batch["proprio"].to(self.proprio.dtype))
        self.noise.copy_(torch.randn_like(self.noise) if noise is None else noise)
        if self.graph is not None:
            self.graph.replay()
            return self.out
        return self._loop()


# ----------------------------------------------------------------------------- measurement helpers
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def pct(v, q):
    return float(np.percentile(v, q))


def make_batches(model, data_dir, n_obs, device, dtype_img=None):
    """n_obs fixed observations: from demo frames (VLADataset) if given, else seeded synthetic."""
    if data_dir:
        from vla.data.dataset import VLADataset

        ds = VLADataset(data_dir, chunk=model.cfg.chunk, part="val")
        idx = np.linspace(0, len(ds) - 1, n_obs).astype(int)
        items = [ds[int(i)] for i in idx]
        img = torch.stack([it["image"] for it in items])
        pro = torch.stack([it["proprio"] for it in items])
        ins = [it["instruction"] for it in items]
    else:
        g = torch.Generator().manual_seed(0)
        img = torch.randint(0, 255, (n_obs, 128, 128, 3), dtype=torch.uint8, generator=g)
        pro = torch.randn(n_obs, model.cfg.proprio_dim, generator=g)
        ins = ["lift the red cube", "push the blue cube to the yellow zone", "reach the green cube"] * (n_obs // 3 + 1)
        ins = ins[:n_obs]
    tok = encode_instructions(model.tokenizer, ins)
    return {"image": img.to(device), "proprio": pro.to(device), **{k: v.to(device) for k, v in tok.items()}}


def take(batch, idx):
    return {k: v[idx] for k, v in batch.items()}


def roofline(model, prefix_len: int, sliced: bool, B: int, dtype_bytes: int) -> dict:
    c = model.llm.config
    n_layer, d = c.num_hidden_layers, c.hidden_size
    kvh = c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or d // c.num_attention_heads
    emb_rows = model.llm.get_input_embeddings().weight.shape[0]
    non_emb = sum(p.numel() for p in model.llm.model.parameters()) - emb_rows * d  # transformer blocks + final norm
    head_full = emb_rows * d
    head_sliced = model.cfg.n_bins * d
    avg_seq = prefix_len + model.n_act_tokens / 2
    kv_bytes = n_layer * 2 * kvh * hd * avg_seq * dtype_bytes * B  # KV read per decode step for the whole batch
    w_bytes = (non_emb + (head_sliced if sliced else head_full)) * dtype_bytes
    per_step = w_bytes + kv_bytes
    flops = 2 * (non_emb + (head_sliced if sliced else head_full)) * B + 4 * n_layer * kvh * hd * avg_seq * B
    return {"weight_bytes": w_bytes, "kv_bytes_per_step": kv_bytes, "bytes_per_step": per_step,
            "head_bytes": (head_sliced if sliced else head_full) * dtype_bytes,
            "head_share_of_weight_bytes": (head_sliced if sliced else head_full) * dtype_bytes / w_bytes,
            "hbm_floor_ms_per_step": per_step / HBM_BPS * 1e3,
            "arith_intensity_flop_per_byte": flops / per_step, "ridge_flop_per_byte": PEAK_FLOPS / HBM_BPS}


def prefill_bytes(model, dtype_bytes):
    n = sum(p.numel() for p in model.vision.parameters()) + sum(p.numel() for p in model.llm.model.parameters())
    return n * dtype_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="discrete-head weights (final.pt / DCP dir); random init if omitted")
    ap.add_argument("--flow-ckpt", default=None)
    ap.add_argument("--model-config", default="configs/model_0p5b.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--batches", default="1,64")
    ap.add_argument("--rows", default="1,2,3,4,5,6")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--n-obs", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet-wait", action="store_true")
    ap.add_argument("--trace", default=None, help="dir: kernel-gap profiler traces for rows 2/4/5 at batch 1")
    ap.add_argument("--out", default="results/latency.json")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    rows = [int(x) for x in a.rows.split(",")]
    dev = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cuda = dev.type == "cuda"
    from vla.infer.policy import VLAPolicy

    def build(ckpt, head, dtype):
        pol = VLAPolicy(a.model_config, ckpt, None, head, str(dev), dtype, a.override)
        return pol.model

    models = {"fp32": build(a.ckpt, "discrete", torch.float32)}
    if cuda or True:
        models["bf16"] = build(a.ckpt, "discrete", torch.bfloat16)
    flow = build(a.flow_ckpt, "flow", torch.bfloat16) if 6 in rows else None
    allobs = make_batches(models["bf16"], a.data_dir, a.n_obs, dev)
    bs = [int(x) for x in a.batches.split(",")]
    meta = {"device": str(dev), "gpu_name": torch.cuda.get_device_name(dev) if cuda else "cpu",
            "torch": torch.__version__, "repeats": a.repeats, "iters": a.iters, "warmup": a.warmup, "n_obs": a.n_obs,
            "ckpt": a.ckpt, "flow_ckpt": a.flow_ckpt, "hbm_bytes_per_s": HBM_BPS}
    if a.quiet_wait:
        meta["quiet_wait"] = quiet_wait(visible_gpus() or [0], 15, 60, 1200)

    # row definitions: name -> (model key, fn(batch) -> tokens)
    engines: dict = {}

    def row_fn(r, B):
        m32, m16 = models["fp32"], models["bf16"]
        if r == 1:
            return lambda b: decode_nocache(m32, b, sliced=False)
        if r == 2:
            return lambda b: decode_nocache(m16, b, sliced=False)
        if r == 3:
            return lambda b: decode_cache(m16, b, sliced=False)
        if r == 4:
            return lambda b: decode_cache(m16, b, sliced=True)
        if r == 5:
            key = ("g", B)
            if key not in engines:
                engines[key] = InferenceEngine(m16, 1.0, "graph" if cuda else "static",
                                               max_prefix=(m16.cfg.image_size // 16) ** 2 + 1 + 32)
            return lambda b: engines[key].generate(b, greedy=True)["bins"]
        if r == 6:
            key = ("f", B)
            if key not in engines:
                engines[key] = FlowGraphRunner(flow, take(allobs, slice(0, B)), flow.cfg.flow.steps, cuda)
            return lambda b: engines[key](b)
        raise ValueError(r)

    # reference tokens: row 2 on all n_obs observations (chunks of <=64)
    ref_tokens = torch.cat([decode_nocache(models["bf16"], take(allobs, slice(i, i + 50)), False)
                            for i in range(0, a.n_obs, 50)])
    results = []
    names = {1: "fp32 eager, no cache", 2: "bf16 eager, no cache", 3: "bf16 SDPA + KV cache",
             4: "bf16 + KV cache + sliced head", 5: "row4 + static cache + CUDA graph", 6: "flow head, 10 steps, graphs"}
    samples: dict = {}
    for rep in range(a.repeats):  # interleaved: all rows once per pass so drift cancels
        for r in rows:
            for B in bs:
                batch = take(allobs, slice(0, B))
                fn = row_fn(r, B)
                ts = time_fn(lambda: fn(batch), a.warmup if rep == 0 else 3, a.iters)
                samples.setdefault((r, B), []).append(ts)
    for r in rows:
        for B in bs:
            reps = samples[(r, B)]
            allts = [t for ts in reps for t in ts]
            rec = {"row": r, "name": names[r], "batch": B, "p50_ms_per_query_batch": pct(allts, 50),
                   "p95_ms_per_query_batch": pct(allts, 95),
                   "p50_ms_per_query": pct(allts, 50) / B,  # amortised over the batch
                   "p50_ms_per_action": pct(allts, 50) / B / model_chunk(models["bf16"]),
                   "repeat_p50_median_ms": statistics.median(pct(t, 50) for t in reps),
                   "repeat_p50_min_ms": min(pct(t, 50) for t in reps), "repeat_p50_max_ms": max(pct(t, 50) for t in reps),
                   "n_repeats": len(reps), "peak_mem_gb": (torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else None)}
            if r <= 5:
                fn = row_fn(r, B)
                toks = torch.cat([fn(take(allobs, slice(i, min(i + B, a.n_obs)))) for i in range(0, a.n_obs - a.n_obs % B or a.n_obs, B)]) \
                    if B <= a.n_obs else None
                n = min(len(toks), len(ref_tokens)) if toks is not None else 0
                rec["token_agreement_vs_row2"] = float((toks[:n] == ref_tokens[:n]).float().mean()) if n else None
                prefix_len = (models["bf16"].cfg.image_size // 16) ** 2 + 1 + allobs["input_ids"].shape[1]
                rf = roofline(models["bf16"], prefix_len, sliced=r >= 4, B=B,
                              dtype_bytes=4 if r == 1 else 2)
                rec["roofline"] = rf
                steps = models["bf16"].n_act_tokens
                floor_query = (steps * rf["hbm_floor_ms_per_step"]
                               + prefill_bytes(models["bf16"], 4 if r == 1 else 2) / HBM_BPS * 1e3)
                rec["hbm_floor_ms_per_query_batch"] = floor_query
                rec["measured_over_floor"] = rec["p50_ms_per_query_batch"] / floor_query
            if r == 5:
                rec["graph_status"] = {str(k[1]): v.graph_status for k, v in engines.items() if k[0] == "g"}
            if r == 6:
                rec["graph_status"] = {str(k[1]): v.status for k, v in engines.items() if k[0] == "f"}
            rec["gpu_snapshot"] = snapshot(gpus=visible_gpus() or None)
            rec["contended"] = bool(rec["gpu_snapshot"].get("contended"))
            results.append(rec)
            print(f"row {r} B={B:3d} {names[r]:34s} p50 {rec['p50_ms_per_query_batch']:8.1f} ms/batch "
                  f"{rec['p50_ms_per_query']:7.2f} ms/query  agree={rec.get('token_agreement_vs_row2')} "
                  f"floor-ratio={rec.get('measured_over_floor')}", flush=True)
    if a.trace and cuda:
        trace_rows(a.trace, models, allobs, engines, row_fn)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    meta["gpu_snapshot_after"] = snapshot(gpus=visible_gpus() or None)
    json.dump({"meta": meta, "rows": results}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


def model_chunk(m):
    return m.cfg.chunk


def trace_rows(out_dir, models, allobs, engines, row_fn):
    """Chrome traces (kernel gaps) for rows 2, 4, 5 at batch 1: eager is launch-overhead bound, graphs are not."""
    from torch.profiler import ProfilerActivity, profile

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    b = take(allobs, slice(0, 1))
    for r in (2, 4, 5):
        fn = row_fn(r, 1)
        for _ in range(3):
            fn(b)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn(b)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(Path(out_dir) / f"decode_row{r}_b1.json"))


if __name__ == "__main__":
    main()
