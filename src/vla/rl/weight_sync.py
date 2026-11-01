"""Learner -> rollout weight sync: bucketed broadcast over `sync_pg`, in-place `copy_` on the receivers.

Receivers keep their parameter storage (same data_ptr), so CUDA graphs captured against the rollout model stay valid.
Sender side accepts plain or FSDP2 (DTensor) params: each param is gathered with `full_tensor()` one bucket at a time,
so the full model is never materialised at once.
"""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn as nn

BUCKET_BYTES = 256 << 20


def clean_name(n: str) -> str:
    return n.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "").replace("_fsdp_wrapped_module.", "")


def _items(model: nn.Module, sync_dtype):
    return [(clean_name(n), p, sync_dtype or p.dtype) for n, p in model.named_parameters()]


def plan_buckets(items, bucket_bytes: int = BUCKET_BYTES) -> list[list[int]]:
    """Group consecutive params (same dtype) into buckets of <= bucket_bytes. Deterministic given shapes/dtypes."""
    buckets, cur, cur_bytes, cur_dtype = [], [], 0, None
    for i, (_, p, dt) in enumerate(items):
        nbytes = p.numel() * torch.empty((), dtype=dt).element_size()
        if cur and (dt != cur_dtype or cur_bytes + nbytes > bucket_bytes):
            buckets.append(cur)
            cur, cur_bytes = [], 0
        cur.append(i)
        cur_bytes += nbytes
        cur_dtype = dt
    if cur:
        buckets.append(cur)
    return buckets


@torch.no_grad()
def sync_weights(model: nn.Module, pg, src: int, version: int = 0, bucket_bytes: int = BUCKET_BYTES,
                 sync_dtype: torch.dtype | None = None) -> dict:
    """Broadcast `model`'s params from global rank `src`. Every rank of `pg` calls this with a model of identical
    structure. Returns {version, latency_ms, nbytes, n_buckets}; on receivers `version` is the sender's version."""
    is_sender = dist.get_rank() == src
    items = _items(model, sync_dtype)
    plan = plan_buckets(items, bucket_bytes)
    dev = items[0][1].device if not hasattr(items[0][1], "device_mesh") else items[0][1].to_local().device
    t0 = time.perf_counter()
    ver = torch.tensor([version], dtype=torch.int64, device=dev)
    dist.broadcast(ver, src=src, group=pg)
    nbytes = 0
    for bucket in plan:
        dt = items[bucket[0]][2]
        numels = [items[i][1].numel() for i in bucket]
        if is_sender:
            flat = torch.cat([_full(items[i][1]).reshape(-1).to(dt) for i in bucket])
        else:
            flat = torch.empty(sum(numels), dtype=dt, device=dev)
        dist.broadcast(flat, src=src, group=pg)
        nbytes += flat.numel() * flat.element_size()
        if not is_sender:
            off = 0
            for i, n in zip(bucket, numels):
                p = items[i][1]
                p.data.copy_(flat[off:off + n].view(p.shape))  # in place: storage (and CUDA graphs) preserved
                off += n
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return {"version": int(ver.item()), "latency_ms": (time.perf_counter() - t0) * 1e3, "nbytes": nbytes,
            "n_buckets": len(plan)}


def _full(p: torch.Tensor) -> torch.Tensor:
    return p.full_tensor() if hasattr(p, "full_tensor") else p.detach()
