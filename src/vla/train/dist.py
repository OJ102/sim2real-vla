"""Process-group init and model wrapping: FSDP2 (fully_shard) and DDP."""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn


def init_dist(timeout_min: int = 30):
    """-> (rank, world, local_rank, device). nccl when CUDA is available, else gloo (CPU smoke tests).
    Works without torchrun (single process, no process group)."""
    cuda = torch.cuda.is_available()
    if "RANK" not in os.environ:
        return 0, 1, 0, torch.device("cuda:0" if cuda else "cpu")
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    if cuda:
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=timeout_min))
    return rank, world, local, torch.device(f"cuda:{local}" if cuda else "cpu")


def layer_modules(model: nn.Module) -> list[nn.Module]:
    """Transformer blocks to shard individually: Qwen2DecoderLayer (+DiT blocks) and SiglipEncoderLayer."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

    from vla.model.flow_head import DiTBlock

    return [m for m in model.modules() if isinstance(m, (Qwen2DecoderLayer, SiglipEncoderLayer, DiTBlock))]


def wrap_fsdp2(model: nn.Module, bf16: bool = True, reshard_after_forward: bool = True, prefetch: bool = False,
               act_ckpt: bool = False) -> nn.Module:
    """fully_shard each block, then the root. bf16 params / fp32 gradient reduction (cast_forward_inputs=False so
    the action targets are not rounded to bf16 before tokenisation)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if act_ckpt:
        for m in (model.llm, model.vision):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False) \
        if bf16 else None
    from torch.distributed.device_mesh import init_device_mesh

    # explicit mesh: the default picks "mps" on macOS and crashes (torch.mps has no is_initialized)
    mesh = init_device_mesh("cuda" if torch.cuda.is_available() else "cpu", (dist.get_world_size(),))
    kw = {"mesh": mesh, **({"mp_policy": mp} if mp is not None else {})}
    layers = layer_modules(model)
    for m in layers:
        fully_shard(m, reshard_after_forward=reshard_after_forward, **kw)
    fully_shard(model, reshard_after_forward=False, **kw)  # root: keep unsharded through backward (last-used params)
    if prefetch and len(layers) > 1:
        for a, b in zip(layers[:-1], layers[1:]):
            a.set_modules_to_forward_prefetch([b])
            b.set_modules_to_backward_prefetch([a])
    return model


def wrap_ddp(model: nn.Module, local_rank: int = 0, act_ckpt: bool = False) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP

    if act_ckpt:
        for m in (model.llm, model.vision):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cuda = torch.cuda.is_available()
    return DDP(model, device_ids=[local_rank] if cuda else None)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def barrier():
    if dist.is_initialized():
        dist.barrier()
