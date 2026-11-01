"""Behaviour cloning of the VLA with FSDP2 or DDP.

    torchrun --nproc_per_node=4 -m vla.train.bc --config configs/bc.yaml [key=value ...]
    torchrun --nproc_per_node=2 -m vla.train.bc --config configs/bc_tiny.yaml data_dir=... (gloo CPU smoke)

Global batch = micro_batch * world_size * grad_accum. Each run writes $VLA_RUNS/<name>/{config.yaml,metrics.jsonl,
tb/,ckpt/}. SIGTERM/SIGUSR1 -> save at the next step boundary and exit 0. --resume loads ckpt/latest.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from vla.data.dataset import VLADataset, make_collate
from vla.data.sampler import ResumableDistributedSampler
from vla.model.vla import VLA, VLAConfig, count_params
from vla.train.checkpoint import Checkpointer, PreemptFlag, export_final
from vla.train.dist import init_dist, unwrap, wrap_ddp, wrap_fsdp2
from vla.utils.config import dump_config, load_config
from vla.utils.logging import MetricsLogger


@dataclass
class BCConfig:
    name: str = "bc"
    model_config: str = "configs/model_0p5b.yaml"
    head: str = ""                      # override model head: discrete | flow
    data_dir: str = ""                  # default $VLA_DATA/demos
    subset: str = ""                    # optional subset_{pct}.json
    runs_dir: str = ""                  # default $VLA_RUNS
    backend: str = "fsdp"               # fsdp | ddp
    global_batch: int = 256
    micro_batch: int = 16               # per GPU; grad_accum = global_batch // (micro_batch * world)
    steps: int = 0                      # 0 => epochs * steps_per_epoch
    epochs: float = 8.0
    lr: float = 1e-4
    lr_vision: float = 1e-5
    weight_decay: float = 0.01
    warmup: int = 500
    grad_clip: float = 1.0
    seed: int = 0
    bf16: bool = True
    act_ckpt: bool = False
    reshard_after_forward: bool = True
    prefetch: bool = False
    num_workers: int = 4
    log_every: int = 10
    val_every: int = 500
    val_batches: int = 20
    ckpt_every: int = 100
    ckpt_mode: str = "sync"             # sync | async
    ckpt_keep: int = 2
    ckpt_mirror: str = ""               # optional rsync target (e.g. NFS) for a /dev/shm ckpt dir
    ckpt_dir: str = ""                  # default <run>/ckpt
    peak_tflops: float = 312.0          # A100 bf16
    export_final: bool = True
    exit_after_step: int = 0            # test aid: hard-exit (like kill -9, no save/flush) after this step


def flops_per_sample(model, seq_len: int) -> float:
    """Fwd+bwd FLOPs per sample: seq_len * (6*N + 12*L*d*S), L/d of the LM, N all params (PLAN v2 formula)."""
    n = count_params(model)["total"]
    c = model.llm.config
    return seq_len * (6 * n + 12 * c.num_hidden_layers * c.hidden_size * seq_len)


def lr_at(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(BCConfig, a.config, a.overrides)
    rank, world, local, device = init_dist()
    cuda = device.type == "cuda"
    run = Path(cfg.runs_dir or os.environ.get("VLA_RUNS", "runs")) / cfg.name
    data_dir = cfg.data_dir or str(Path(os.environ["VLA_DATA"]) / "demos")
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, str(run / "config.yaml"))
    log = MetricsLogger(run, enabled=rank == 0)
    preempt = PreemptFlag()

    mcfg = load_config(VLAConfig, cfg.model_config, [f"head={cfg.head}"] if cfg.head else [])
    torch.manual_seed(cfg.seed)  # identical init on all ranks (FSDP2 shards from rank-consistent full params)
    model = VLA(mcfg)
    tok = model.tokenizer
    if not cuda:
        cfg.bf16 = False
    if cuda:
        model.to(device)
    model.train()
    if cfg.backend == "fsdp":
        model = wrap_fsdp2(model, cfg.bf16, cfg.reshard_after_forward, cfg.prefetch, cfg.act_ckpt)
    elif cfg.backend == "ddp":
        model = wrap_ddp(model, local, cfg.act_ckpt)
    else:
        raise ValueError(cfg.backend)
    base = unwrap(model)

    vis = [p for n, p in model.named_parameters() if "vision." in n]
    rest = [p for n, p in model.named_parameters() if "vision." not in n]
    opt = torch.optim.AdamW([{"params": rest, "lr": cfg.lr}, {"params": vis, "lr": cfg.lr_vision}],
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    ds = VLADataset(data_dir, chunk=mcfg.chunk, part="train", subset=cfg.subset or None)
    val_ds = VLADataset(data_dir, chunk=mcfg.chunk, part="val")
    collate = make_collate(tok)
    assert cfg.global_batch % (cfg.micro_batch * world) == 0, "global_batch must divide by micro_batch*world"
    accum = cfg.global_batch // (cfg.micro_batch * world)
    per_rank_step = cfg.micro_batch * accum
    sampler = ResumableDistributedSampler(len(ds), rank, world, seed=cfg.seed)
    steps_per_epoch = sampler.num_per_rank // per_rank_step
    total = cfg.steps or int(cfg.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, cfg.warmup))
    ckpt = Checkpointer(cfg.ckpt_dir or run / "ckpt", cfg.ckpt_mode, cfg.ckpt_keep, cfg.ckpt_mirror or None)
    if rank == 0:
        print(f"world={world} backend={cfg.backend} frames={len(ds)} steps/epoch={steps_per_epoch} total={total} "
              f"micro={cfg.micro_batch} accum={accum} global={cfg.global_batch} params={count_params(base)['total']/1e6:.1f}M",
              flush=True)

    step, epoch, in_epoch = 0, 0, 0
    if a.resume:
        ex = ckpt.load_latest(model, opt, {"step": 0, "epoch": 0, "in_epoch": 0, "sched": sched.state_dict()})
        if ex is not None:
            step, epoch, in_epoch = ex["step"], ex["epoch"], ex["in_epoch"]
            sched.load_state_dict(ex["sched"])
            if rank == 0:
                print(f"resume_from={step}", flush=True)
                log.log(step, {"event_resume_from": step})

    def to_dev(b):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    def extra():
        return {"step": step, "epoch": epoch, "in_epoch": in_epoch, "sched": sched.state_dict(),
                "sampler": sampler.state_dict(consumed=in_epoch * per_rank_step)}

    def validate() -> dict:
        model.eval()
        vs = ResumableDistributedSampler(len(val_ds), rank, world, seed=0, shuffle=False)
        dl = DataLoader(val_ds, batch_size=cfg.micro_batch, sampler=vs, collate_fn=collate, num_workers=0)
        tot = torch.zeros(3, device=device)
        with torch.no_grad():
            for i, b in enumerate(dl):
                if i >= cfg.val_batches:
                    break
                torch.manual_seed(12345 + i)
                out = model(to_dev(b))
                tot += torch.tensor([out["loss"].item(), out.get("tok_acc", torch.zeros(())).item(), 1.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(tot)
        model.train()
        return {"val_loss": (tot[0] / tot[2]).item(), "val_tok_acc": (tot[1] / tot[2]).item()}

    flops_s = None
    stop = False
    t_prev = time.perf_counter()
    while step < total and not stop:
        sampler.set_epoch(epoch)
        sampler.consumed = in_epoch * per_rank_step
        dl = DataLoader(ds, batch_size=cfg.micro_batch, sampler=sampler, collate_fn=collate,
                        num_workers=cfg.num_workers, drop_last=True, persistent_workers=False)
        it = iter(dl)
        while in_epoch < steps_per_epoch and step < total:
            t0 = time.perf_counter()
            torch.manual_seed(cfg.seed * 1_000_003 + step * world + rank)  # deterministic per step, resume-safe
            loss_acc = acc_acc = 0.0
            for mb in range(accum):
                b = to_dev(next(it))
                ctx = model.no_sync() if (cfg.backend == "ddp" and mb < accum - 1) else contextlib.nullcontext()
                with ctx, (torch.autocast("cuda", dtype=torch.bfloat16) if (cuda and cfg.backend == "ddp" and cfg.bf16)
                           else contextlib.nullcontext()):
                    out = model(b)
                    (out["loss"] / accum).backward()
                loss_acc += out["loss"].item() / accum
                acc_acc += out.get("tok_acc", torch.zeros(())).item() / accum
            if flops_s is None:
                n_img = (mcfg.image_size // 16) ** 2 if not mcfg.tiny else (mcfg.image_size // 16) ** 2
                flops_s = flops_per_sample(base, n_img + 1 + b["input_ids"].shape[1] + base.n_act_tokens)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for g, base_lr in zip(opt.param_groups, (cfg.lr, cfg.lr_vision)):
                g["lr"] = base_lr * lr_at(step, total, cfg.warmup)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            in_epoch += 1
            ckpt.poll()
            if cfg.exit_after_step and step == cfg.exit_after_step:
                os._exit(137)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            stall = 0.0
            if step % cfg.ckpt_every == 0:
                stall = ckpt.save(step, model, opt, extra())
            if step % cfg.log_every == 0 or step == total:
                m = torch.tensor([loss_acc, acc_acc], device=device)
                if dist.is_initialized():
                    dist.all_reduce(m)
                    m /= world
                mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0], device=device)
                mems = [torch.zeros_like(mem) for _ in range(world)]
                if dist.is_initialized():
                    dist.all_gather(mems, mem)
                else:
                    mems = [mem]
                rec = {"loss": m[0].item(), "tok_acc": m[1].item(), "grad_norm": float(gn), "step_time": dt,
                       "samples_per_s": cfg.global_batch / dt, "lr": opt.param_groups[0]["lr"], "epoch": epoch,
                       "mfu": flops_s * cfg.global_batch / dt / (world * cfg.peak_tflops * 1e12),
                       "mem_alloc_gb_per_rank": [x.item() for x in mems], "ckpt_stall_s": stall}
                if step % cfg.val_every == 0 and len(val_ds):
                    rec.update(validate())
                log.log(step, rec)
                if rank == 0:
                    print(f"step {step}/{total} loss {rec['loss']:.4f} acc {rec['tok_acc']:.3f} "
                          f"{rec['samples_per_s']:.1f} samp/s mfu {rec['mfu']*100:.1f}%", flush=True)
            if preempt.should_stop(device):
                ckpt.save(step, model, opt, extra())
                ckpt.wait()
                if rank == 0:
                    print(f"preempted: saved step {step}, exiting", flush=True)
                stop = True
                break
        if not stop and in_epoch >= steps_per_epoch:
            epoch, in_epoch = epoch + 1, 0
    ckpt.wait()
    if not stop:
        if step % cfg.ckpt_every != 0:
            ckpt.save(step, model, opt, extra())
            ckpt.wait()
        if cfg.export_final:
            export_final(model, run / "final.pt")
    log.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
