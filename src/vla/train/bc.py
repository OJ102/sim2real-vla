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
import sys
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
from vla.utils.logging import MetricsLogger, host_mem


@dataclass
class BCConfig:
    name: str = "bc"
    model_config: str = "configs/model_0p5b.yaml"
    head: str = ""                      # override model head: discrete | flow
    data_dir: str = ""                  # default $VLA_DATA/demos
    subset: str = ""                    # optional subset_{pct}.json
    runs_dir: str = ""                  # default $VLA_RUNS
    backend: str = "fsdp"               # fsdp | ddp | deepspeed (ZeRO-3; throughput sweep only, no checkpoints)
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
    compile_layers: bool = False        # torch.compile each transformer block before fully_shard (FSDP2 only)
    fused_adam: bool = True             # fused AdamW kernels (CUDA only)
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
    pg_timeout_min: float = 10.0        # explicit timeout on every process group (hangs raise instead of sitting)
    hang_dump_s: float = 0.0            # >0: dump all thread stacks to stdout if a step takes longer (also SIGUSR2)
    ckpt_slow_s: float = 0.0            # test aid: slow DCP writer (reproduces async-overlap bugs on CPU)
    prefetch_factor: int = 4
    pin_memory: bool = True
    profile: bool = False               # torch.profiler: 5 warmup + 3 traced steps -> <run>/trace_rank<r>.json, then exit
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
    rank, world, local, device = init_dist(cfg.pg_timeout_min)
    import faulthandler
    import signal as _sig

    run0 = Path(cfg.runs_dir or os.environ.get("VLA_RUNS", "runs")) / cfg.name
    run0.mkdir(parents=True, exist_ok=True)
    err = open(run0 / f"rank{rank}.err", "a", buffering=1)  # survives even if the launcher's tee/tmux dies
    faulthandler.enable(file=err, all_threads=True)          # segfault / fatal-signal stacks
    faulthandler.register(_sig.SIGUSR2, all_threads=True, file=err)  # kill -USR2 <pid>: dump every thread stack
    _sig.signal(_sig.SIGPIPE, _sig.SIG_IGN)  # dead tee/pipe must not kill training silently

    def say(msg: str, **_kw) -> None:
        """stdout (rank-tagged, tolerant of a dead pipe) + per-rank err file."""
        try:
            err.write(f"[rank {rank}] {msg}\n")
        except OSError:
            pass
        if rank == 0 or msg.startswith("!!!"):
            try:
                print(msg, flush=True)
            except OSError:
                pass

    err.write(f"[rank {rank}] pid {os.getpid()} start {time.ctime()} host_mem {host_mem()}\n")
    cuda = device.type == "cuda"
    torch.set_float32_matmul_precision("high")
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
        model = wrap_fsdp2(model, cfg.bf16, cfg.reshard_after_forward, cfg.prefetch, cfg.act_ckpt,
                           compile_layers=cfg.compile_layers)
    elif cfg.backend == "ddp":
        model = wrap_ddp(model, local, cfg.act_ckpt)
    elif cfg.backend == "deepspeed":
        for m_ in (model.llm, model.vision):
            if cfg.act_ckpt:
                m_.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        raise ValueError(cfg.backend)
    base = unwrap(model)

    vis = [p for n, p in model.named_parameters() if "vision." in n]
    rest = [p for n, p in model.named_parameters() if "vision." not in n]
    opt = torch.optim.AdamW([{"params": rest, "lr": cfg.lr}, {"params": vis, "lr": cfg.lr_vision}],
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
                            fused=bool(cuda and cfg.fused_adam))

    engine = None
    if cfg.backend == "deepspeed":
        import deepspeed

        assert cfg.global_batch % (cfg.micro_batch * world) == 0
        ds_cfg = {"train_micro_batch_size_per_gpu": cfg.micro_batch,
                  "gradient_accumulation_steps": cfg.global_batch // (cfg.micro_batch * world),
                  "bf16": {"enabled": cfg.bf16}, "gradient_clipping": cfg.grad_clip,
                  "zero_optimization": {"stage": 3, "overlap_comm": True, "contiguous_gradients": True},
                  "zero_allow_untested_optimizer": True, "steps_per_print": 10**9}
        engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_cfg)
        model = engine
        base = engine.module
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
    # checkpoints (8 GB each) go to $VLA_CKPT/<name> (RAM-backed on the server) when set; only final.pt/metrics on home
    ckpt_dir = cfg.ckpt_dir or (str(Path(os.environ["VLA_CKPT"]) / cfg.name / "ckpt") if os.environ.get("VLA_CKPT")
                                else str(run / "ckpt"))
    ckpt = Checkpointer(ckpt_dir, cfg.ckpt_mode, cfg.ckpt_keep, cfg.ckpt_mirror or None,
                        cfg.pg_timeout_min, cfg.ckpt_slow_s, log=say)
    say(f"attn_implementation: llm={base.llm.config._attn_implementation} "
        f"vision={base.vision.config._attn_implementation}")
    if rank == 0:
        say(f"world={world} backend={cfg.backend} frames={len(ds)} steps/epoch={steps_per_epoch} total={total} "
              f"micro={cfg.micro_batch} accum={accum} global={cfg.global_batch} params={count_params(base)['total']/1e6:.1f}M",
              flush=True)

    step, epoch, in_epoch = 0, 0, 0
    if a.resume:
        ex = ckpt.load_latest(model, opt, {"step": 0, "epoch": 0, "in_epoch": 0, "sched": sched.state_dict()})
        if ex is not None:
            step, epoch, in_epoch = ex["step"], ex["epoch"], ex["in_epoch"]
            sched.load_state_dict(ex["sched"])
            if rank == 0:
                say(f"resume_from={step}", flush=True)
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

    prof = None
    if cfg.profile:
        from torch.profiler import ProfilerActivity, profile, schedule

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        prof = profile(activities=acts, schedule=schedule(wait=0, warmup=5, active=3),
                       on_trace_ready=lambda p: p.export_chrome_trace(str(run / f"trace_rank{rank}.json")))
        prof.start()
    flops_s = None
    stop = False
    t_prev = time.perf_counter()
    while step < total and not stop:
        sampler.set_epoch(epoch)
        sampler.consumed = in_epoch * per_rank_step
        dl = DataLoader(ds, batch_size=cfg.micro_batch, sampler=sampler, collate_fn=collate,
                        num_workers=cfg.num_workers, drop_last=True, persistent_workers=False,
                        pin_memory=cuda and cfg.pin_memory,
                        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None)
        it = iter(dl)
        while in_epoch < steps_per_epoch and step < total:
            t0 = time.perf_counter()
            torch.manual_seed(cfg.seed * 1_000_003 + step * world + rank)  # deterministic per step, resume-safe
            if cfg.hang_dump_s:
                import faulthandler

                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(cfg.hang_dump_s, repeat=False, file=sys.stdout)
            loss_acc = torch.zeros((), device=device)
            acc_acc = torch.zeros((), device=device)
            data_wait = 0.0
            for mb in range(accum):
                t_d = time.perf_counter()
                b = to_dev(next(it))
                data_wait += time.perf_counter() - t_d
                ctx = model.no_sync() if (cfg.backend == "ddp" and mb < accum - 1) else contextlib.nullcontext()
                if engine is not None:  # DeepSpeed: engine owns scaling/accumulation/clipping/optimizer step
                    out = model(b)
                    model.backward(out["loss"])
                    model.step()
                    loss_acc += out["loss"].detach() / accum
                    continue
                with ctx, (torch.autocast("cuda", dtype=torch.bfloat16) if (cuda and cfg.backend == "ddp" and cfg.bf16)
                           else contextlib.nullcontext()):
                    out = model(b)
                    (out["loss"] / accum).backward()
                loss_acc += out["loss"].detach() / accum  # tensors: no host sync per micro-batch
                acc_acc += out.get("tok_acc", torch.zeros((), device=device)).detach() / accum
            if flops_s is None:
                n_img = (mcfg.image_size // 16) ** 2 if not mcfg.tiny else (mcfg.image_size // 16) ** 2
                flops_s = flops_per_sample(base, n_img + 1 + b["input_ids"].shape[1] + base.n_act_tokens)
            if engine is not None:
                gn = engine.get_global_grad_norm() or 0.0
            else:
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
            if step % cfg.ckpt_every == 0 and engine is None:
                err.write(f"[rank {rank}] step {step} save start host_mem {host_mem()}\n")
                stall = ckpt.save(step, model, opt, extra())
                err.write(f"[rank {rank}] step {step} save staged ({stall:.2f}s) host_mem {host_mem()}\n")
            if step % cfg.log_every == 0 or step == total:
                m = torch.stack([loss_acc, acc_acc])
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
                       "mem_alloc_gb_per_rank": [x.item() for x in mems], "ckpt_stall_s": stall,
                       "data_wait_s": data_wait, "compute_s": dt - data_wait, **host_mem()}
                if step % cfg.val_every == 0 and len(val_ds):
                    rec.update(validate())
                log.log(step, rec)
                if rank == 0:
                    say(f"step {step}/{total} loss {rec['loss']:.4f} acc {rec['tok_acc']:.3f} "
                          f"{rec['samples_per_s']:.1f} samp/s mfu {rec['mfu']*100:.1f}%", flush=True)
            if prof is not None:
                prof.step()
                if step >= 8:  # 5 warmup + 3 active
                    prof.stop()
                    if rank == 0:
                        say(f"profile written: {run}/trace_rank0.json", flush=True)
                    stop = True
                    break
            if preempt.should_stop(device):
                ckpt.save(step, model, opt, extra())
                ckpt.wait()
                if rank == 0:
                    say(f"preempted: saved step {step}, exiting", flush=True)
                stop = True
                break
        if not stop and in_epoch >= steps_per_epoch:
            epoch, in_epoch = epoch + 1, 0
    ckpt.wait()
    if not stop:
        if step % cfg.ckpt_every != 0 and engine is None:
            ckpt.save(step, model, opt, extra())
            ckpt.wait()
        if cfg.export_final and engine is None:
            try:
                export_final(model, run / "final.pt")
            except (OSError, RuntimeError) as e:  # home quota full: keep the weights somewhere that works
                alt = Path(ckpt_dir).parent / "final.pt"
                say(f"!!! final.pt export to {run} failed ({e}); retrying at {alt}")
                export_final(model, alt)
    log.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
