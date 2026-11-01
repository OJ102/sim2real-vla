"""Reproducible micro-benchmark of the BC training step (synthetic batches, no dataloader, no checkpoints).

Worker (under torchrun):
  torchrun --nproc_per_node=4 scripts/bench_bc_step.py --worker --micro-batch 32 --accum 2 [--no-reshard] [--prefetch]
           [--compile] [--no-fused] [--act-ckpt] [--steps 30] [--warmup 10] [--model-config configs/model_0p5b.yaml]
Driver (runs the cumulative sweep, one torchrun per config, prints a before/after table):
  uv run python scripts/bench_bc_step.py --sweep --gpus 0,1,2,4 [--quiet-wait] [--out results/bench_bc_step.json]
Each config reports samples/s (global), MFU, median compute_s (step time), peak process-local memory.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

CONFIGS = [  # cumulative: each row adds one change on top of the previous (lead's order)
    ("baseline (reshard, no prefetch, mb32x2, eager, fused)", {}),
    ("+ reshard_after_forward=false", {"no_reshard": True}),
    ("+ prefetch", {"prefetch": True}),
    ("+ micro_batch 64, accum 1", {"micro_batch": 64, "accum": 1}),
    ("+ torch.compile per block", {"compile": True}),
    ("(ablation) fused AdamW off", {"no_fused": True}),
]


def worker(a):
    import torch
    import torch.distributed as dist

    from vla.model.prompts import encode_instructions
    from vla.model.vla import VLA, VLAConfig, count_params
    from vla.train.bc import flops_per_sample
    from vla.train.dist import init_dist, unwrap, wrap_fsdp2
    from vla.utils.config import load_config

    rank, world, local, device = init_dist()
    cuda = device.type == "cuda"
    torch.set_float32_matmul_precision("high")
    mcfg = load_config(VLAConfig, a.model_config, [f"head={a.head}"])
    torch.manual_seed(0)
    model = VLA(mcfg)
    if cuda:
        model.to(device)
    model.train()
    model = wrap_fsdp2(model, bf16=cuda, reshard_after_forward=not a.no_reshard, prefetch=a.prefetch,
                       act_ckpt=a.act_ckpt, compile_layers=a.compile) if world > 1 or cuda else model
    base = unwrap(model)
    attn = (base.llm.config._attn_implementation, base.vision.config._attn_implementation)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=bool(cuda and not a.no_fused))
    tok = base.tokenizer
    B = a.micro_batch
    batch = {"image": torch.randint(0, 255, (B, 128, 128, 3), dtype=torch.uint8, device=device),
             "proprio": torch.randn(B, 11, device=device), "actions": torch.rand(B, 4, 4, device=device) * 2 - 1,
             **{k: v.to(device) for k, v in encode_instructions(tok, ["lift the red cube"] * B).items()}}
    S = (mcfg.image_size // 16) ** 2 + 1 + batch["input_ids"].shape[1] + base.n_act_tokens
    fps = flops_per_sample(base, S)
    times = []
    for it in range(a.steps):
        t0 = time.perf_counter()
        for _ in range(a.accum):
            out = model(batch)
            (out["loss"] / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = statistics.median(times[a.warmup:])
    gb = B * a.accum * world
    mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0], device=device)
    if dist.is_initialized():
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        res = {"samples_per_s": gb / t, "mfu": fps * gb / t / (world * 312e12), "compute_s": t,
               "step_min_s": min(times[a.warmup:]), "step_max_s": max(times[a.warmup:]), "peak_mem_gb": mem.item(),
               "global_batch": gb, "world": world, "attn_impl": attn, "params_M": count_params(base)["total"] / 1e6,
               "flags": {k: getattr(a, k) for k in ("micro_batch", "accum", "no_reshard", "prefetch", "compile",
                                                      "no_fused", "act_ckpt")}}
        print("BENCH " + json.dumps(res), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


def driver(a):
    from vla.utils.gpu import quiet_wait, snapshot

    gpus = a.gpus.split(",")
    rows, port = [], 29900
    for name, delta in CONFIGS if not a.only else [c for c in CONFIGS if a.only in c[0]]:
        cum = {}
        for n2, d2 in CONFIGS:  # cumulative flags up to and including this row (ablation row = previous + its flag)
            cum.update(d2)
            if n2 == name:
                break
        if name.startswith("(ablation)"):
            cum["compile"] = a.ablation_keeps_compile and cum.get("compile", False)
        flags = ["--worker", f"--model-config={a.model_config}", f"--head={a.head}", f"--steps={a.steps}",
                 f"--warmup={a.warmup}", f"--micro-batch={cum.get('micro_batch', 32)}", f"--accum={cum.get('accum', 2)}"]
        for k in ("no_reshard", "prefetch", "compile", "no_fused", "act_ckpt"):
            if cum.get(k):
                flags.append("--" + k.replace("_", "-"))
        port += 1
        rec = {"config": name}
        if a.quiet_wait:
            rec["quiet_wait"] = quiet_wait([int(g) for g in gpus[: a.n_gpu]], 15, 60, 1200)
        rec["gpu_snapshot_before"] = snapshot(gpus=[int(g) for g in gpus[: a.n_gpu]])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus[: a.n_gpu]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        cmd = ["torchrun", "--nnodes=1", f"--nproc_per_node={a.n_gpu}", "--rdzv_backend=c10d",
               f"--rdzv_endpoint=localhost:{port}", __file__, *flags]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("BENCH ")]
        rec["gpu_snapshot_after"] = snapshot(gpus=[int(g) for g in gpus[: a.n_gpu]])
        rec["contended"] = bool(rec["gpu_snapshot_before"].get("contended") or rec["gpu_snapshot_after"].get("contended"))
        if line:
            rec.update(json.loads(line[0][6:]))
            print(f"{name:58s} {rec['samples_per_s']:7.1f} samp/s  MFU {rec['mfu']*100:5.1f}%  step {rec['compute_s']:.2f}s  "
                  f"{rec['peak_mem_gb']:.1f} GB  contended={rec['contended']}", flush=True)
        else:
            rec["error"] = (r.stderr or "")[-800:]
            print(f"{name}: FAILED\n{rec['error']}", flush=True)
        rows.append(rec)
        if a.out:
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--model-config", default="configs/model_0p5b.yaml")
    ap.add_argument("--head", default="discrete")
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    for f in ("no-reshard", "prefetch", "compile", "no-fused", "act-ckpt"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--gpus", default="0,1,2,4")
    ap.add_argument("--n-gpu", type=int, default=4)
    ap.add_argument("--quiet-wait", action="store_true")
    ap.add_argument("--only", default="", help="substring: run only matching sweep rows")
    ap.add_argument("--ablation-keeps-compile", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    (worker if a.worker else driver)(a)
