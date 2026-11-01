"""P4b profiler traces: 3 traced FSDP2 steps (after 5 warmup) on N GPUs, prefetch off vs on; rank-0 Chrome traces are
gzipped into results/traces/ (must stay < 20 MB each).

    uv run python scripts/profile_step.py --n-gpu 4 --gpus 0,1,2,4 [--out-dir results/traces]
Open the .json.gz in https://ui.perfetto.dev or chrome://tracing; look for all-gather / reduce-scatter kernels (NCCL)
overlapping compute streams. nsys is not used (needs privileges); torch.profiler only."""
import argparse
import gzip
import os
import shutil
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--n-gpu", type=int, default=4)
ap.add_argument("--gpus", default="0,1,2,4")
ap.add_argument("--out-dir", default="results/traces")
ap.add_argument("--runs-dir", default=os.environ.get("VLA_RUNS", "runs"))
ap.add_argument("--data-dir", default=str(Path(os.environ.get("VLA_DATA", "")) / "demos"))
ap.add_argument("extra", nargs="*")
a = ap.parse_args()
gpus = a.gpus.split(",")[: a.n_gpu]
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
for prefetch in (False, True):
    name = f"profile_prefetch{int(prefetch)}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    cmd = ["torchrun", "--nnodes=1", f"--nproc_per_node={a.n_gpu}", "--rdzv_backend=c10d",
           f"--rdzv_endpoint=localhost:{29800 + int(prefetch)}", "-m", "vla.train.bc", "--config", "configs/bc.yaml",
           f"name={name}", f"runs_dir={a.runs_dir}", f"data_dir={a.data_dir}", "profile=true",
           f"prefetch={str(prefetch).lower()}", "steps=20", "ckpt_every=100000", "export_final=false", *a.extra]
    subprocess.run(cmd, env=env, check=True)
    src = Path(a.runs_dir) / name / "trace_rank0.json"
    dst = Path(a.out_dir) / f"fsdp2_{a.n_gpu}gpu_prefetch{int(prefetch)}.json.gz"
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=9) as g:
        shutil.copyfileobj(f, g)
    mb = dst.stat().st_size / 2**20
    print(f"{dst} {mb:.1f} MB" + ("  WARNING: > 20 MB, reduce active steps or trim" if mb > 20 else ""))
