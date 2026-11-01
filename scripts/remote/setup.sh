#!/usr/bin/env bash
# Server setup. Runs fully in user space (no root, no system package manager). Usage: bash scripts/remote/setup.sh [--osmesa]
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f .env ] && set -a && . ./.env && set +a   # CUDA_VISIBLE_DEVICES etc.

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv python install 3.11
uv sync || { echo "ERROR: uv sync failed"; exit 1; }
# deepspeed is optional (sdist, needs torch at build time); never fatal
uv pip install --no-build-isolation deepspeed || echo "WARN: deepspeed install failed; DeepSpeed backend unavailable"

nvidia-smi --query-gpu=name,driver_version --format=csv || echo "WARN: nvidia-smi unavailable"
# flash-attn: prebuilt wheel only (no source build under nvcc 13.2). Fallback: torch SDPA flash backend.
uv pip install --only-binary=:all: flash-attn 2>/dev/null \
  || echo "WARN: no prebuilt flash-attn wheel matches this torch/CUDA; using SDPA flash backend as fallback"

# Headless GL: prefer the NVIDIA driver's libEGL; fall back to OSMesa from conda-forge (user space).
GL=egl
if [[ "${1:-}" == "--osmesa" ]] || ! ldconfig -p 2>/dev/null | grep -qi 'libEGL_nvidia'; then
  echo "libEGL_nvidia not found (or --osmesa): using OSMesa via conda-forge mesalib (user space)"
  if ! command -v micromamba >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/micromamba" ]; then
    mkdir -p "$HOME/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
  fi
  MM=$(command -v micromamba || echo "$HOME/.local/bin/micromamba")
  "$MM" create -y -p "$HOME/.vla_mesa" -c conda-forge mesalib libglu
  export LD_LIBRARY_PATH="$HOME/.vla_mesa/lib:${LD_LIBRARY_PATH:-}"
  GL=osmesa
  echo "export LD_LIBRARY_PATH=$HOME/.vla_mesa/lib:\$LD_LIBRARY_PATH; export MUJOCO_GL=osmesa  # add to your shell rc" > .env.gl
fi
export MUJOCO_GL=$GL
echo "MUJOCO_GL=$GL"

# Fetch the Panda assets once (workers must not race to clone menagerie).
uv run python -c "from robot_descriptions import panda_mj_description" >/dev/null 2>&1

uv run python - <<'PY'
import os, sys, torch
print("torch", torch.__version__, "torch.version.cuda", torch.version.cuda)
n = torch.cuda.device_count()
print(f"cuda devices: {n}")
bad = False
for i in range(n):
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i} {name} free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB")
    bad |= "A100" not in name
if n == 0 or bad:
    print("ERROR: no devices or a visible device is not an A100 (check CUDA_VISIBLE_DEVICES in .env)")
    sys.exit(1)
import subprocess
code = ("from vla.sim.env import TabletopEnv; e=TabletopEnv('lift','train'); "
        "print(e.reset(seed=0)[0]['image'].shape)")
ok = []
for dev in range(max(n, 5)):  # EGL ids may not match CUDA order; probe each in a subprocess
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_EGL_DEVICE_ID": str(dev)})
    shape = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
    print(f"  MUJOCO_EGL_DEVICE_ID={dev}: " + (f"egl render ok {tuple(eval(shape))}" if shape else "failed"))
    if shape: ok.append(dev)
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} working EGL device ids: {ok}")
if not ok:
    sys.exit("ERROR: no EGL device could render")
PY
