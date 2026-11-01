#!/usr/bin/env bash
# Server setup. Runs fully in user space (no root, no system package manager). Usage: bash scripts/remote/setup.sh [--osmesa]
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f .env ] && set -a && . ./.env && set +a   # CUDA_VISIBLE_DEVICES etc.

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv python install 3.11
uv sync || { echo "ERROR: uv sync failed"; exit 1; }
# uv pip ignores UV_PROJECT_ENVIRONMENT, so point it at the project venv explicitly
PY="${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python"; [ -x "$PY" ] || PY=.venv/bin/python
# deepspeed is optional (sdist, needs torch at build time); never fatal
DS_BUILD_OPS=0 uv pip install --python "$PY" --no-build-isolation deepspeed || echo "WARN: deepspeed install failed; DeepSpeed backend unavailable"

nvidia-smi --query-gpu=name,driver_version --format=csv || echo "WARN: nvidia-smi unavailable"
# flash-attn: prebuilt wheel only (no source build under nvcc 13.2). Fallback: torch SDPA flash backend.
uv pip install --python "$PY" --only-binary=:all: flash-attn 2>/dev/null \
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
from vla.sim.egl import map_devices
try:
    rows = map_devices()
    n_egl = len(rows)
except Exception as e:  # mapping failed: still probe a generous range
    print(f"EGL->GPU mapping unavailable ({e}); probing ids 0..15")
    rows, n_egl = [], 16
# every EGL device is rendered in its own subprocess; GL_RENDERER must contain NVIDIA (llvmpipe etc. = failed)
code = ("from vla.sim.env import TabletopEnv; from vla.sim.egl import gl_renderer\n"
        "e=TabletopEnv('lift','train',allow_missing_dtd=True)\n"  # probe tests GL only; DTD is checked separately
        "s=e.reset(seed=0)[0]['image'].shape; print(s, '|', gl_renderer())")
ok = []
for dev in range(n_egl):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_EGL_DEVICE_ID": str(dev)})
    last = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    gpu = rows[dev]["gpu_name"] if dev < len(rows) else None
    if "|" in last:
        shape, renderer = (x.strip() for x in last.split("|", 1))
        good = "NVIDIA" in renderer
        print(f"  EGL id {dev} [{gpu}]: render {shape} GL_RENDERER={renderer!r} -> {'OK' if good else 'REJECTED (not NVIDIA)'}")
        if good:
            ok.append(dev)
    else:
        err = (r.stderr.strip().splitlines() or ["?"])[-1][:120]
        print(f"  EGL id {dev} [{gpu}]: FAILED ({err})")
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} working NVIDIA EGL ids: {ok}")
a100 = [r["egl_id"] for r in rows if r["gpu_name"] and "A100" in r["gpu_name"]]
rec = [d for d in a100 if d in ok] if rows else ok
print(f"recommended VLA_EGL_DEVICES={','.join(map(str, rec))}  (A100 + NVIDIA renderer; put in .env, or leave unset to auto-pick)")
if not rec:
    sys.exit("ERROR: no usable NVIDIA EGL device on an A100")
PY
