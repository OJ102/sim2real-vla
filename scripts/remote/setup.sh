#!/usr/bin/env bash
# Server setup. Runs fully in user space (no root, no system package manager). Usage: bash scripts/remote/setup.sh [--osmesa]
set -euo pipefail
cd "$(dirname "$0")/../.."

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv python install 3.11
uv sync --extra gpu || { echo "WARN: gpu extra failed (flash-attn?); retrying without it"; uv sync; }

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
import torch
print(f"cuda devices: {torch.cuda.device_count()}")
from vla.sim.env import TabletopEnv
env = TabletopEnv(task="lift", split="train")
obs, _ = env.reset(seed=0)
print(f"egl render ok {obs['image'].shape}")
PY
