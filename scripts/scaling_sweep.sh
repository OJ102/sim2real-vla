#!/usr/bin/env bash
# Wrapper: sources .env then runs the sweep. Usage: bash scripts/scaling_sweep.sh [scaling_sweep.py args]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
exec uv run python scripts/scaling_sweep.py "$@"
