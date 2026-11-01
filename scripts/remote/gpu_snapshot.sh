#!/usr/bin/env bash
# Usage: bash scripts/remote/gpu_snapshot.sh [outfile]   (default results/gpu_snapshot.csv)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-results/gpu_snapshot.csv}
mkdir -p "$(dirname "$OUT")"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv > "$OUT"
echo "wrote $OUT"; cat "$OUT"
