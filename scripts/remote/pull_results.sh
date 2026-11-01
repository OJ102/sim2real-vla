#!/usr/bin/env bash
# Pull small artifacts (results/, metrics.jsonl) from the server. Never pulls checkpoints.
# Usage: bash scripts/remote/pull_results.sh user@host [remote_dir] [remote_runs_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${1:?user@host}; DIR=${2:-~/vla}; RUNS=${3:-$(ssh "$HOST" "cd $DIR && . ./.env && echo \$VLA_RUNS")}
rsync -az "$HOST:$DIR/results/" results/
mkdir -p runs_pulled
rsync -az --include='*/' --include='metrics.jsonl' --include='config.yaml' --exclude='*' "$HOST:$RUNS/" runs_pulled/
