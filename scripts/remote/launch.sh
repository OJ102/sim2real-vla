#!/usr/bin/env bash
# Run a command in a detached tmux session. Usage: bash scripts/remote/launch.sh <name> <cmd...>
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && set -a && . ./.env && set +a
NAME=${1:?run name}; shift
: "${VLA_RUNS:?set VLA_RUNS in .env}"
mkdir -p "$VLA_RUNS/$NAME"
# tmux sessions inherit the tmux SERVER env, not ours: source .env inside the session (UV_* vars, CUDA_VISIBLE_DEVICES)
tmux new -d -s "$NAME" "cd $PWD && set -a && . ./.env && set +a && $* 2>&1 | tee -a $VLA_RUNS/$NAME/stdout.log"
echo "started tmux session '$NAME' (logs: $VLA_RUNS/$NAME/stdout.log); attach: tmux attach -t $NAME"
