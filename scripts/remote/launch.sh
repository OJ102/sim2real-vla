#!/usr/bin/env bash
# Run a command in a detached tmux session. Usage: bash scripts/remote/launch.sh <name> <cmd...>
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && set -a && . ./.env && set +a
NAME=${1:?run name}; shift
: "${VLA_RUNS:?set VLA_RUNS in .env}"
mkdir -p "$VLA_RUNS/$NAME"
tmux new -d -s "$NAME" "cd $PWD && $* 2>&1 | tee -a $VLA_RUNS/$NAME/stdout.log"
echo "started tmux session '$NAME' (logs: $VLA_RUNS/$NAME/stdout.log); attach: tmux attach -t $NAME"
