#!/usr/bin/env bash
# Run a command in a detached tmux session, keeping its exit code and a per-run log.
# Usage: bash scripts/remote/launch.sh <name> <cmd...>
# Writes $VLA_RUNS/<name>/{run.sh,stdout.log}; stdout.log ends with "EXIT=<code> <timestamp>" (pipefail, PIPESTATUS).
# A missing EXIT line means the process tree was SIGKILLed (OOM-killer / cgroup kill): see the header's memory limits.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && set -a && . ./.env && set +a
NAME=${1:?run name}; shift
: "${VLA_RUNS:?set VLA_RUNS in .env}"
mkdir -p "$VLA_RUNS/$NAME"
LOG="$VLA_RUNS/$NAME/stdout.log"
{
  echo '#!/usr/bin/env bash'
  echo 'set -o pipefail'
  echo "cd $(printf %q "$PWD"); [ -f .env ] && set -a && . ./.env && set +a"
  echo "LOG=$(printf %q "$LOG")"
  echo 'echo "=== START $(date +%Y-%m-%dT%H:%M:%S%z) host=$(hostname) pid=$$" >> "$LOG"'
  echo 'CG=/sys/fs/cgroup$(grep "^0::" /proc/self/cgroup | cut -c4-)'
  echo 'echo "cgroup=$CG memory.max=$(cat $CG/memory.max 2>/dev/null) memory.high=$(cat $CG/memory.high 2>/dev/null) pids.max=$(cat $CG/pids.max 2>/dev/null)" >> "$LOG"'
  echo 'echo "ulimit: $(ulimit -l) memlock, $(ulimit -n) nofile; shm: $(df -h /dev/shm | tail -1)" >> "$LOG"'
  printf '%s' "$(printf '%q ' "$@")" > "$VLA_RUNS/$NAME/cmd.txt"
  echo "$(printf '%q ' "$@") 2>&1 | tee -a \"\$LOG\""
  echo 'rc=${PIPESTATUS[0]}'
  echo 'echo "EXIT=$rc $(date +%Y-%m-%dT%H:%M:%S%z)" >> "$LOG"'
} > "$VLA_RUNS/$NAME/run.sh"
tmux new -d -s "$NAME" "bash $VLA_RUNS/$NAME/run.sh"
echo "started tmux session '$NAME' (log: $LOG); attach: tmux attach -t $NAME; per-rank errors: $VLA_RUNS/<run>/rank*.err"
