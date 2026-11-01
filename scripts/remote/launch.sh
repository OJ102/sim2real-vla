#!/usr/bin/env bash
# Run a command in a detached tmux session, keeping its exit code and a per-run log.
# Usage: bash scripts/remote/launch.sh <name> <cmd...>
# Writes $VLA_RUNS/<name>/stdout.log and a copy $VLA_LOGS/<name>.log (default /dev/shm/$USER_vla_logs; survives a full home); stdout.log ends with "EXIT=<code> <timestamp>" (pipefail, PIPESTATUS).
# A missing EXIT line means the process tree was SIGKILLed (OOM-killer / cgroup kill): see the header's memory limits.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && set -a && . ./.env && set +a
NAME=${1:?run name}; shift
: "${VLA_RUNS:?set VLA_RUNS in .env}"
mkdir -p "$VLA_RUNS/$NAME" || echo "WARN: cannot create $VLA_RUNS/$NAME (home quota?); logging to /dev/shm copy only"
LOG="$VLA_RUNS/$NAME/stdout.log"
LOG2="${VLA_LOGS:-/dev/shm/${USER:-user}_vla_logs}/$NAME.log"   # copy that survives a full home quota
mkdir -p "$(dirname "$LOG2")"
RUNSH="$(dirname "$LOG2")/$NAME.run.sh"   # not on home: launching must work even when the quota is full
{
  echo '#!/usr/bin/env bash'
  echo 'set -o pipefail'
  echo "cd $(printf %q "$PWD"); [ -f .env ] && set -a && . ./.env && set +a"
  echo "LOG=$(printf %q "$LOG"); LOG2=$(printf %q "$LOG2")"
  echo 'TEE="tee -a"; tee --output-error=warn </dev/null >/dev/null 2>&1 && TEE="tee --output-error=warn -a"  # keep going if one file fails'
  echo 'echo "=== START $(date +%Y-%m-%dT%H:%M:%S%z) host=$(hostname) pid=$$" | $TEE "$LOG" "$LOG2" >/dev/null'

  echo 'CG=/sys/fs/cgroup$(grep "^0::" /proc/self/cgroup | cut -c4-)'
  echo 'echo "cgroup=$CG memory.max=$(cat $CG/memory.max 2>/dev/null) memory.high=$(cat $CG/memory.high 2>/dev/null) pids.max=$(cat $CG/pids.max 2>/dev/null)" | $TEE "$LOG" "$LOG2" >/dev/null'
  echo 'echo "ulimit: $(ulimit -l) memlock, $(ulimit -n) nofile; shm: $(df -h /dev/shm | tail -1)" | $TEE "$LOG" "$LOG2" >/dev/null'
  printf '%s' "$(printf '%q ' "$@")" > "$(dirname "$LOG2")/$NAME.cmd.txt"
  echo "$(printf '%q ' "$@") 2>&1 | \$TEE \"\$LOG\" \"\$LOG2\""
  echo 'rc=${PIPESTATUS[0]}'
  echo 'echo "EXIT=$rc $(date +%Y-%m-%dT%H:%M:%S%z)" | $TEE "$LOG" "$LOG2" >/dev/null'
} > "$RUNSH"
tmux new -d -s "$NAME" "bash $RUNSH"
echo "started tmux session '$NAME' (log: $LOG); attach: tmux attach -t $NAME; per-rank errors: $VLA_RUNS/<run>/rank*.err"
