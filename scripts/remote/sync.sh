#!/usr/bin/env bash
# rsync fallback when git pull is not possible. Usage: bash scripts/remote/sync.sh user@host [remote_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${1:?user@host}; DIR=${2:-~/vla}
rsync -az --delete --exclude .git --exclude .venv --exclude data --exclude runs --exclude '__pycache__' \
  --exclude results/demo --exclude .env ./ "$HOST:$DIR/"
