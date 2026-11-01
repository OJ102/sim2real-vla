#!/usr/bin/env bash
# rsync code to the server. Usage: bash scripts/remote/sync.sh [host] [remote_dir]
# remote_dir is relative to the REMOTE home unless absolute (default: vla). Never use ~ here: it expands locally.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${1:-ucgpu}; DIR=${2:-vla}
# Explicit P (protect) rules: a dir-merge ':- .gitignore' does NOT protect receiver-side files from --delete.
rsync -az --delete --exclude .git --exclude .env --filter='P .env' --filter='P .env.gl' \
  --filter=':- .gitignore' ./ "$HOST:$DIR/"
