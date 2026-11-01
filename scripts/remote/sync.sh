#!/usr/bin/env bash
# rsync code to the server. Usage: bash scripts/remote/sync.sh [host] [remote_dir]
# remote_dir is relative to the REMOTE home unless absolute (default: vla). Never use ~ here: it expands locally.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${1:-ucgpu}; DIR=${2:-vla}
# --delete does not remove excluded files on the server (.env, .venv, data stay)
rsync -az --delete --exclude .git --filter=':- .gitignore' ./ "$HOST:$DIR/"
