#!/usr/bin/env bash
# Download DTD (Describable Textures Dataset) to $VLA_DATA/dtd/<class>/*.jpg  (default $HOME/.cache/vla/dtd).
# Primary: Oxford tarball. Fallback (host unreachable from some networks): HF mirror tanganke/dtd (test split).
# Usage: bash scripts/fetch_dtd.sh [images_per_class_for_fallback=4]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
DEST=${VLA_DATA:-$HOME/.cache/vla}/dtd
N=${1:-4}
if [ -d "$DEST" ] && [ "$(ls "$DEST" | wc -l)" -ge 47 ]; then echo "DTD already at $DEST"; exit 0; fi
mkdir -p "$DEST"
TMP=$(mktemp -d "${VLA_DATA:-$HOME/.cache/vla}/dtd_dl.XXXXXX")   # not /tmp: may be full on the server
trap 'rm -rf "$TMP"' EXIT
if curl -fsS -m 30 -I https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz >/dev/null 2>&1; then
  curl -fL https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -o "$TMP/dtd.tgz"
  tar -xzf "$TMP/dtd.tgz" -C "$TMP"
  cp -r "$TMP"/dtd/images/* "$DEST"/
else
  echo "Oxford host unreachable; using HF mirror tanganke/dtd (test split, $N images/class)"
  curl -fL https://huggingface.co/datasets/tanganke/dtd/resolve/main/data/test-00000-of-00001.parquet -o "$TMP/dtd.parquet"
  uv run --with pyarrow python scripts/_dtd_from_parquet.py "$TMP/dtd.parquet" "$DEST" "$N"
fi
echo "DTD classes: $(ls "$DEST" | wc -l) at $DEST"
