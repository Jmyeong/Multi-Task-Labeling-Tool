#!/usr/bin/env bash
set -euo pipefail
MT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$MT_ROOT"
exec "${PYTHON:-python3}" -m gui.server \
  --dataset-root "${INPUT:-$MT_ROOT/dataset}" \
  --output-root "${OUTPUT_DIR:-$MT_ROOT/outputs/review}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8766}" "$@"
