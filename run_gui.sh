#!/usr/bin/env bash
set -euo pipefail
MT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$MT_ROOT/run.py" gui "$@"
