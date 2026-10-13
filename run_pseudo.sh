#!/usr/bin/env bash
set -euo pipefail
MT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$MT_ROOT"
exec "${PYTHON:-python3}" -m scripts.generate_pseudo_labels \
  --dataset-root "${INPUT:-$MT_ROOT/dataset}" \
  --output-root "${OUTPUT_DIR:-$MT_ROOT/outputs/pseudo_labels}" \
  --checkpoint "${CHECKPOINT:-$MT_ROOT/pseudo_labeling/checkpoint/best_joint.pt}" \
  --model-template "${MODEL_TEMPLATE:-$MT_ROOT/pseudo_labeling/checkpoint/yolo26s.pt}" \
  --device "${DEVICE:-cpu}" "$@"
