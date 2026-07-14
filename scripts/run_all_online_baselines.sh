#!/usr/bin/env bash
set -euo pipefail

# One-command formal pipeline. The full Frozen alignment gate must pass before
# Naive/ER/DERpp are allowed to run.
ROOT_PATH="${ROOT_PATH:-dataset}"
WEIGHT_DIR="${WEIGHT_DIR:-weights}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/online_baselines}"
DEVICE="${DEVICE:-cuda}"

python scripts/validate_baseline_alignment.py \
  --root_path "$ROOT_PATH" \
  --weight_dir "$WEIGHT_DIR" \
  --output_dir "$OUTPUT_DIR/alignment_ETTm1_H1" \
  --device "$DEVICE"

python scripts/eval_online_baselines.py \
  --root_path "$ROOT_PATH" \
  --weight_dir "$WEIGHT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --methods frozen naive er derpp \
  --allow_adaptive \
  "$@"
