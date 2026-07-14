#!/usr/bin/env bash
set -euo pipefail

# Usage on the 5090 server:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_binary_gate_test_matrix.sh preflight
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_binary_gate_test_matrix.sh all \
#       > logs/binary_gate_test_matrix.log 2>&1 &

MODE="${1:-preflight}"
shift || true

python -u scripts/run_binary_gate_test_matrix.py --mode "${MODE}" "$@"
