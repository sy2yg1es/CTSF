#!/bin/bash
set -euo pipefail

# Backward-compatible alias. The old name hides that this is the second main
# bundle: 16 cases covering WTH/ETTm1/ETTm2/ETTh2.
exec bash scripts/run_prompt_z_main_16cases.sh "$@"
