#!/bin/bash
set -euo pipefail

# Backward-compatible alias. The old name sounds like a parameter sweep, but the
# script is really the main 12-case run for ECL/Traffic/ETTh1.
exec bash scripts/run_prompt_z_main_12cases.sh "$@"
