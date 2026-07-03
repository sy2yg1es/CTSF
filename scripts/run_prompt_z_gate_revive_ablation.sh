#!/bin/bash
set -euo pipefail

# ==============================================================================
# Prompt-Z gate/mask revival ablation.
#
# B: A + long mask_floor
# A: weak regularization, no mask floor
# C: A + no no-op penalty
#
# Override CASES to run a different subset.
# ==============================================================================

CASES=${CASES:-"Traffic.csv:1:862 ECL.csv:96:321 ETTh2.csv:96:7 WTH.csv:96:12"}
BASE_TAG=${BASE_TAG:-gate_revive}

COMMON_ENV=(
    "CASES=${CASES}"
    "DELAYED_RESIDUAL_TRAINING=1"
    "ENABLE_VALIDATION_FALLBACK=1"
    "MAX_DELTA_RATIO=0.05"
    "LAMBDA_DELTA=0.0002"
    "LAMBDA_MASK=0.0001"
    "TARGET_MASK_RATIO=0.10"
    "GAMMA_FLOOR=0.1"
    "GAMMA_FLOOR_STEPS=8000"
    "NOOP_WARMUP_STEPS=14000"
    "NOOP_RAMP_STEPS=2000"
)

run_variant() {
    local name="$1"
    shift

    echo "================================================================="
    echo " Prompt-Z Gate Revival: ${name}"
    echo "================================================================="

    env \
        "${COMMON_ENV[@]}" \
        "RUN_TAG=${BASE_TAG}_${name}" \
        "$@" \
        bash scripts/run_prompt_z_complete.sh
}

run_variant "B_maskfloor" \
    "LAMBDA_NOOP=0.005" \
    "MASK_FLOOR=0.05" \
    "MASK_FLOOR_STEPS=12000"

run_variant "A_weakreg" \
    "LAMBDA_NOOP=0.005" \
    "MASK_FLOOR=0.0" \
    "MASK_FLOOR_STEPS=0"

run_variant "C_no_noop" \
    "LAMBDA_NOOP=0.0" \
    "MASK_FLOOR=0.0" \
    "MASK_FLOOR_STEPS=0"
