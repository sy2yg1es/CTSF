#!/bin/bash
set -euo pipefail

# ==============================================================================
# Prompt-Z conservative delta-ratio probe.
#
# Runs the requested 8 cases with max_delta_ratio=0.02:
#   degraded cases: ECL H24, ETTh2 H96, WTH H48, WTH H96
#   strong-win cases: ECL H96, Traffic H24, ETTh1 H1, Traffic H48
#
# Compare logs/prompt_z/summary_${RUN_TAG}.csv against the corresponding
# max_delta_ratio=0.05 delayed run.
# ==============================================================================

RUN_TAG=${RUN_TAG:-clamp002_floor_delayed_probe}
MAX_DELTA_RATIO=${MAX_DELTA_RATIO:-0.02}

CASES=${CASES:-"\
ECL.csv:24:321 ECL.csv:96:321 \
ETTh2.csv:96:7 \
WTH.csv:48:12 WTH.csv:96:12 \
Traffic.csv:24:862 Traffic.csv:48:862 \
ETTh1.csv:1:7"}

DELAYED_RESIDUAL_TRAINING=${DELAYED_RESIDUAL_TRAINING:-1}
LAMBDA_DELTA=${LAMBDA_DELTA:-0.0002}
LAMBDA_MASK=${LAMBDA_MASK:-0.0001}
LAMBDA_NOOP=${LAMBDA_NOOP:-0.005}
TARGET_MASK_RATIO=${TARGET_MASK_RATIO:-0.10}
GAMMA_FLOOR=${GAMMA_FLOOR:-0.1}
GAMMA_FLOOR_STEPS=${GAMMA_FLOOR_STEPS:-8000}
MASK_FLOOR=${MASK_FLOOR:-0.05}
MASK_FLOOR_STEPS=${MASK_FLOOR_STEPS:-12000}
NOOP_WARMUP_STEPS=${NOOP_WARMUP_STEPS:-14000}
ENABLE_VALIDATION_FALLBACK=${ENABLE_VALIDATION_FALLBACK:-1}
FALLBACK_MARGIN=${FALLBACK_MARGIN:-0.005}
VAL_RATIO=${VAL_RATIO:-0.1}
FALLBACK_MODE=${FALLBACK_MODE:-mode0}

export RUN_TAG CASES MAX_DELTA_RATIO
export DELAYED_RESIDUAL_TRAINING ENABLE_VALIDATION_FALLBACK
export LAMBDA_DELTA LAMBDA_MASK LAMBDA_NOOP TARGET_MASK_RATIO
export GAMMA_FLOOR GAMMA_FLOOR_STEPS MASK_FLOOR MASK_FLOOR_STEPS
export NOOP_WARMUP_STEPS
export FALLBACK_MARGIN VAL_RATIO FALLBACK_MODE

echo "================================================================="
echo " Prompt-Z max_delta_ratio=0.02 Probe"
echo "================================================================="
echo "RUN_TAG=${RUN_TAG}"
echo "CASES=${CASES}"
echo "MAX_DELTA_RATIO=${MAX_DELTA_RATIO}"
echo "LAMBDA_DELTA=${LAMBDA_DELTA} LAMBDA_MASK=${LAMBDA_MASK} LAMBDA_NOOP=${LAMBDA_NOOP} TARGET_MASK_RATIO=${TARGET_MASK_RATIO}"
echo "GAMMA_FLOOR=${GAMMA_FLOOR} GAMMA_FLOOR_STEPS=${GAMMA_FLOOR_STEPS}"
echo "MASK_FLOOR=${MASK_FLOOR} MASK_FLOOR_STEPS=${MASK_FLOOR_STEPS}"
echo "NOOP_WARMUP_STEPS=${NOOP_WARMUP_STEPS}"
echo "DELAYED_RESIDUAL_TRAINING=${DELAYED_RESIDUAL_TRAINING}"
echo "ENABLE_VALIDATION_FALLBACK=${ENABLE_VALIDATION_FALLBACK} FALLBACK_MARGIN=${FALLBACK_MARGIN}"
echo "================================================================="

bash scripts/run_prompt_z_complete.sh
