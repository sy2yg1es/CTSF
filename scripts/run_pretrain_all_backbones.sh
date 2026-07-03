#!/bin/bash
set -euo pipefail

# ==============================================================================
# Pretrain all PatchTST backbone weights needed by Prompt-Z experiments.
#
# Produces canonical names consumed by run_prompt_z_complete.sh:
#   weights/patchtst_pretrained_<Dataset>_H<horizon>.pth
#
# Defaults cover the 28-case matrix:
#   7 datasets x H in {1,24,48,96}
#
# Useful overrides:
#   PRETRAIN_EPOCHS=10 PRETRAIN_TRAIN_RATIO=0.6 FORCE_PRETRAIN=1 bash scripts/run_pretrain_all_backbones.sh
#   CASES="ECL.csv:1 ECL.csv:24" bash scripts/run_pretrain_all_backbones.sh
#   PRETRAIN_ACCUM_STEPS=2 bash scripts/run_pretrain_all_backbones.sh
# ==============================================================================

ROOT_PATH=${ROOT_PATH:-}
if [ -z "${ROOT_PATH}" ]; then
    if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi
fi

LOGDIR=${LOGDIR:-logs/pretrain_backbone}
BACKBONE=${BACKBONE:-patchtst}
D_MODEL=${D_MODEL:-512}
SEQ_LEN=${SEQ_LEN:-96}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-10}
PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-32}
PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS:-1}
PRETRAIN_MAX_EFFECTIVE_BS=${PRETRAIN_MAX_EFFECTIVE_BS:-1000}
PRETRAIN_TRAIN_RATIO=${PRETRAIN_TRAIN_RATIO:-0.6}
FORCE_PRETRAIN=${FORCE_PRETRAIN:-0}

CASES=${CASES:-"\
ECL.csv:1 ECL.csv:24 ECL.csv:48 ECL.csv:96 \
Traffic.csv:1 Traffic.csv:24 Traffic.csv:48 Traffic.csv:96 \
ETTh1.csv:1 ETTh1.csv:24 ETTh1.csv:48 ETTh1.csv:96 \
WTH.csv:1 WTH.csv:24 WTH.csv:48 WTH.csv:96 \
ETTm2.csv:1 ETTm2.csv:24 ETTm2.csv:48 ETTm2.csv:96 \
ETTm1.csv:1 ETTm1.csv:24 ETTm1.csv:48 ETTm1.csv:96 \
ETTh2.csv:1 ETTh2.csv:24 ETTh2.csv:48 ETTh2.csv:96"}

mkdir -p "${LOGDIR}" weights

canonical_weight_path() {
    local ds="$1"
    local h="$2"
    echo "weights/${BACKBONE}_pretrained_${ds}_H${h}.pth"
}

actual_pretrain_output_path() {
    local ds="$1"
    local h="$2"
    if [ "${PRETRAIN_ACCUM_STEPS}" = "1" ]; then
        canonical_weight_path "${ds}" "${h}"
    else
        echo "weights/${BACKBONE}_pretrained_${ds}_H${h}_GA${PRETRAIN_ACCUM_STEPS}.pth"
    fi
}

echo "================================================================="
echo " Backbone Pretrain All"
echo "================================================================="
echo "ROOT_PATH=${ROOT_PATH}"
echo "LOGDIR=${LOGDIR}"
echo "BACKBONE=${BACKBONE}"
echo "CASES=${CASES}"
echo "PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}"
echo "PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE}"
echo "PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS}"
echo "PRETRAIN_MAX_EFFECTIVE_BS=${PRETRAIN_MAX_EFFECTIVE_BS}"
echo "PRETRAIN_TRAIN_RATIO=${PRETRAIN_TRAIN_RATIO}"
echo "FORCE_PRETRAIN=${FORCE_PRETRAIN}"
echo "================================================================="

for item in ${CASES}; do
    data=$(echo "${item}" | cut -d: -f1)
    h=$(echo "${item}" | cut -d: -f2)
    ds="${data%.csv}"
    canonical=$(canonical_weight_path "${ds}" "${h}")
    actual=$(actual_pretrain_output_path "${ds}" "${h}")
    log="${LOGDIR}/pretrain_${BACKBONE}_${ds}_H${h}.log"

    echo ""
    echo "================================================================="
    echo " Pretrain backbone: ${ds} H=${h}"
    echo "================================================================="
    echo "[*] canonical=${canonical}"
    echo "[*] actual=${actual}"

    if [ -f "${canonical}" ] && [ "${FORCE_PRETRAIN}" != "1" ]; then
        echo "[SKIP] Existing canonical backbone: ${canonical}"
        continue
    fi

    PYTHONPATH=. python pretrain.py \
        --root_path "${ROOT_PATH}" \
        --data_path "${data}" \
        --features M \
        --seq_len "${SEQ_LEN}" \
        --forecast_H "${h}" \
        --D_model "${D_MODEL}" \
        --batch_size "${PRETRAIN_BATCH_SIZE}" \
        --accum_steps "${PRETRAIN_ACCUM_STEPS}" \
        --max_effective_bs "${PRETRAIN_MAX_EFFECTIVE_BS}" \
        --epochs "${PRETRAIN_EPOCHS}" \
        --train_ratio "${PRETRAIN_TRAIN_RATIO}" \
        --backbone "${BACKBONE}" \
        2>&1 | tee "${log}"

    if [ ! -f "${actual}" ]; then
        echo "[FATAL] Expected pretrain output not found: ${actual}" >&2
        echo "        See log: ${log}" >&2
        exit 1
    fi

    if [ "${actual}" != "${canonical}" ]; then
        cp -f "${actual}" "${canonical}"
        echo "[*] Copied ${actual} -> ${canonical}"
    fi

    if [ ! -f "${canonical}" ]; then
        echo "[FATAL] Missing canonical backbone after pretrain: ${canonical}" >&2
        exit 1
    fi
done

echo ""
echo "================================================================="
echo " Backbone Pretrain Finished"
echo "================================================================="
python scripts/audit_prompt_z_weights.py
