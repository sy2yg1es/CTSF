#!/bin/bash
set -euo pipefail

# ==============================================================================
# Pretrain the requested backbone grid:
#   1) PatchTST on all datasets with H=12
#   2) iTransformer on all datasets with H in {1, 12, 24, 48}
#
# Canonical outputs:
#   weights/patchtst_pretrained_<Dataset>_H12.pth
#   weights/itransformer_pretrained_<Dataset>_H<horizon>.pth
#
# Usage:
#   bash scripts/run_pretrain_patchtst_h12_itransformer_grid.sh
#
# Useful overrides:
#   FORCE_PRETRAIN=1 bash scripts/run_pretrain_patchtst_h12_itransformer_grid.sh
#   PRETRAIN_EPOCHS=20 PRETRAIN_ACCUM_STEPS=2 bash scripts/run_pretrain_patchtst_h12_itransformer_grid.sh
#   DATASETS="ECL.csv Traffic.csv" bash scripts/run_pretrain_patchtst_h12_itransformer_grid.sh
# ==============================================================================

ROOT_PATH=${ROOT_PATH:-}
if [ -z "${ROOT_PATH}" ]; then
    if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi
fi

LOGDIR=${LOGDIR:-logs/pretrain_backbone/h12_itransformer_grid}
D_MODEL=${D_MODEL:-512}
SEQ_LEN=${SEQ_LEN:-96}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-10}
PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-32}
PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS:-1}
PRETRAIN_MAX_EFFECTIVE_BS=${PRETRAIN_MAX_EFFECTIVE_BS:-1000}
PRETRAIN_TRAIN_RATIO=${PRETRAIN_TRAIN_RATIO:-0.6}
PRETRAIN_VAL_RATIO=${PRETRAIN_VAL_RATIO:-0.1}
FORCE_PRETRAIN=${FORCE_PRETRAIN:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

DATASETS=${DATASETS:-"ECL.csv Traffic.csv ETTh1.csv WTH.csv ETTm2.csv ETTm1.csv ETTh2.csv"}
PATCHTST_HORIZONS=${PATCHTST_HORIZONS:-"12"}
ITRANSFORMER_HORIZONS=${ITRANSFORMER_HORIZONS:-"1 12 24 48"}

mkdir -p "${LOGDIR}" weights

canonical_weight_path() {
    local backbone="$1"
    local ds="$2"
    local h="$3"
    echo "weights/${backbone}_pretrained_${ds}_H${h}.pth"
}

actual_pretrain_output_path() {
    local backbone="$1"
    local ds="$2"
    local h="$3"

    if [ "${PRETRAIN_ACCUM_STEPS}" = "1" ]; then
        canonical_weight_path "${backbone}" "${ds}" "${h}"
    else
        echo "weights/${backbone}_pretrained_${ds}_H${h}_GA${PRETRAIN_ACCUM_STEPS}.pth"
    fi
}

run_pretrain_case() {
    local backbone="$1"
    local data="$2"
    local h="$3"
    local ds="${data%.csv}"
    local canonical
    local actual
    local log

    canonical=$(canonical_weight_path "${backbone}" "${ds}" "${h}")
    actual=$(actual_pretrain_output_path "${backbone}" "${ds}" "${h}")
    log="${LOGDIR}/pretrain_${backbone}_${ds}_H${h}.log"

    echo ""
    echo "================================================================="
    echo " Pretrain backbone: ${backbone} | ${ds} H=${h}"
    echo "================================================================="
    echo "[*] canonical=${canonical}"
    echo "[*] actual=${actual}"
    echo "[*] log=${log}"

    if [ -f "${canonical}" ] && [ "${FORCE_PRETRAIN}" != "1" ]; then
        echo "[SKIP] Existing canonical backbone: ${canonical}"
        return 0
    fi

    PYTHONPATH=. "${PYTHON_BIN}" pretrain.py \
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
        --val_ratio "${PRETRAIN_VAL_RATIO}" \
        --backbone "${backbone}" \
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
}

run_phase() {
    local backbone="$1"
    local horizons="$2"

    echo ""
    echo "#################################################################"
    echo "# Phase: ${backbone} | horizons: ${horizons}"
    echo "#################################################################"

    local data
    local h
    for data in ${DATASETS}; do
        for h in ${horizons}; do
            run_pretrain_case "${backbone}" "${data}" "${h}"
        done
    done
}

verify_phase() {
    local backbone="$1"
    local horizons="$2"
    local missing=0
    local data
    local h
    local ds
    local path

    for data in ${DATASETS}; do
        ds="${data%.csv}"
        for h in ${horizons}; do
            path=$(canonical_weight_path "${backbone}" "${ds}" "${h}")
            if [ ! -f "${path}" ]; then
                echo "[MISSING] ${path}" >&2
                missing=$((missing + 1))
            else
                echo "[OK] ${path}"
            fi
        done
    done

    return "${missing}"
}

echo "================================================================="
echo " Backbone Pretrain: PatchTST H=12 + iTransformer H=1/12/24/48"
echo "================================================================="
echo "ROOT_PATH=${ROOT_PATH}"
echo "LOGDIR=${LOGDIR}"
echo "DATASETS=${DATASETS}"
echo "PATCHTST_HORIZONS=${PATCHTST_HORIZONS}"
echo "ITRANSFORMER_HORIZONS=${ITRANSFORMER_HORIZONS}"
echo "D_MODEL=${D_MODEL}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}"
echo "PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE}"
echo "PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS}"
echo "PRETRAIN_MAX_EFFECTIVE_BS=${PRETRAIN_MAX_EFFECTIVE_BS}"
echo "PRETRAIN_TRAIN_RATIO=${PRETRAIN_TRAIN_RATIO}"
echo "PRETRAIN_VAL_RATIO=${PRETRAIN_VAL_RATIO}"
echo "FORCE_PRETRAIN=${FORCE_PRETRAIN}"
echo "================================================================="

run_phase "patchtst" "${PATCHTST_HORIZONS}"
run_phase "itransformer" "${ITRANSFORMER_HORIZONS}"

echo ""
echo "================================================================="
echo " Verify canonical weights"
echo "================================================================="
verify_phase "patchtst" "${PATCHTST_HORIZONS}"
verify_phase "itransformer" "${ITRANSFORMER_HORIZONS}"

echo ""
echo "================================================================="
echo " Backbone Pretrain Finished"
echo "================================================================="
