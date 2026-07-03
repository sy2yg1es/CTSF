#!/bin/bash
set -euo pipefail

# ==============================================================================
# Prompt-Z complete rerun under the strict CTSF protocol
#
# What this script fixes:
#   1) Never silently falls back to random backbone initialization.
#   2) Finds the current CTSF pretrained weight naming convention:
#        weights/patchtst_pretrained_<Dataset>_H<horizon>.pth
#   3) Trains Prompt-Z with the loaded backbone.
#   4) Evaluates frozen / random-PromptZ / trained mode0 / trained mode1.
#   5) Writes a compact CSV summary with deltas vs frozen.
#
# Usage:
#   bash scripts/run_prompt_z_complete.sh
#
# Useful overrides:
#   RUN_TAG=pretrained_v2 FORCE_TRAIN=1 RUN_PRETRAIN=1 PRETRAIN_EPOCHS=10 \
#     bash scripts/run_prompt_z_complete.sh
#
#   CASES="ECL.csv:96:321 Traffic.csv:1:862 ETTh1.csv:24:7 WTH.csv:24:auto" \
#     bash scripts/run_prompt_z_complete.sh
# ==============================================================================

ROOT_PATH=${ROOT_PATH:-}
if [ -z "${ROOT_PATH}" ]; then
    if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi
fi

LOGDIR=${LOGDIR:-logs/prompt_z}
WEIGHT_DIR=${WEIGHT_DIR:-weights/prompt_z}
RUN_TAG=${RUN_TAG:-pretrained}
WEIGHT_TAG=${WEIGHT_TAG:-$RUN_TAG}
SELECTED_ONLY=${SELECTED_ONLY:-0}

BACKBONE=${BACKBONE:-patchtst}
D_MODEL=${D_MODEL:-512}
D_FF=${D_FF:-512}
E_LAYERS=${E_LAYERS:-3}
SEQ_LEN=${SEQ_LEN:-96}
WORKERS=${WORKERS:-4}

RUN_PRETRAIN=${RUN_PRETRAIN:-1}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-10}
PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-32}
PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS:-1}
PRETRAIN_MAX_EFFECTIVE_BS=${PRETRAIN_MAX_EFFECTIVE_BS:-1000}
PRETRAIN_TRAIN_RATIO=${PRETRAIN_TRAIN_RATIO:-0.6}

FORCE_TRAIN=${FORCE_TRAIN:-1}
EPOCHS=${EPOCHS:-3}
LR=${LR:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
RANK=${RANK:-8}
D_DRIFT=${D_DRIFT:-64}
RESIDUAL_WINDOW_K=${RESIDUAL_WINDOW_K:-24}
GAMMA_INIT_BIAS=${GAMMA_INIT_BIAS:--3.0}
MASK_INIT_BIAS=${MASK_INIT_BIAS:--1.5}
MAX_DELTA_RATIO=${MAX_DELTA_RATIO:-0.05}
LAMBDA_DELTA=${LAMBDA_DELTA:-0.0002}
LAMBDA_MASK=${LAMBDA_MASK:-0.0001}
LAMBDA_NOOP=${LAMBDA_NOOP:-0.005}
TARGET_MASK_RATIO=${TARGET_MASK_RATIO:-0.10}
REG_WARMUP_STEPS=${REG_WARMUP_STEPS:-2000}
NOOP_WARMUP_STEPS=${NOOP_WARMUP_STEPS:-14000}
NOOP_RAMP_STEPS=${NOOP_RAMP_STEPS:-2000}
NOOP_MIN_EFFECTIVE_RATIO=${NOOP_MIN_EFFECTIVE_RATIO:-0.0001}
GAMMA_FLOOR=${GAMMA_FLOOR:-0.1}
GAMMA_FLOOR_STEPS=${GAMMA_FLOOR_STEPS:-8000}
MASK_FLOOR=${MASK_FLOOR:-0.05}
MASK_FLOOR_STEPS=${MASK_FLOOR_STEPS:-12000}
DELAYED_RESIDUAL_TRAINING=${DELAYED_RESIDUAL_TRAINING:-1}
CALIBRATION_LR=${CALIBRATION_LR:-0.0001}
ENABLE_VALIDATION_FALLBACK=${ENABLE_VALIDATION_FALLBACK:-1}
FALLBACK_MARGIN=${FALLBACK_MARGIN:-0.005}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.1}
VALIDATION_STEPS=${VALIDATION_STEPS:-}
FALLBACK_MODE=${FALLBACK_MODE:-mode0}

CASES=${CASES:-"ECL.csv:96:321 Traffic.csv:1:862 ETTh1.csv:24:7"}

if [ "${SELECTED_ONLY}" = "1" ]; then
    RUN_PRETRAIN=0
    FORCE_TRAIN=0
fi

mkdir -p "${LOGDIR}" "${WEIGHT_DIR}" weights

detect_enc_in() {
    local data="$1"
    local path=""
    local candidate
    for candidate in "${ROOT_PATH%/}/${data}" "./data/${data}" "./dataset/${data}"; do
        if [ -f "${candidate}" ]; then
            path="${candidate}"
            break
        fi
    done
    if [ -z "${path}" ]; then
        echo ""
        return 0
    fi

    python -c 'import csv,sys; h=next(csv.reader(open(sys.argv[1], newline=""))); print(len([c for c in h if c.strip().lower()!="date"]))' "${path}"
}

show_prompt_z_weight_candidates() {
    local ds="$1"
    local h="$2"
    local found=0
    local path

    while IFS= read -r path; do
        found=1
        echo "  - $(basename "${path}")" >&2
    done < <(find "${WEIGHT_DIR}" -maxdepth 1 -type f -name "prompt_z_${ds}_H${h}_*_final.pth" | sort)

    if [ "${found}" = "0" ]; then
        echo "  (none found for ${ds} H=${h})" >&2
    fi
}

find_backbone_weights() {
    local ds="$1"
    local h="$2"
    local ga="$3"

    local pats=()
    if [ "${BACKBONE}" = "patchtst" ]; then
        pats+=("weights/patchtst_pretrained_${ds}_H${h}_GA${ga}.pth")
        pats+=("weights/patchtst_pretrained_${ds}_H${h}_GA1.pth")
        pats+=("weights/patchtst_pretrained_${ds}_H${h}.pth")
        pats+=("weights/backbone/patchtst_pretrained_${ds}_H${h}_GA${ga}.pth")
        pats+=("weights/backbone/patchtst_pretrained_${ds}_H${h}_GA1.pth")
        pats+=("weights/backbone/patchtst_pretrained_${ds}_H${h}.pth")
        pats+=("weights/pretrained/patchtst_pretrained_${ds}_H${h}_GA${ga}.pth")
        pats+=("weights/pretrained/patchtst_pretrained_${ds}_H${h}_GA1.pth")
        pats+=("weights/pretrained/patchtst_pretrained_${ds}_H${h}.pth")
    else
        pats+=("weights/itransformer_pretrained_${ds}_H${h}.pth")
        pats+=("weights/backbone/itransformer_pretrained_${ds}_H${h}.pth")
        pats+=("weights/pretrained/itransformer_pretrained_${ds}_H${h}.pth")
    fi
    pats+=("weights/pretrain_${ds}_H${h}.pth")
    pats+=("weights/pretrained_${ds}_H${h}.pth")
    pats+=("weights/${ds}_H${h}_best.pth")
    pats+=("weights/${ds}_H${h}.pth")
    pats+=("checkpoints/pretrain_${ds}_H${h}/checkpoint.pth")

    local pat
    for pat in "${pats[@]}"; do
        if [ -f "${pat}" ]; then
            echo "${pat}"
            return 0
        fi
    done
    return 1
}

show_backbone_weight_candidates() {
    local ds="$1"
    local h="$2"
    local ga="$3"

    echo "  searched:" >&2
    if [ "${BACKBONE}" = "patchtst" ]; then
        echo "  - weights/patchtst_pretrained_${ds}_H${h}_GA${ga}.pth" >&2
        echo "  - weights/patchtst_pretrained_${ds}_H${h}_GA1.pth" >&2
        echo "  - weights/patchtst_pretrained_${ds}_H${h}.pth" >&2
        echo "  - weights/backbone/patchtst_pretrained_${ds}_H${h}*.pth" >&2
        echo "  - weights/pretrained/patchtst_pretrained_${ds}_H${h}*.pth" >&2
    else
        echo "  - weights/itransformer_pretrained_${ds}_H${h}.pth" >&2
        echo "  - weights/backbone/itransformer_pretrained_${ds}_H${h}.pth" >&2
        echo "  - weights/pretrained/itransformer_pretrained_${ds}_H${h}.pth" >&2
    fi
    echo "  - weights/pretrain_${ds}_H${h}.pth" >&2
    echo "  - weights/pretrained_${ds}_H${h}.pth" >&2
    echo "  - weights/${ds}_H${h}_best.pth" >&2
    echo "  - weights/${ds}_H${h}.pth" >&2
    echo "  - checkpoints/pretrain_${ds}_H${h}/checkpoint.pth" >&2
}

ensure_backbone_weights() {
    local data="$1"
    local h="$2"
    local enc_in="$3"
    local ds="${data%.csv}"
    local weights=""

    if weights=$(find_backbone_weights "${ds}" "${h}" "${PRETRAIN_ACCUM_STEPS}"); then
        echo "${weights}"
        return 0
    fi

    if [ "${RUN_PRETRAIN}" != "1" ]; then
        echo "[FATAL] Missing pretrained backbone for ${ds} H=${h}; set RUN_PRETRAIN=1 or upload weights." >&2
        show_backbone_weight_candidates "${ds}" "${h}" "${PRETRAIN_ACCUM_STEPS}"
        return 1
    fi

    echo "[PRETRAIN] ${ds} H=${h} -> weights/${BACKBONE}_pretrained_${ds}_H${h}.pth" >&2
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
        > "${LOGDIR}/pretrain_${ds}_H${h}_${RUN_TAG}.log" 2>&1

    if weights=$(find_backbone_weights "${ds}" "${h}" "${PRETRAIN_ACCUM_STEPS}"); then
        echo "${weights}"
        return 0
    fi

    echo "[FATAL] Pretrain finished but no backbone weight was found for ${ds} H=${h}." >&2
    return 1
}

train_prompt_z_case() {
    local data="$1"
    local h="$2"
    local enc_in="$3"
    local weights="$4"
    local ds="${data%.csv}"
    local final_weight="${WEIGHT_DIR}/prompt_z_${ds}_H${h}_${RUN_TAG}_final.pth"
    local log="${LOGDIR}/train_${ds}_H${h}_${RUN_TAG}.log"

    if [ -f "${final_weight}" ] && [ "${FORCE_TRAIN}" != "1" ]; then
        echo "[TRAIN] ${ds} H=${h} skip, exists: ${final_weight}"
        return 0
    fi

    local delay_args=()
    if [ "${DELAYED_RESIDUAL_TRAINING}" = "1" ]; then
        delay_args+=(--delayed_residual_training)
    else
        delay_args+=(--no_delayed_residual_training)
    fi

    echo "[TRAIN] ${ds} H=${h} | backbone=${weights}"
    PYTHONPATH=. python train_prompt_z.py \
        --root_path "${ROOT_PATH}" \
        --data_path "${data}" \
        --features M \
        --seq_len "${SEQ_LEN}" \
        --forecast_H "${h}" \
        --enc_in "${enc_in}" \
        --backbone "${BACKBONE}" \
        --D_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --e_layers "${E_LAYERS}" \
        --pretrained_weights "${weights}" \
        --d_drift "${D_DRIFT}" \
        --rank "${RANK}" \
        --gamma_init_bias "${GAMMA_INIT_BIAS}" \
        --mask_init_bias "${MASK_INIT_BIAS}" \
        --max_delta_ratio "${MAX_DELTA_RATIO}" \
        --residual_window_K "${RESIDUAL_WINDOW_K}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --lambda_delta "${LAMBDA_DELTA}" \
        --lambda_mask "${LAMBDA_MASK}" \
        --lambda_noop "${LAMBDA_NOOP}" \
        --target_mask_ratio "${TARGET_MASK_RATIO}" \
        --reg_warmup_steps "${REG_WARMUP_STEPS}" \
        --noop_warmup_steps "${NOOP_WARMUP_STEPS}" \
        --noop_ramp_steps "${NOOP_RAMP_STEPS}" \
        --noop_min_effective_ratio "${NOOP_MIN_EFFECTIVE_RATIO}" \
        --gamma_floor "${GAMMA_FLOOR}" \
        --gamma_floor_steps "${GAMMA_FLOOR_STEPS}" \
        --mask_floor "${MASK_FLOOR}" \
        --mask_floor_steps "${MASK_FLOOR_STEPS}" \
        "${delay_args[@]}" \
        --num_workers "${WORKERS}" \
        --save_dir "${WEIGHT_DIR}" \
        --train_ratio "${TRAIN_RATIO}" \
        --experiment_tag "${RUN_TAG}" \
        2>&1 | tee "${log}"

    if ! grep -q "Loaded pretrained weights" "${log}"; then
        echo "[FATAL] ${ds} H=${h} training log did not confirm pretrained backbone loading." >&2
        return 1
    fi
    if [ ! -f "${final_weight}" ]; then
        echo "[FATAL] Missing Prompt-Z final weight: ${final_weight}" >&2
        return 1
    fi
}

run_eval_case() {
    local tag="$1"
    local data="$2"
    local h="$3"
    local enc_in="$4"
    local weights="$5"
    local mode="$6"
    local pz_weight="${7:-}"
    local log="${LOGDIR}/${tag}.log"

    local args=(
        --root_path "${ROOT_PATH}"
        --data_path "${data}"
        --features M
        --seq_len "${SEQ_LEN}"
        --forecast_H "${h}"
        --enc_in "${enc_in}"
        --backbone "${BACKBONE}"
        --D_model "${D_MODEL}"
        --d_ff "${D_FF}"
        --e_layers "${E_LAYERS}"
        --pretrained_weights "${weights}"
        --d_drift "${D_DRIFT}"
        --rank "${RANK}"
        --gamma_init_bias "${GAMMA_INIT_BIAS}"
        --mask_init_bias "${MASK_INIT_BIAS}"
        --max_delta_ratio "${MAX_DELTA_RATIO}"
        --residual_window_K "${RESIDUAL_WINDOW_K}"
        --streaming_mode "${mode}"
        --calibration_lr "${CALIBRATION_LR}"
        --num_workers "${WORKERS}"
        --experiment_tag "${tag}"
    )
    if [ -n "${pz_weight}" ]; then
        args+=(--prompt_z_weights "${pz_weight}")
    fi

    echo -n "[EVAL] ${tag} ..."
    PYTHONPATH=. python main_prompt_z.py "${args[@]}" > "${log}" 2>&1

    if ! grep -q "Loaded backbone" "${log}"; then
        echo " FAIL"
        echo "[FATAL] ${tag} did not confirm backbone loading. See ${log}" >&2
        return 1
    fi

    local mse
    mse=$(grep "Final MSE" "${log}" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)
    echo " MSE=${mse:-FAIL}"
    if [ -z "${mse}" ]; then
        echo "[FATAL] Could not parse Final MSE for ${tag}. See ${log}" >&2
        return 1
    fi
}

run_selected_case() {
    local tag="$1"
    local data="$2"
    local h="$3"
    local enc_in="$4"
    local weights="$5"
    local pz_weight="$6"
    local log="${LOGDIR}/${tag}.log"

    local args=(
        --root_path "${ROOT_PATH}"
        --data_path "${data}"
        --features M
        --seq_len "${SEQ_LEN}"
        --forecast_H "${h}"
        --enc_in "${enc_in}"
        --backbone "${BACKBONE}"
        --D_model "${D_MODEL}"
        --d_ff "${D_FF}"
        --e_layers "${E_LAYERS}"
        --pretrained_weights "${weights}"
        --d_drift "${D_DRIFT}"
        --rank "${RANK}"
        --gamma_init_bias "${GAMMA_INIT_BIAS}"
        --mask_init_bias "${MASK_INIT_BIAS}"
        --max_delta_ratio "${MAX_DELTA_RATIO}"
        --residual_window_K "${RESIDUAL_WINDOW_K}"
        --streaming_mode "${FALLBACK_MODE}"
        --prompt_z_weights "${pz_weight}"
        --calibration_lr "${CALIBRATION_LR}"
        --enable_validation_fallback
        --fallback_margin "${FALLBACK_MARGIN}"
        --val_ratio "${VAL_RATIO}"
        --num_workers "${WORKERS}"
        --experiment_tag "${tag}"
    )
    if [ -n "${VALIDATION_STEPS}" ]; then
        args+=(--validation_steps "${VALIDATION_STEPS}")
    fi

    echo -n "[EVAL] ${tag} ..."
    PYTHONPATH=. python main_prompt_z.py "${args[@]}" > "${log}" 2>&1

    if ! grep -q "Loaded backbone" "${log}"; then
        echo " FAIL"
        echo "[FATAL] ${tag} did not confirm backbone loading. See ${log}" >&2
        return 1
    fi

    local mse
    mse=$(grep "Final MSE" "${log}" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)
    echo " MSE=${mse:-FAIL}"
    if [ -z "${mse}" ]; then
        echo "[FATAL] Could not parse Final MSE for ${tag}. See ${log}" >&2
        return 1
    fi
}

write_summary() {
    PYTHONPATH=. python scripts/summarize_prompt_z.py \
        --tag "${RUN_TAG}" \
        --logdir "${LOGDIR}"
}

echo "================================================================="
echo " Prompt-Z Complete Rerun"
echo "================================================================="
echo "ROOT_PATH=${ROOT_PATH}"
echo "LOGDIR=${LOGDIR}"
echo "WEIGHT_DIR=${WEIGHT_DIR}"
echo "RUN_TAG=${RUN_TAG}"
echo "WEIGHT_TAG=${WEIGHT_TAG}"
echo "SELECTED_ONLY=${SELECTED_ONLY}"
echo "CASES=${CASES}"
echo "RUN_PRETRAIN=${RUN_PRETRAIN} PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS} PRETRAIN_TRAIN_RATIO=${PRETRAIN_TRAIN_RATIO}"
echo "FORCE_TRAIN=${FORCE_TRAIN} EPOCHS=${EPOCHS}"
echo "MAX_DELTA_RATIO=${MAX_DELTA_RATIO}"
echo "LAMBDA_DELTA=${LAMBDA_DELTA} LAMBDA_MASK=${LAMBDA_MASK} LAMBDA_NOOP=${LAMBDA_NOOP} TARGET_MASK_RATIO=${TARGET_MASK_RATIO}"
echo "REG_WARMUP_STEPS=${REG_WARMUP_STEPS} NOOP_WARMUP_STEPS=${NOOP_WARMUP_STEPS} NOOP_RAMP_STEPS=${NOOP_RAMP_STEPS}"
echo "GAMMA_FLOOR=${GAMMA_FLOOR} GAMMA_FLOOR_STEPS=${GAMMA_FLOOR_STEPS}"
echo "MASK_FLOOR=${MASK_FLOOR} MASK_FLOOR_STEPS=${MASK_FLOOR_STEPS}"
echo "DELAYED_RESIDUAL_TRAINING=${DELAYED_RESIDUAL_TRAINING}"
echo "ENABLE_VALIDATION_FALLBACK=${ENABLE_VALIDATION_FALLBACK} FALLBACK_MODE=${FALLBACK_MODE} FALLBACK_MARGIN=${FALLBACK_MARGIN} TRAIN_RATIO=${TRAIN_RATIO} VAL_RATIO=${VAL_RATIO} VALIDATION_STEPS=${VALIDATION_STEPS}"
echo "================================================================="

for item in ${CASES}; do
    data=$(echo "${item}" | cut -d: -f1)
    h=$(echo "${item}" | cut -d: -f2)
    enc_in=$(echo "${item}" | cut -d: -f3)
    ds="${data%.csv}"
    case_tag="${ds}_H${h}_${RUN_TAG}"
    detected_enc=$(detect_enc_in "${data}")
    if [ -n "${detected_enc}" ]; then
        if [ -z "${enc_in}" ] || [ "${enc_in}" = "auto" ]; then
            echo "[*] Auto-detected enc_in for ${data}: ${detected_enc}"
        elif [ "${enc_in}" != "${detected_enc}" ]; then
            echo "[!] enc_in mismatch for ${data}: CASES=${enc_in}, CSV=${detected_enc}; using CSV value."
        fi
        enc_in="${detected_enc}"
    fi

    echo ""
    echo "================================================================="
    echo " Case: ${ds} H=${h}"
    echo "================================================================="

    backbone_weights=$(ensure_backbone_weights "${data}" "${h}" "${enc_in}")
    echo "[*] Backbone weights: ${backbone_weights}"

    prompt_z_weight="${WEIGHT_DIR}/prompt_z_${ds}_H${h}_${WEIGHT_TAG}_final.pth"

    if [ "${SELECTED_ONLY}" = "1" ]; then
        if [ ! -f "${prompt_z_weight}" ]; then
            echo "[FATAL] SELECTED_ONLY=1 but missing Prompt-Z weight: ${prompt_z_weight}" >&2
            echo "[HINT] Available Prompt-Z final weights for ${ds} H=${h}:" >&2
            show_prompt_z_weight_candidates "${ds}" "${h}"
            echo "[HINT] If you want to reuse an existing trained weight with a new eval/log tag," >&2
            echo "       keep RUN_TAG for outputs and set WEIGHT_TAG to the training tag." >&2
            echo "       Example: RUN_TAG=${RUN_TAG} WEIGHT_TAG=main_B_12cases SELECTED_ONLY=1 bash scripts/run_prompt_z_main_12cases.sh" >&2
            exit 1
        fi
        echo "[*] Prompt-Z weights: ${prompt_z_weight}"
        run_selected_case "${case_tag}_pz_selected" "${data}" "${h}" "${enc_in}" "${backbone_weights}" "${prompt_z_weight}"
        continue
    fi

    train_prompt_z_case "${data}" "${h}" "${enc_in}" "${backbone_weights}"
    prompt_z_weight="${WEIGHT_DIR}/prompt_z_${ds}_H${h}_${RUN_TAG}_final.pth"
    echo "[*] Prompt-Z weights: ${prompt_z_weight}"

    run_eval_case "${case_tag}_frozen" "${data}" "${h}" "${enc_in}" "${backbone_weights}" frozen
    run_eval_case "${case_tag}_pz_random_mode0" "${data}" "${h}" "${enc_in}" "${backbone_weights}" mode0
    run_eval_case "${case_tag}_pz_mode0" "${data}" "${h}" "${enc_in}" "${backbone_weights}" mode0 "${prompt_z_weight}"
    run_eval_case "${case_tag}_pz_mode1" "${data}" "${h}" "${enc_in}" "${backbone_weights}" mode1 "${prompt_z_weight}"
    if [ "${ENABLE_VALIDATION_FALLBACK}" = "1" ]; then
        run_selected_case "${case_tag}_pz_selected" "${data}" "${h}" "${enc_in}" "${backbone_weights}" "${prompt_z_weight}"
    fi
done

echo ""
echo "================================================================="
echo " Summary"
echo "================================================================="
write_summary

echo "================================================================="
echo " Prompt-Z Complete Rerun Finished"
echo "================================================================="
