#!/bin/bash
set -euo pipefail

# ==============================================================================
# Train Prompt-Z and evaluate frozen-vs-PZ for iTransformer on all datasets with
# H in {1,12,24,48}.
#
# Expected canonical backbone weights:
#   weights/itransformer_pretrained_<Dataset>_H<horizon>.pth
#
# Outputs:
#   logs/prompt_z_grid/itransformer_h1_12_24_48/summary.tsv
#
# Usage:
#   bash scripts/run_eval_itransformer_grid.sh
#
# Useful overrides:
#   RUN_TAG=strict_label_v1 FORCE_TRAIN=1 bash scripts/run_eval_itransformer_grid.sh
#   PZ_MODE=mode1 bash scripts/run_eval_itransformer_grid.sh
#   DATASETS="ECL.csv Traffic.csv" HORIZONS="1 24" bash scripts/run_eval_itransformer_grid.sh
# ==============================================================================

ROOT_PATH=${ROOT_PATH:-}
if [ -z "${ROOT_PATH}" ]; then
    if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi
fi

BACKBONE="itransformer"
RUN_TAG=${RUN_TAG:-promptz_eval}
PROMPT_TAG=${PROMPT_TAG:-"${BACKBONE}_${RUN_TAG}"}
LOGDIR=${LOGDIR:-logs/prompt_z_grid/itransformer_h1_12_24_48}
WEIGHT_DIR=${WEIGHT_DIR:-weights/prompt_z}
SUMMARY_PATH=${SUMMARY_PATH:-"${LOGDIR}/summary.tsv"}
PYTHON_BIN=${PYTHON_BIN:-python}

DATASETS=${DATASETS:-"ECL.csv Traffic.csv ETTh1.csv WTH.csv ETTm2.csv ETTm1.csv ETTh2.csv"}
HORIZONS=${HORIZONS:-"1 12 24 48"}

SEQ_LEN=${SEQ_LEN:-96}
D_MODEL=${D_MODEL:-512}
D_FF=${D_FF:-512}
E_LAYERS=${E_LAYERS:-3}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.1}
WORKERS=${WORKERS:-4}
PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS:-1}
SKIP_MISSING=${SKIP_MISSING:-0}
PRINT_STEP_LOGS=${PRINT_STEP_LOGS:-1}
STEP_LOG_TAIL=${STEP_LOG_TAIL:-12}

EPOCHS=${EPOCHS:-3}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-1}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
FORCE_TRAIN=${FORCE_TRAIN:-0}
PZ_MODE=${PZ_MODE:-mode0}

D_DRIFT=${D_DRIFT:-64}
RANK=${RANK:-8}
GAMMA_INIT_BIAS=${GAMMA_INIT_BIAS:--3.0}
MASK_INIT_BIAS=${MASK_INIT_BIAS:--1.5}
MAX_DELTA_RATIO=${MAX_DELTA_RATIO:-0.05}
RESIDUAL_WINDOW_K=${RESIDUAL_WINDOW_K:-24}

LAMBDA_DELTA=${LAMBDA_DELTA:-2e-4}
LAMBDA_MASK=${LAMBDA_MASK:-1e-4}
LAMBDA_NOOP=${LAMBDA_NOOP:-0.005}
TARGET_MASK_RATIO=${TARGET_MASK_RATIO:-0.10}
REG_WARMUP_STEPS=${REG_WARMUP_STEPS:-2000}
NOOP_WARMUP_STEPS=${NOOP_WARMUP_STEPS:-6000}
NOOP_RAMP_STEPS=${NOOP_RAMP_STEPS:-2000}
NOOP_MIN_EFFECTIVE_RATIO=${NOOP_MIN_EFFECTIVE_RATIO:-1e-4}
GAMMA_FLOOR=${GAMMA_FLOOR:-0.1}
GAMMA_FLOOR_STEPS=${GAMMA_FLOOR_STEPS:-8000}
MASK_FLOOR=${MASK_FLOOR:-0.05}
MASK_FLOOR_STEPS=${MASK_FLOOR_STEPS:-12000}
DELAYED_RESIDUAL_TRAINING=${DELAYED_RESIDUAL_TRAINING:-1}

mkdir -p "${LOGDIR}" "${WEIGHT_DIR}" logs/prompt_z

if [ "${PZ_MODE}" != "mode0" ] && [ "${PZ_MODE}" != "mode1" ]; then
    echo "[FATAL] PZ_MODE must be mode0 or mode1, got: ${PZ_MODE}" >&2
    exit 1
fi

find_backbone_weights() {
    local ds="$1"
    local h="$2"
    local ga="$3"
    local pats=(
        "weights/itransformer_pretrained_${ds}_H${h}_GA${ga}.pth"
        "weights/itransformer_pretrained_${ds}_H${h}_GA1.pth"
        "weights/itransformer_pretrained_${ds}_H${h}.pth"
        "weights/backbone/itransformer_pretrained_${ds}_H${h}_GA${ga}.pth"
        "weights/backbone/itransformer_pretrained_${ds}_H${h}_GA1.pth"
        "weights/backbone/itransformer_pretrained_${ds}_H${h}.pth"
        "weights/pretrained/itransformer_pretrained_${ds}_H${h}_GA${ga}.pth"
        "weights/pretrained/itransformer_pretrained_${ds}_H${h}_GA1.pth"
        "weights/pretrained/itransformer_pretrained_${ds}_H${h}.pth"
    )

    local pat
    for pat in "${pats[@]}"; do
        if [ -f "${pat}" ]; then
            echo "${pat}"
            return 0
        fi
    done
    return 1
}

show_weight_candidates() {
    local ds="$1"
    local h="$2"
    local ga="$3"
    echo "  searched:" >&2
    echo "  - weights/itransformer_pretrained_${ds}_H${h}_GA${ga}.pth" >&2
    echo "  - weights/itransformer_pretrained_${ds}_H${h}_GA1.pth" >&2
    echo "  - weights/itransformer_pretrained_${ds}_H${h}.pth" >&2
    echo "  - weights/backbone/itransformer_pretrained_${ds}_H${h}*.pth" >&2
    echo "  - weights/pretrained/itransformer_pretrained_${ds}_H${h}*.pth" >&2
}

confirm_training_loaded_backbone() {
    local log="$1"

    if grep -Eq "Loaded (pretrained weights|backbone)" "${log}"; then
        return 0
    fi

    echo "[FATAL] Training log did not confirm pretrained backbone loading. See ${log}" >&2
    echo "[FATAL] Last 80 lines of ${log}:" >&2
    tail -80 "${log}" >&2 || true
    return 1
}

print_step_records() {
    local label="$1"
    local log="$2"
    local records

    if [ "${PRINT_STEP_LOGS}" != "1" ] || [ ! -f "${log}" ]; then
        return 0
    fi

    records=$(grep -E \
        "Loaded (pretrained weights|backbone|PromptZ weights)|delayed_residual_training=|^[[:space:]]*\\[Epoch [0-9]+ Step |^\\[Epoch [0-9]+\\]|Final weights saved|Training log saved|Final MSE|MSE=|gamma=|mask_ratio=|raw_d/h=|eff_d/h=|residual_tracker_warmed" \
        "${log}" || true)

    echo "[TRACE] ${label} key step records (${log})" >&2
    if [ -n "${records}" ]; then
        printf "%s\n" "${records}" | tail -n "${STEP_LOG_TAIL}" >&2
    else
        echo "[TRACE] no matched key records; last ${STEP_LOG_TAIL} log lines:" >&2
        tail -n "${STEP_LOG_TAIL}" "${log}" >&2 || true
    fi
}

prompt_z_final_path() {
    local ds="$1"
    local h="$2"
    echo "${WEIGHT_DIR}/prompt_z_${ds}_H${h}_${PROMPT_TAG}_final.pth"
}

train_prompt_z_case() {
    local data="$1"
    local h="$2"
    local weights="$3"
    local ds="${data%.csv}"
    local final_weight
    local log
    local delay_args=()

    final_weight=$(prompt_z_final_path "${ds}" "${h}")
    log="${LOGDIR}/train_${BACKBONE}_${ds}_H${h}_${RUN_TAG}.log"

    if [ -f "${final_weight}" ] && [ "${FORCE_TRAIN}" != "1" ]; then
        echo "[TRAIN] ${BACKBONE} ${ds} H=${h} skip, exists: ${final_weight}" >&2
        print_step_records "TRAIN-SKIP ${BACKBONE} ${ds} H=${h}" "${log}"
        echo "${final_weight}"
        return 0
    fi

    if [ "${DELAYED_RESIDUAL_TRAINING}" = "1" ]; then
        delay_args+=(--delayed_residual_training)
    else
        delay_args+=(--no_delayed_residual_training)
    fi

    echo "[TRAIN] ${BACKBONE} ${ds} H=${h} | backbone=${weights}" >&2
    if ! PYTHONPATH=. "${PYTHON_BIN}" train_prompt_z.py \
        --root_path "${ROOT_PATH}" \
        --data_path "${data}" \
        --features M \
        --seq_len "${SEQ_LEN}" \
        --forecast_H "${h}" \
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
        --batch_size "${PROMPT_BATCH_SIZE}" \
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
        --val_ratio "${VAL_RATIO}" \
        --experiment_tag "${PROMPT_TAG}" \
        > "${log}" 2>&1; then
        echo "[FATAL] Training command failed for ${BACKBONE} ${ds} H=${h}. See ${log}" >&2
        print_step_records "TRAIN-FAILED ${BACKBONE} ${ds} H=${h}" "${log}"
        tail -80 "${log}" >&2 || true
        return 1
    fi

    print_step_records "TRAIN ${BACKBONE} ${ds} H=${h}" "${log}"

    if ! confirm_training_loaded_backbone "${log}"; then
        return 1
    fi
    if [ ! -f "${final_weight}" ]; then
        echo "[FATAL] Missing Prompt-Z final weight: ${final_weight}" >&2
        return 1
    fi

    echo "${final_weight}"
}

run_eval_case() {
    local data="$1"
    local h="$2"
    local weights="$3"
    local mode="$4"
    local pz_weight="${5:-}"
    local ds="${data%.csv}"
    local tag="${BACKBONE}_${ds}_H${h}_${RUN_TAG}_${mode}"
    local log="${LOGDIR}/${tag}.log"
    local args=(
        --root_path "${ROOT_PATH}"
        --data_path "${data}"
        --features M
        --seq_len "${SEQ_LEN}"
        --forecast_H "${h}"
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
        --train_ratio "${TRAIN_RATIO}"
        --val_ratio "${VAL_RATIO}"
        --num_workers "${WORKERS}"
        --experiment_tag "${tag}"
    )
    local mse

    if [ -n "${pz_weight}" ]; then
        args+=(--prompt_z_weights "${pz_weight}")
    fi

    echo -n "[EVAL] ${tag} ..." >&2
    if ! PYTHONPATH=. "${PYTHON_BIN}" main_prompt_z.py "${args[@]}" > "${log}" 2>&1; then
        echo " FAIL" >&2
        echo "[FATAL] Eval command failed for ${tag}. See ${log}" >&2
        print_step_records "EVAL-FAILED ${tag}" "${log}"
        tail -80 "${log}" >&2 || true
        return 1
    fi

    if ! grep -q "Loaded backbone" "${log}"; then
        echo " FAIL" >&2
        echo "[FATAL] ${tag} did not confirm backbone loading. See ${log}" >&2
        print_step_records "EVAL-LOAD-FAILED ${tag}" "${log}"
        return 1
    fi
    if [ -n "${pz_weight}" ] && ! grep -q "Loaded PromptZ weights" "${log}"; then
        echo " FAIL" >&2
        echo "[FATAL] ${tag} did not confirm Prompt-Z loading. See ${log}" >&2
        print_step_records "EVAL-PZ-LOAD-FAILED ${tag}" "${log}"
        return 1
    fi

    mse=$(grep "Final MSE" "${log}" | grep -Eo '[0-9]+(\.[0-9]+)?' | tail -1 || true)
    if [ -z "${mse}" ]; then
        echo " FAIL" >&2
        echo "[FATAL] Could not parse Final MSE for ${tag}. See ${log}" >&2
        print_step_records "EVAL-PARSE-FAILED ${tag}" "${log}"
        return 1
    fi

    echo " MSE=${mse}" >&2
    print_step_records "EVAL ${tag}" "${log}"
    echo "${mse}"
}

delta_pct() {
    "${PYTHON_BIN}" -c 'import sys; f=float(sys.argv[1]); p=float(sys.argv[2]); print(f"{((p-f)/f*100):.2f}")' "$1" "$2"
}

echo "case	backbone	frozen_MSE	pz_mode	PZ_Mode_MSE	delta_pct	prompt_z_weights	backbone_weights" > "${SUMMARY_PATH}"

echo "================================================================="
echo " Prompt-Z Train + Eval: iTransformer"
echo "================================================================="
echo "ROOT_PATH=${ROOT_PATH}"
echo "LOGDIR=${LOGDIR}"
echo "WEIGHT_DIR=${WEIGHT_DIR}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "DATASETS=${DATASETS}"
echo "HORIZONS=${HORIZONS}"
echo "RUN_TAG=${RUN_TAG}"
echo "PROMPT_TAG=${PROMPT_TAG}"
echo "PZ_MODE=${PZ_MODE}"
echo "EPOCHS=${EPOCHS} PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE} FORCE_TRAIN=${FORCE_TRAIN}"
echo "TRAIN_RATIO=${TRAIN_RATIO} VAL_RATIO=${VAL_RATIO}"
echo "PRETRAIN_ACCUM_STEPS=${PRETRAIN_ACCUM_STEPS}"
echo "SKIP_MISSING=${SKIP_MISSING}"
echo "PRINT_STEP_LOGS=${PRINT_STEP_LOGS} STEP_LOG_TAIL=${STEP_LOG_TAIL}"
echo "================================================================="

for data in ${DATASETS}; do
    ds="${data%.csv}"
    for h in ${HORIZONS}; do
        weights=""
        if ! weights=$(find_backbone_weights "${ds}" "${h}" "${PRETRAIN_ACCUM_STEPS}"); then
            echo "[MISSING] ${BACKBONE} ${ds} H=${h}" >&2
            show_weight_candidates "${ds}" "${h}" "${PRETRAIN_ACCUM_STEPS}"
            if [ "${SKIP_MISSING}" = "1" ]; then
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${ds}_H${h}" "${BACKBONE}" "MISSING" "${PZ_MODE}" "MISSING" "MISSING" "MISSING" "MISSING" >> "${SUMMARY_PATH}"
                continue
            fi
            exit 1
        fi

        pz_weight=$(train_prompt_z_case "${data}" "${h}" "${weights}")
        frozen_mse=$(run_eval_case "${data}" "${h}" "${weights}" "frozen")
        pz_mse=$(run_eval_case "${data}" "${h}" "${weights}" "${PZ_MODE}" "${pz_weight}")
        delta=$(delta_pct "${frozen_mse}" "${pz_mse}")

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${ds}_H${h}" "${BACKBONE}" "${frozen_mse}" "${PZ_MODE}" "${pz_mse}" "${delta}" "${pz_weight}" "${weights}" >> "${SUMMARY_PATH}"
    done
done

echo ""
echo "================================================================="
echo " Summary"
echo "================================================================="
column -t -s $'\t' "${SUMMARY_PATH}" 2>/dev/null || cat "${SUMMARY_PATH}"
