#!/bin/bash
# Final decision run: single online prompt as the main method.
#
# Default matrix:
#   datasets: ECL Traffic ETTh1 ETTh2 ETTm1 ETTm2 WTH
#   horizons: 1 24 48 96
#   methods:
#     frozen   - E=1, K=1, zero online updates
#     full_ft  - E=1, K=1, update all channels
#     single   - E=1, K=1, improved detector + selective updates
#     ours_e32 - E=32, K=2, MoE ablation
#
# Environment overrides:
#   DATASETS="ECL ETTh1 WTH"
#   HORIZONS="1 24 96"
#   RUN_PRETRAIN=0
#   PRETRAIN_EPOCHS=10

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

DATASETS=${DATASETS:-"ECL Traffic ETTh1 ETTh2 ETTm1 ETTm2 WTH"}
HORIZONS=${HORIZONS:-"1 24 48 96"}
RUN_PRETRAIN=${RUN_PRETRAIN:-1}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-10}

D_MODEL=512
WINDOW_K=12
TAU=0.1
PATIENCE=2
LR=0.0001
L_AUX=0.0
WORKERS=4
TEMP_T=1.0
SEED=2026

LOGDIR="logs/final_single_prompt"
SUMMARY_CSV="${LOGDIR}/summary.csv"
mkdir -p "${LOGDIR}" logs/monitor

extract_metrics() {
    sed -n \
        's/.*MAE: \([0-9.]*\), MSE: \([0-9.]*\), RMSE: \([0-9.]*\).*/\1,\2,\3/p' \
        "$1" | tail -1
}

is_complete() {
    [ -f "$1" ] && grep -q "Streaming Evaluation Completed" "$1"
}

ensure_pretrain() {
    local dataset=$1
    local horizon=$2
    local data_file="${dataset}.csv"
    local weight="./weights/patchtst_pretrained_${dataset}_H${horizon}.pth"
    local log="${LOGDIR}/pretrain_${dataset}_H${horizon}.log"

    if [ -f "${weight}" ]; then
        echo "  [WEIGHT OK] ${dataset} H=${horizon}"
        return 0
    fi

    if [ "${RUN_PRETRAIN}" != "1" ]; then
        echo "  [WEIGHT MISSING] ${weight}"
        return 1
    fi

    echo "  [PRETRAIN] ${dataset} H=${horizon}"
    PYTHONPATH=. python pretrain.py \
        --root_path "${ROOT_PATH}" --data_path "${data_file}" \
        --features M --seq_len 96 --forecast_H "${horizon}" \
        --D_model "${D_MODEL}" --accum_steps 1 \
        --epochs "${PRETRAIN_EPOCHS}" --backbone patchtst \
        > "${log}" 2>&1

    if [ -f "${weight}" ]; then
        echo "    completed"
        return 0
    fi

    echo "    FAILED: inspect ${log}"
    return 1
}

run_eval() {
    local dataset=$1
    local horizon=$2
    local method=$3
    local mode=$4
    local experts=$5
    local top_k=$6
    local data_file="${dataset}.csv"
    local tag="${dataset}_H${horizon}_${method}"
    local log="${LOGDIR}/${tag}.log"

    if is_complete "${log}"; then
        local cached
        cached=$(extract_metrics "${log}")
        echo "  [SKIP] ${tag} ${cached}"
        return 0
    fi

    echo "  [RUN] ${tag}"
    PYTHONPATH=. python main.py \
        --root_path "${ROOT_PATH}" --data_path "${data_file}" \
        --features M --seq_len 96 --forecast_H "${horizon}" \
        --D_model "${D_MODEL}" --accum_steps 1 --backbone patchtst \
        --streaming_mode "${mode}" \
        --window_K "${WINDOW_K}" --threshold_tau "${TAU}" \
        --patience_C "${PATIENCE}" \
        --num_experts "${experts}" --top_k "${top_k}" \
        --temp_T "${TEMP_T}" --seed "${SEED}" \
        --learning_rate "${LR}" --l_aux_weight "${L_AUX}" \
        --num_workers "${WORKERS}" --experiment_tag "${tag}" \
        > "${log}" 2>&1

    if is_complete "${log}"; then
        echo "    $(extract_metrics "${log}")"
        return 0
    fi

    echo "    FAILED: inspect ${log}"
    return 1
}

echo "================================================================="
echo " Final Single-Prompt Evaluation"
echo "================================================================="
echo "datasets=${DATASETS}"
echo "horizons=${HORIZONS}"
echo "detector=improved seed=${SEED} temperature=${TEMP_T}"
echo "single protocol: E=1 K=1"
echo "MoE ablation: E=32 K=2"
echo ""

for dataset in ${DATASETS}; do
    for horizon in ${HORIZONS}; do
        ensure_pretrain "${dataset}" "${horizon}"
    done
done

for dataset in ${DATASETS}; do
    echo ""
    echo "================================================================="
    echo " ${dataset}"
    echo "================================================================="

    for horizon in ${HORIZONS}; do
        weight="./weights/patchtst_pretrained_${dataset}_H${horizon}.pth"
        if [ ! -f "${weight}" ]; then
            echo "  [SKIP CASE] ${dataset} H=${horizon}: missing ${weight}"
            continue
        fi

        # Main protocol: identical E=1 architecture, only update rule differs.
        run_eval "${dataset}" "${horizon}" frozen frozen 1 1
        run_eval "${dataset}" "${horizon}" full_ft full_ft 1 1
        run_eval "${dataset}" "${horizon}" single ours 1 1

        # MoE ablation only. It is not the default method.
        run_eval "${dataset}" "${horizon}" ours_e32 ours 32 2
    done
done

echo "dataset,horizon,method,mae,mse,rmse,status" > "${SUMMARY_CSV}"

for dataset in ${DATASETS}; do
    for horizon in ${HORIZONS}; do
        for method in frozen full_ft single ours_e32; do
            log="${LOGDIR}/${dataset}_H${horizon}_${method}.log"
            metrics=$(extract_metrics "${log}" 2>/dev/null)
            if [ -n "${metrics}" ]; then
                echo "${dataset},${horizon},${method},${metrics},OK" >> "${SUMMARY_CSV}"
            else
                echo "${dataset},${horizon},${method},,,,FAIL" >> "${SUMMARY_CSV}"
            fi
        done
    done
done

echo ""
echo "================================================================="
echo " Final Report"
echo "================================================================="
printf "%-10s %5s %10s %10s %10s %12s %12s\n" \
    "Dataset" "H" "Frozen" "Full-FT" "Single" "Single-v-F" "Single-v-E32"

for dataset in ${DATASETS}; do
    for horizon in ${HORIZONS}; do
        frozen_log="${LOGDIR}/${dataset}_H${horizon}_frozen.log"
        full_log="${LOGDIR}/${dataset}_H${horizon}_full_ft.log"
        single_log="${LOGDIR}/${dataset}_H${horizon}_single.log"
        e32_log="${LOGDIR}/${dataset}_H${horizon}_ours_e32.log"

        frozen=$(extract_metrics "${frozen_log}" 2>/dev/null | cut -d, -f2)
        full_ft=$(extract_metrics "${full_log}" 2>/dev/null | cut -d, -f2)
        single=$(extract_metrics "${single_log}" 2>/dev/null | cut -d, -f2)
        e32=$(extract_metrics "${e32_log}" 2>/dev/null | cut -d, -f2)

        if [ -n "${single}" ] && [ -n "${frozen}" ]; then
            delta_f=$(python3 -c "print(f'{(float(\"${single}\")-float(\"${frozen}\"))/float(\"${frozen}\")*100:+.1f}%')")
        else
            delta_f="N/A"
        fi

        if [ -n "${single}" ] && [ -n "${e32}" ]; then
            delta_e=$(python3 -c "print(f'{(float(\"${single}\")-float(\"${e32}\"))/float(\"${e32}\")*100:+.1f}%')")
        else
            delta_e="N/A"
        fi

        printf "%-10s %5s %10s %10s %10s %12s %12s\n" \
            "${dataset}" "${horizon}" "${frozen:-FAIL}" "${full_ft:-FAIL}" \
            "${single:-FAIL}" "${delta_f}" "${delta_e}"
    done
done

echo ""
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Logs: ${LOGDIR}/"
