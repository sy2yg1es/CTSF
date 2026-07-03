#!/bin/bash

# ==============================================================================
# 最终严格对照实验: 4-Horizon × 3-Mode × 2-Dataset
# ==============================================================================
# H = {1, 24, 48, 96}
# Modes: frozen / full_ft / ours
# Datasets: ECL (C=321), Traffic (C=862)
# 不用 GA — 5090 直跑
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

LOGDIR="logs/final_eval"
mkdir -p ${LOGDIR}
mkdir -p logs/monitor

# ==============================================================================
# Pretrain: 确保所有 H 的权重存在
# ==============================================================================

echo "================================================================="
echo "[Phase 0] Pretrain 权重检查"
echo "================================================================="

# Dataset name → CSV filename mapping
DATASETS="ECL Traffic ETTh1 ETTh2 ETTm1 ETTm2 WTH"

get_csv() {
    echo "${1}.csv"
}

for DATASET in ${DATASETS}; do
    DATA_FILE=$(get_csv ${DATASET})
    for H in 1 24 48 96; do
        WEIGHT="./weights/patchtst_pretrained_${DATASET}_H${H}.pth"
        if [ -f "$WEIGHT" ]; then
            echo "  [OK] ${DATASET} H=${H}"
        else
            echo "  [TRAIN] ${DATASET} H=${H} ..."
            PYTHONPATH=. python pretrain.py \
                --root_path ${ROOT_PATH} --data_path ${DATA_FILE} \
                --features M --seq_len 96 --forecast_H ${H} --D_model ${D_MODEL} \
                --accum_steps 1 --epochs 10 \
                > ${LOGDIR}/pretrain_${DATASET}_H${H}.log 2>&1
            echo "    done"
        fi
    done
done

# ==============================================================================
# Main Experiments
# ==============================================================================

echo ""
echo "================================================================="
echo "[Phase 1] Streaming 3-Mode × 4-Horizon × 2-Dataset"
echo "================================================================="

for DATASET in ${DATASETS}; do
    DATA_FILE=$(get_csv ${DATASET})

    echo ""
    echo "--- ${DATASET} ---"

    for H in 1 24 48 96; do
        for MODE in frozen full_ft ours; do
            TAG="${DATASET}_H${H}_${MODE}"
            echo "  [${TAG}] ..."

            PYTHONPATH=. python main.py \
                --root_path ${ROOT_PATH} --data_path ${DATA_FILE} \
                --features M --seq_len 96 --forecast_H ${H} --D_model ${D_MODEL} \
                --accum_steps 1 \
                --streaming_mode ${MODE} \
                --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
                --num_experts ${EXPERTS} --top_k ${TOP_K} \
                --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
                --experiment_tag ${TAG} \
                > ${LOGDIR}/streaming_${TAG}.log 2>&1

            echo "    done"
        done
    done
done

# ==============================================================================
# 战报输出
# ==============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                          FINAL EVALUATION REPORT                                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

for DATASET in ${DATASETS}; do
    echo "┌──────────────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Dataset: ${DATASET}                                                                  │"
    echo "├──────────┬──────────┬──────────┬──────────┬────────────┬──────────┬──────────────────┤"
    printf "│ %-8s │ %-8s │ %-8s │ %-8s │ %-10s │ %-8s │ %-16s │\n" \
           "H" "Mode" "MAE" "MSE" "RMSE" "Updates" "Avg Ch Ratio"
    echo "├──────────┼──────────┼──────────┼──────────┼────────────┼──────────┼──────────────────┤"

    for H in 1 24 48 96; do
        for MODE in frozen full_ft ours; do
            TAG="${DATASET}_H${H}_${MODE}"
            LOG="${LOGDIR}/streaming_${TAG}.log"

            MAE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'MAE: \K[0-9.]+' | tail -1)
            MSE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'MSE: \K[0-9.]+' | tail -1)
            RMSE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'RMSE: \K[0-9.]+' | tail -1)
            UPD=$(grep "Updates:" "$LOG" 2>/dev/null | grep -oP 'Updates: \K[0-9]+/[0-9]+' | tail -1)
            CHR=$(grep "Avg Ch Ratio" "$LOG" 2>/dev/null | grep -oP 'Avg Ch Ratio: \K[0-9.]+%' | tail -1)

            if [ -z "$MAE" ]; then
                printf "│ %-8s │ %-8s │ %-8s │ %-8s │ %-10s │ %-8s │ %-16s │\n" \
                       "$H" "$MODE" "FAIL" "-" "-" "-" "-"
            else
                printf "│ %-8s │ %-8s │ %-8s │ %-8s │ %-10s │ %-8s │ %-16s │\n" \
                       "$H" "$MODE" "$MAE" "$MSE" "$RMSE" "${UPD:-0/0}" "${CHR:-0.0%}"
            fi
        done
        echo "├──────────┼──────────┼──────────┼──────────┼────────────┼──────────┼──────────────────┤"
    done

    echo "└──────────┴──────────┴──────────┴──────────┴────────────┴──────────┴──────────────────┘"
    echo ""
done

echo "[*] All logs saved to ${LOGDIR}/"
echo "[*] Monitor JSONs saved to logs/monitor/"
