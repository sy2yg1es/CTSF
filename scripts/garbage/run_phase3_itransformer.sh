#!/bin/bash

# ==============================================================================
# Phase 3 P1: iTransformer × 3-Mode × 4-Horizon × 7-Dataset
# ==============================================================================
# 目标：验证 CI-MoE 插件在 iTransformer backbone 上同样有效
# 对照：PatchTST 已有结果 (logs/final_eval/)
#
# 实验矩阵: 7 数据集 × 4 视界 × 3 模式 = 84 个实验
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

LOGDIR="logs/phase3_itransformer"
mkdir -p ${LOGDIR}
mkdir -p logs/monitor

DATASETS="ECL Traffic ETTh1 ETTh2 ETTm1 ETTm2 WTH"

echo "================================================================="
echo "[*] Phase 3 P1: iTransformer Backbone"
echo "================================================================="

# ==============================================================================
# Step 0: Pretrain iTransformer 权重 (如不存在)
# ==============================================================================

echo ""
echo "[Phase 0] iTransformer Pretrain 权重检查"

for DATASET in ${DATASETS}; do
    DATA_FILE="${DATASET}.csv"
    for H in 1 24 48 96; do
        WEIGHT="./weights/itransformer_pretrained_${DATASET}_H${H}.pth"
        if [ -f "$WEIGHT" ]; then
            echo "  [OK] iTransformer ${DATASET} H=${H}"
        else
            echo "  [TRAIN] iTransformer ${DATASET} H=${H} ..."
            PYTHONPATH=. python pretrain.py \
                --root_path ${ROOT_PATH} --data_path ${DATA_FILE} \
                --features M --seq_len 96 --forecast_H ${H} --D_model ${D_MODEL} \
                --accum_steps 1 --epochs 10 \
                --backbone itransformer \
                --max_effective_bs 20000 \
                > ${LOGDIR}/pretrain_${DATASET}_H${H}.log 2>&1
            echo "    done"
        fi
    done
done

# ==============================================================================
# Step 1: 3-Mode Streaming Evaluation
# ==============================================================================

echo ""
echo "[Phase 1] iTransformer 3-Mode × 4-Horizon × 7-Dataset"
echo "================================================================="

for DATASET in ${DATASETS}; do
    DATA_FILE="${DATASET}.csv"
    echo ""
    echo "--- iTransformer | ${DATASET} ---"

    for H in 1 24 48 96; do
        for MODE in frozen full_ft ours; do
            TAG="iT_${DATASET}_H${H}_${MODE}"
            echo "  [${TAG}] ..."

            PYTHONPATH=. python main.py \
                --root_path ${ROOT_PATH} --data_path ${DATA_FILE} \
                --features M --seq_len 96 --forecast_H ${H} --D_model ${D_MODEL} \
                --accum_steps 1 \
                --backbone itransformer \
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
# 战报 — iTransformer
# ==============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════╗"
echo "║              Phase 3 P1 — iTransformer Results                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

printf "%-32s %8s %8s %8s %12s %10s\n" "Experiment" "MAE" "MSE" "RMSE" "Updates" "Ch Ratio"
printf "%-32s %8s %8s %8s %12s %10s\n" "$(printf '%0.s-' {1..32})" "--------" "--------" "--------" "------------" "----------"

for DATASET in ${DATASETS}; do
    for H in 1 24 48 96; do
        for MODE in frozen full_ft ours; do
            TAG="iT_${DATASET}_H${H}_${MODE}"
            LOG="${LOGDIR}/streaming_${TAG}.log"

            MAE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'MAE: \K[0-9.]+' | tail -1)
            MSE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'MSE: \K[0-9.]+' | tail -1)
            RMSE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'RMSE: \K[0-9.]+' | tail -1)
            UPD=$(grep "Updates:" "$LOG" 2>/dev/null | grep -oP 'Updates: \K[0-9]+/[0-9]+' | tail -1)
            CHR=$(grep "Avg Ch Ratio" "$LOG" 2>/dev/null | grep -oP 'Avg Ch Ratio: \K[0-9.]+%' | tail -1)

            if [ -z "$MAE" ]; then
                printf "%-32s %8s\n" "$TAG" "FAIL"
            else
                printf "%-32s %8s %8s %8s %12s %10s\n" \
                    "$TAG" "$MAE" "$MSE" "$RMSE" "${UPD:-N/A}" "${CHR:-N/A}"
            fi
        done
    done
    echo ""
done

echo "[*] All logs: ${LOGDIR}/"
