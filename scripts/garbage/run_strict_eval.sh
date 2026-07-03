#!/bin/bash

# ==============================================================================
# 严格对照实验: 3-Mode Streaming Evaluation
# ==============================================================================
# 三组:
#   frozen  — 零在线更新 (Streaming-Frozen baseline)
#   full_ft — 每步全通道 GD (Online Fine-Tuning baseline)
#   ours    — Channel-Independent 选择性更新 (论文方法)
#
# 视界: H = 24, 48, 96
# 数据集: ECL (C=321)
# 权重: 原始 pretrain (accum_steps=1, no GA)
# 不用 GA — 5090 显存够
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/strict_eval
mkdir -p logs/monitor

echo "================================================================="
echo "[*] 严格对照实验: ECL 3-Mode × 3-Horizon"
echo "================================================================="

# ==============================================================================
# Step 0: Pretrain for H=24, 48, 96 (如果权重不存在)
# ==============================================================================

for H in 24 48 96; do
    WEIGHT_PATH="./weights/patchtst_pretrained_ECL_H${H}.pth"
    if [ -f "$WEIGHT_PATH" ]; then
        echo "[*] Pretrain weights for H=${H} already exist: ${WEIGHT_PATH}"
    else
        echo "[*] Pretraining H=${H}..."
        PYTHONPATH=. python pretrain.py \
            --root_path ${ROOT_PATH} --data_path ECL.csv \
            --features M --seq_len 96 --forecast_H ${H} --D_model ${D_MODEL} \
            --accum_steps 1 --epochs 10 \
            > logs/strict_eval/pretrain_H${H}.log 2>&1
        echo "    pretrain H=${H} done"
    fi
done

# ==============================================================================
# Step 1: Run 3 modes × 3 horizons = 9 experiments
# ==============================================================================

for H in 24 48 96; do
    echo ""
    echo "=== H=${H} ==="

    for MODE in frozen full_ft ours; do
        TAG="ECL_H${H}_${MODE}"
        echo "  [${TAG}] running..."

        PYTHONPATH=. python main.py \
            --root_path ${ROOT_PATH} --data_path ECL.csv \
            --features M --seq_len 96 --forecast_H ${H} --D_model ${D_MODEL} \
            --accum_steps 1 \
            --streaming_mode ${MODE} \
            --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
            --num_experts ${EXPERTS} --top_k ${TOP_K} \
            --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
            --experiment_tag ${TAG} \
            > logs/strict_eval/streaming_${TAG}.log 2>&1

        echo "    done"
    done
done

# ==============================================================================
# 汇总
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] 实验完成！结果汇总:"
echo "================================================================="
echo ""

printf "%-30s %10s %10s %10s\n" "Experiment" "MAE" "MSE" "RMSE"
printf "%-30s %10s %10s %10s\n" "------------------------------" "----------" "----------" "----------"

for H in 24 48 96; do
    for MODE in frozen full_ft ours; do
        TAG="ECL_H${H}_${MODE}"
        RESULT=$(grep "Completed" logs/strict_eval/streaming_${TAG}.log 2>/dev/null | tail -1)
        if [ -n "$RESULT" ]; then
            MAE=$(echo "$RESULT" | grep -oP 'MAE: \K[0-9.]+')
            MSE=$(echo "$RESULT" | grep -oP 'MSE: \K[0-9.]+')
            RMSE=$(echo "$RESULT" | grep -oP 'RMSE: \K[0-9.]+')
            printf "%-30s %10s %10s %10s\n" "$TAG" "$MAE" "$MSE" "$RMSE"
        else
            printf "%-30s %10s\n" "$TAG" "FAILED"
        fi
    done
    echo ""
done
