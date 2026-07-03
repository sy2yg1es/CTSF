#!/bin/bash

# ==============================================================================
# P0 Sanity Check: 验证重构后 PatchTST 路径输出与之前一致
# ==============================================================================
# 用 ECL H=96 的现有权重，跑 frozen/ours 两个 mode
# 对照之前的结果:
#   frozen:  MSE=0.3217
#   full_ft: MSE=0.2696
#   ours:    MSE=0.2687 (之前跑过的精确值)
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/p0_sanity

echo "================================================================="
echo "[*] P0 Sanity Check: backbone-agnostic refactor validation"
echo "================================================================="

# ECL H=96, frozen — 应该给出 MSE≈0.3217
echo "[1/3] ECL H=96 frozen..."
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --streaming_mode frozen \
    --backbone patchtst \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag p0_ECL_H96_frozen \
    > logs/p0_sanity/ECL_H96_frozen.log 2>&1
echo "    done"

# ECL H=96, ours — 应该给出 MSE≈0.2687
echo "[2/3] ECL H=96 ours..."
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --streaming_mode ours \
    --backbone patchtst \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag p0_ECL_H96_ours \
    > logs/p0_sanity/ECL_H96_ours.log 2>&1
echo "    done"

# ETTh1 H=24, ours — 应该给出 MSE≈0.3550
echo "[3/3] ETTh1 H=24 ours..."
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ETTh1.csv \
    --features M --seq_len 96 --forecast_H 24 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --streaming_mode ours \
    --backbone patchtst \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag p0_ETTh1_H24_ours \
    > logs/p0_sanity/ETTh1_H24_ours.log 2>&1
echo "    done"

echo ""
echo "=== P0 Results (should match previous values) ==="
echo ""
echo "--- Expected ---"
echo "ECL H=96 frozen:  MSE≈0.3217"
echo "ECL H=96 ours:    MSE≈0.2687"
echo "ETTh1 H=24 ours:  MSE≈0.3550"
echo ""
echo "--- Actual ---"
grep "Completed" logs/p0_sanity/*.log
