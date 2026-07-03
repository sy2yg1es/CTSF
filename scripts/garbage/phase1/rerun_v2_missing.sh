#!/bin/bash

# ==============================================================================
# 补跑：ECL v2 H96 GA16 (从 Epoch 33 续跑) + Traffic v2 全部
# ==============================================================================
# ECL v2 已完成：H384 GA1/8/16 全部 OK
# 缺失：
#   1. ECL H96 GA16 — pretrain 断在 Epoch 32，需要从头重跑 (无 checkpoint)
#   2. Traffic v2 — 全部未跑
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/ECL_bs_ablation_v2
mkdir -p logs/traffic_ga_ablation_v2

# ==============================================================================
# Part 1: ECL H=96 GA=16 control (从头跑 40 epochs)
# ==============================================================================

echo "==================================================================="
echo "[1/3] ECL Control: forecast_H=96 | D_model=512 | GA=16 | Epochs=40"
echo "==================================================================="

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 \
    --D_model ${D_MODEL} --accum_steps 16 --epochs 40 \
    > logs/ECL_bs_ablation_v2/pretrain_H96_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 \
    --D_model ${D_MODEL} --accum_steps 16 \
    > logs/ECL_bs_ablation_v2/static_H96_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 \
    --D_model ${D_MODEL} --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/ECL_bs_ablation_v2/streaming_H96_GA16_E${EXPERTS}.log 2>&1
echo "    streaming done"

# ==============================================================================
# Part 2: Traffic forecast_H=192 GA=16 (原始最优 + GA)
# ==============================================================================

echo ""
echo "==================================================================="
echo "[2/3] Traffic: forecast_H=192 | D_model=512 | GA=16 | Epochs=40"
echo "==================================================================="

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 \
    --D_model ${D_MODEL} --accum_steps 16 --epochs 40 \
    > logs/traffic_ga_ablation_v2/pretrain_H192_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 \
    --D_model ${D_MODEL} --accum_steps 16 \
    > logs/traffic_ga_ablation_v2/static_H192_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 \
    --D_model ${D_MODEL} --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_ga_ablation_v2/streaming_H192_GA16_E${EXPERTS}.log 2>&1
echo "    streaming done"

# ==============================================================================
# Part 3: Traffic forecast_H=384 GA=16 (原始最差 + GA，验证假设)
# ==============================================================================

echo ""
echo "==================================================================="
echo "[3/3] Traffic: forecast_H=384 | D_model=512 | GA=16 | Epochs=40"
echo "==================================================================="

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 \
    --D_model ${D_MODEL} --accum_steps 16 --epochs 40 \
    > logs/traffic_ga_ablation_v2/pretrain_H384_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 \
    --D_model ${D_MODEL} --accum_steps 16 \
    > logs/traffic_ga_ablation_v2/static_H384_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 \
    --D_model ${D_MODEL} --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_ga_ablation_v2/streaming_H384_GA16_E${EXPERTS}.log 2>&1
echo "    streaming done"

echo ""
echo "==================================================================="
echo "[*] 补跑完成！"
echo "==================================================================="
echo ""
echo "=== ECL v2 完整结果 ==="
grep "Completed" logs/ECL_bs_ablation_v2/static_H*.log
grep "Completed" logs/ECL_bs_ablation_v2/streaming_H*_E${EXPERTS}.log
echo ""
echo "=== Traffic v2 结果 ==="
grep "Completed" logs/traffic_ga_ablation_v2/static_H*.log
grep "Completed" logs/traffic_ga_ablation_v2/streaming_H*_E${EXPERTS}.log
