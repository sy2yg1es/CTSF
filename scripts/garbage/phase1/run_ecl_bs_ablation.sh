#!/bin/bash

# ==============================================================================
# Gradient Accumulation Ablation: ECL H=384 优化稳定性验证
# ==============================================================================
#
# 核心假设：
#   "大模型在 streaming forecasting 中失败的根因是优化稳定性，而非表征能力"
#
# 当前状况：ECL C=321, adaptive BS=3, H=384 严重欠收敛
# 实验设计：通过 GA 提升 effective BS，观察 H=384 能否恢复性能
#
# epochs 缩放策略：epochs × sqrt(accum_steps)
#   accum=1  → epochs=10
#   accum=4  → epochs=20
#   accum=8  → epochs=30
#   accum=16 → epochs=40
# ==============================================================================

# 基础数据路径
if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

DATA_PATH="ECL.csv"
SEQ_LEN=96
FORECAST_H=384

# Streaming 核心参数 (与 run_ecl.sh 保持一致)
WINDOW_K=12
TAU=0.1
PATIENCE=2
TOP_K=2
LR=0.0001
L_AUX=0.0
WORKERS=4
EXPERTS=32

# 创建日志目录
mkdir -p logs/ECL_bs_ablation

echo "================================================================="
echo "[*] GA Ablation Experiment: ECL H=${FORECAST_H}"
echo "[*] 验证假设: 大模型失败 = 优化不稳定 or 表征能力不足?"
echo "================================================================="

# ==============================================================================
# 主实验：H=384, accum_steps = 1 / 4 / 8 / 16
# ==============================================================================

declare -A ACCUM_EPOCHS
ACCUM_EPOCHS[1]=10
ACCUM_EPOCHS[4]=20
ACCUM_EPOCHS[8]=30
ACCUM_EPOCHS[16]=40

for ACCUM in 1 4 8 16; do
    EPOCHS=${ACCUM_EPOCHS[$ACCUM]}
    EFF_BS=$((3 * ACCUM))
    
    echo ""
    echo "================================================================="
    echo "[*] H=384 | accum_steps=${ACCUM} | Effective BS=${EFF_BS} | Epochs=${EPOCHS}"
    echo "================================================================="
    
    # --- Phase 1: Pretrain ---
    echo ">>> [Phase 1] Pretrain H=384 accum=${ACCUM} ..."
    python pretrain.py \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --features M \
        --seq_len ${SEQ_LEN} \
        --forecast_H ${FORECAST_H} \
        --D_model 384 \
        --accum_steps ${ACCUM} \
        --epochs ${EPOCHS} > logs/ECL_bs_ablation/pretrain_H384_GA${ACCUM}.log 2>&1
    echo "    - Pretrain 完成"

    # --- Phase 2: Static Baseline ---
    echo ">>> [Phase 2] Static Eval H=384 accum=${ACCUM} ..."
    python eval_static.py \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --features M \
        --seq_len ${SEQ_LEN} \
        --forecast_H ${FORECAST_H} \
        --D_model 384 \
        --accum_steps ${ACCUM} > logs/ECL_bs_ablation/static_H384_GA${ACCUM}.log 2>&1
    echo "    - Static 完成"

    # --- Phase 3: Streaming ---
    echo ">>> [Phase 3] Streaming Eval H=384 accum=${ACCUM} E=${EXPERTS} ..."
    python main.py \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --features M \
        --seq_len ${SEQ_LEN} \
        --forecast_H ${FORECAST_H} \
        --D_model 384 \
        --accum_steps ${ACCUM} \
        --window_K ${WINDOW_K} \
        --threshold_tau ${TAU} \
        --patience_C ${PATIENCE} \
        --num_experts ${EXPERTS} \
        --top_k ${TOP_K} \
        --learning_rate ${LR} \
        --l_aux_weight ${L_AUX} \
        --num_workers ${WORKERS} > logs/ECL_bs_ablation/streaming_H384_GA${ACCUM}_E${EXPERTS}.log 2>&1
    echo "    - Streaming 完成"
done

# ==============================================================================
# Control Group: H=96 + GA-16 (验证小模型对 BS 不敏感)
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] Control Group: H=96 | accum_steps=16 | Effective BS=48 | Epochs=40"
echo "================================================================="

# --- Pretrain ---
echo ">>> [Control] Pretrain H=96 accum=16 ..."
python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 \
    --epochs 40 > logs/ECL_bs_ablation/pretrain_H96_GA16.log 2>&1
echo "    - Pretrain 完成"

# --- Static ---
echo ">>> [Control] Static Eval H=96 accum=16 ..."
python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 > logs/ECL_bs_ablation/static_H96_GA16.log 2>&1
echo "    - Static 完成"

# --- Streaming ---
echo ">>> [Control] Streaming Eval H=96 accum=16 E=${EXPERTS} ..."
python main.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 \
    --window_K ${WINDOW_K} \
    --threshold_tau ${TAU} \
    --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} \
    --top_k ${TOP_K} \
    --learning_rate ${LR} \
    --l_aux_weight ${L_AUX} \
    --num_workers ${WORKERS} > logs/ECL_bs_ablation/streaming_H96_GA16_E${EXPERTS}.log 2>&1
echo "    - Streaming 完成"

echo ""
echo "================================================================="
echo "[*] 🎉 GA Ablation 全部实验完毕！日志在 ./logs/ECL_bs_ablation/"
echo "================================================================="
echo ""
echo "=== 结果汇总命令 ==="
echo "grep 'Average Loss\|Effective Batch\|Optimizer Steps' logs/ECL_bs_ablation/pretrain_*.log"
echo "grep 'MAE\|MSE\|RMSE\|Drift Updates' logs/ECL_bs_ablation/streaming_*.log"
echo "grep 'param_delta' logs/ECL_bs_ablation/streaming_*.log | tail -20"
