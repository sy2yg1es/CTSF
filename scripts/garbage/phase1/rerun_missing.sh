#!/bin/bash

# ==============================================================================
# 补跑脚本：H=384 GA=16 (中断重跑) + H=96 GA=16 Control Group (修复 utils)
# ==============================================================================
# 使用方法：bash scripts/rerun_missing.sh
# 日志输出：logs/ECL_bs_ablation/（覆盖原有残缺日志）
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

DATA_PATH="ECL.csv"
SEQ_LEN=96

# Streaming 核心参数（与原实验一致）
WINDOW_K=12
TAU=0.1
PATIENCE=2
TOP_K=2
LR=0.0001
L_AUX=0.0
WORKERS=4
EXPERTS=32

mkdir -p logs/ECL_bs_ablation

# ==============================================================================
# Part 1: H=384, GA=16 — 从头重跑（原始 Epoch 8 被 Ctrl+C 中断，权重不完整）
# ==============================================================================

echo "================================================================="
echo "[*] Part 1: H=384 | accum_steps=16 | Effective BS=48 | Epochs=40"
echo "================================================================="

echo ">>> [Phase 1] Pretrain H=384 GA=16 ..."
python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 384 \
    --D_model 384 \
    --accum_steps 16 \
    --epochs 40 > logs/ECL_bs_ablation/pretrain_H384_GA16.log 2>&1
echo "    - Pretrain 完成"

echo ">>> [Phase 2] Static Eval H=384 GA=16 ..."
python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 384 \
    --D_model 384 \
    --accum_steps 16 > logs/ECL_bs_ablation/static_H384_GA16.log 2>&1
echo "    - Static 完成"

echo ">>> [Phase 3] Streaming Eval H=384 GA=16 E=${EXPERTS} ..."
python main.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 384 \
    --D_model 384 \
    --accum_steps 16 \
    --window_K ${WINDOW_K} \
    --threshold_tau ${TAU} \
    --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} \
    --top_k ${TOP_K} \
    --learning_rate ${LR} \
    --l_aux_weight ${L_AUX} \
    --num_workers ${WORKERS} > logs/ECL_bs_ablation/streaming_H384_GA16_E${EXPERTS}.log 2>&1
echo "    - Streaming 完成"

# ==============================================================================
# Part 2: H=96, GA=16 — Control Group
# 修复方案：PYTHONPATH=. 让 Python 能找到项目根目录下的 utils/ 包
# 原因：layers/SelfAttention_Family.py 用了 `from utils.masking import ...`
#       这是 Time-Series-Library 的内部约定，需要从项目根运行
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] Part 2: H=96 | accum_steps=16 | Effective BS=48 | Epochs=40 (Control)"
echo "================================================================="

echo ">>> [Control Phase 1] Pretrain H=96 GA=16 ..."
PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 \
    --epochs 40 > logs/ECL_bs_ablation/pretrain_H96_GA16.log 2>&1
echo "    - Pretrain 完成"

echo ">>> [Control Phase 2] Static Eval H=96 GA=16 ..."
PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 > logs/ECL_bs_ablation/static_H96_GA16.log 2>&1
echo "    - Static 完成"

echo ">>> [Control Phase 3] Streaming Eval H=96 GA=16 E=${EXPERTS} ..."
PYTHONPATH=. python main.py \
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
echo "[*] 补跑完毕！快速查看结果："
echo "================================================================="
echo ""
echo "# Pretrain final loss:"
echo "grep 'Epoch.*Average Loss' logs/ECL_bs_ablation/pretrain_H384_GA16.log | tail -1"
echo "grep 'Epoch.*Average Loss' logs/ECL_bs_ablation/pretrain_H96_GA16.log | tail -1"
echo ""
echo "# Static + Streaming:"
echo "grep 'Completed' logs/ECL_bs_ablation/static_H384_GA16.log logs/ECL_bs_ablation/static_H96_GA16.log"
echo "grep 'Completed' logs/ECL_bs_ablation/streaming_H384_GA16_E32.log logs/ECL_bs_ablation/streaming_H96_GA16_E32.log"
