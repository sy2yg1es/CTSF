#!/bin/bash

# ==============================================================================
# Traffic GA Ablation: 跨数据集验证"大模型失败根因"假设
# ==============================================================================
# 背景：Traffic C=862, adaptive BS=1 (比 ECL 的 BS=3 更极端)
#
# 研究问题：ECL 上 GA 修复了 H=384 的 17% MSE 退化。Traffic 上是否相同？
#
# 实验设计：
#   - H=192 GA=1  (baseline复现，确认结果一致)
#   - H=192 GA=16 (最优模型+GA，能否进一步提升？)
#   - H=384 GA=16 (大模型+GA，能否追上 H=192？)
#   - H=96  GA=16 (control: 表征瓶颈≠优化问题，GA 无法修复)
#
# epochs 缩放策略：epochs × sqrt(accum_steps)
#   accum=1  → epochs=10
#   accum=16 → epochs=40
#
# 预期结论：
#   - H=384 GA=16 相比 H=384 GA=1 有明显改善  → 优化假设在 Traffic 上也成立
#   - H=96  GA=16 相比 H=96  GA=1 无明显改善  → 表征瓶颈不是 BS 问题
#   - H=384 GA=16 仍不如 H=192 GA=16          → ECL 的两层病因在 Traffic 也存在
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

DATA_PATH="Traffic.csv"
SEQ_LEN=96

WINDOW_K=12
TAU=0.1
PATIENCE=2
TOP_K=2
LR=0.0001
L_AUX=0.0
WORKERS=4
EXPERTS=32   # 沿用 ECL 实验设置，Traffic 上 E=64 仅比 E=32 好一点点

mkdir -p logs/traffic_ga_ablation

echo "================================================================="
echo "[*] Traffic GA Ablation: 跨数据集验证优化假设"
echo "[*] C=862, adaptive BS=1 (极端场景)"
echo "================================================================="

# ==============================================================================
# Exp 1: H=192 GA=1 (baseline 复现)
# 目的：确认复现结果与原始实验一致（sanity check）
# ==============================================================================

echo ""
echo "[Exp 1/4] H=192 | GA=1 | Eff BS=1 | Epochs=10  (baseline 复现)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 192 \
    --D_model 192 \
    --accum_steps 1 \
    --epochs 10 > logs/traffic_ga_ablation/pretrain_H192_GA1.log 2>&1
echo "    - Pretrain 完成"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 192 \
    --D_model 192 \
    --accum_steps 1 > logs/traffic_ga_ablation/static_H192_GA1.log 2>&1
echo "    - Static 完成"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 192 \
    --D_model 192 \
    --accum_steps 1 \
    --window_K ${WINDOW_K} \
    --threshold_tau ${TAU} \
    --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} \
    --top_k ${TOP_K} \
    --learning_rate ${LR} \
    --l_aux_weight ${L_AUX} \
    --num_workers ${WORKERS} > logs/traffic_ga_ablation/streaming_H192_GA1_E${EXPERTS}.log 2>&1
echo "    - Streaming 完成"

# ==============================================================================
# Exp 2: H=384 GA=16 (大模型+GA，核心验证实验)
# ==============================================================================

echo ""
echo "[Exp 2/4] H=384 | GA=16 | Eff BS=16 | Epochs=40  (核心验证)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 384 \
    --D_model 384 \
    --accum_steps 16 \
    --epochs 40 > logs/traffic_ga_ablation/pretrain_H384_GA16.log 2>&1
echo "    - Pretrain 完成"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 384 \
    --D_model 384 \
    --accum_steps 16 > logs/traffic_ga_ablation/static_H384_GA16.log 2>&1
echo "    - Static 完成"

PYTHONPATH=. python main.py \
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
    --num_workers ${WORKERS} > logs/traffic_ga_ablation/streaming_H384_GA16_E${EXPERTS}.log 2>&1
echo "    - Streaming 完成"

# ==============================================================================
# Exp 3: H=192 GA=16 (最优模型+GA，能否再上一层楼？)
# ==============================================================================

echo ""
echo "[Exp 3/4] H=192 | GA=16 | Eff BS=16 | Epochs=40  (最优模型+GA)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 192 \
    --D_model 192 \
    --accum_steps 16 \
    --epochs 40 > logs/traffic_ga_ablation/pretrain_H192_GA16.log 2>&1
echo "    - Pretrain 完成"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 192 \
    --D_model 192 \
    --accum_steps 16 > logs/traffic_ga_ablation/static_H192_GA16.log 2>&1
echo "    - Static 完成"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 192 \
    --D_model 192 \
    --accum_steps 16 \
    --window_K ${WINDOW_K} \
    --threshold_tau ${TAU} \
    --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} \
    --top_k ${TOP_K} \
    --learning_rate ${LR} \
    --l_aux_weight ${L_AUX} \
    --num_workers ${WORKERS} > logs/traffic_ga_ablation/streaming_H192_GA16_E${EXPERTS}.log 2>&1
echo "    - Streaming 完成"

# ==============================================================================
# Exp 4: H=96 GA=16 (control: 表征瓶颈，GA 无法修复)
# 预期：GA 不会大幅改善 H=96，因为问题是表征容量不足而非优化不稳定
# ==============================================================================

echo ""
echo "[Exp 4/4] H=96  | GA=16 | Eff BS=16 | Epochs=40  (control: 表征瓶颈)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 \
    --epochs 40 > logs/traffic_ga_ablation/pretrain_H96_GA16.log 2>&1
echo "    - Pretrain 完成"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --features M \
    --seq_len ${SEQ_LEN} \
    --forecast_H 96 \
    --D_model 96 \
    --accum_steps 16 > logs/traffic_ga_ablation/static_H96_GA16.log 2>&1
echo "    - Static 完成"

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
    --num_workers ${WORKERS} > logs/traffic_ga_ablation/streaming_H96_GA16_E${EXPERTS}.log 2>&1
echo "    - Streaming 完成"

# ==============================================================================
# 汇总
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] 所有实验完成！日志在 ./logs/traffic_ga_ablation/"
echo "================================================================="
echo ""
echo "=== 快速汇总 ==="
echo "# Pretrain final loss:"
grep "Epoch.*Average Loss" logs/traffic_ga_ablation/pretrain_H192_GA1.log  | tail -1
grep "Epoch.*Average Loss" logs/traffic_ga_ablation/pretrain_H384_GA16.log | tail -1
grep "Epoch.*Average Loss" logs/traffic_ga_ablation/pretrain_H192_GA16.log | tail -1
grep "Epoch.*Average Loss" logs/traffic_ga_ablation/pretrain_H96_GA16.log  | tail -1
echo ""
echo "# Static + Streaming:"
grep "Completed" logs/traffic_ga_ablation/static_H*.log
grep "Completed" logs/traffic_ga_ablation/streaming_H*_E${EXPERTS}.log
