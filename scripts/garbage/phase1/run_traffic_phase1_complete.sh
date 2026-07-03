#!/bin/bash

# ==============================================================================
# Traffic Phase 1 完整版 — 跨数据集验证 Optimization Collapse 假设
# ==============================================================================
#
# 关键修正：epochs 固定为 10（和原始实验一致），只改 GA
# 这样每个配置看到的数据量完全相同（10 passes），
# 唯一的变量是梯度质量（eff BS = 1 vs 8 vs 16）
#
# 原始 baseline（BS=1, 10 epochs, D_model=512）：
#   H=96:  pretrain 0.6364 | static MSE 0.6733 | streaming MSE 0.6718
#   H=192: pretrain 0.3327 | static MSE 0.3572 | streaming MSE 0.3525
#   H=384: pretrain 0.3437 | static MSE 0.4165 | streaming MSE 0.4224
#
# 如果 H=384 GA=8 的 static/streaming MSE 降到接近 H=192 → 假设跨数据集成立
# 如果没降或更差 → Traffic 的问题是 Layer 2（过参数化），不是 Layer 1
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
EPOCHS=10    # 和原始实验完全一致，不做 epoch scaling

WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/traffic_phase1_complete

echo "================================================================="
echo "[*] Traffic Phase 1 完整版: 固定 epochs=10，只变 GA"
echo "[*] C=862, adaptive BS=1, D_model=512"
echo "================================================================="

# ==============================================================================
# Exp 1: H=384 GA=8 (核心验证)
# 对比: 原始 H=384 static 0.4165 / streaming 0.4224
# ==============================================================================

echo ""
echo "[1/5] H=384 | GA=8 | Eff BS=8 | Epochs=10 (核心验证)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 --epochs ${EPOCHS} \
    > logs/traffic_phase1_complete/pretrain_H384_GA8.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    > logs/traffic_phase1_complete/static_H384_GA8.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_phase1_complete/streaming_H384_GA8.log 2>&1
echo "    streaming done"

# ==============================================================================
# Exp 2: H=384 GA=16 (看 GA 梯度继续扩大的边际效应)
# ==============================================================================

echo ""
echo "[2/5] H=384 | GA=16 | Eff BS=16 | Epochs=10"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 16 --epochs ${EPOCHS} \
    > logs/traffic_phase1_complete/pretrain_H384_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 16 \
    > logs/traffic_phase1_complete/static_H384_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_phase1_complete/streaming_H384_GA16.log 2>&1
echo "    streaming done"

# ==============================================================================
# Exp 3: H=192 GA=8 (最优配置+GA，看能否进一步提升)
# 对比: 原始 H=192 static 0.3572 / streaming 0.3525
# ==============================================================================

echo ""
echo "[3/5] H=192 | GA=8 | Eff BS=8 | Epochs=10 (最优+GA)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 8 --epochs ${EPOCHS} \
    > logs/traffic_phase1_complete/pretrain_H192_GA8.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 8 \
    > logs/traffic_phase1_complete/static_H192_GA8.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_phase1_complete/streaming_H192_GA8.log 2>&1
echo "    streaming done"

# ==============================================================================
# Exp 4: H=96 GA=8 (control: 表征瓶颈 → GA 应该无效)
# 对比: 原始 H=96 static 0.6733 / streaming 0.6718
# ==============================================================================

echo ""
echo "[4/5] H=96  | GA=8 | Eff BS=8 | Epochs=10 (control)"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 8 --epochs ${EPOCHS} \
    > logs/traffic_phase1_complete/pretrain_H96_GA8.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 8 \
    > logs/traffic_phase1_complete/static_H96_GA8.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_phase1_complete/streaming_H96_GA8.log 2>&1
echo "    streaming done"

# ==============================================================================
# Exp 5: H=192 GA=16 (看最优配置的 GA 边际效应)
# ==============================================================================

echo ""
echo "[5/5] H=192 | GA=16 | Eff BS=16 | Epochs=10"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 16 --epochs ${EPOCHS} \
    > logs/traffic_phase1_complete/pretrain_H192_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 16 \
    > logs/traffic_phase1_complete/static_H192_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_phase1_complete/streaming_H192_GA16.log 2>&1
echo "    streaming done"

# ==============================================================================
# 汇总
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] Phase 1 完成！"
echo "================================================================="
echo ""
echo "=== Pretrain Final Loss ==="
grep "Epoch 10 Average" logs/traffic_phase1_complete/pretrain_H*.log
echo ""
echo "=== Static Baseline ==="
grep "Completed" logs/traffic_phase1_complete/static_H*.log
echo ""
echo "=== Streaming ==="
grep "Completed" logs/traffic_phase1_complete/streaming_H*.log
