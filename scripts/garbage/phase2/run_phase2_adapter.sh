#!/bin/bash

# ==============================================================================
# Phase 2: Bottleneck Adapter 实验
# ==============================================================================
# 实验矩阵：
#   1. ECL H=384 GA=8 无 adapter (baseline, 已有数据但重跑以获取监控指标)
#   2. ECL H=384 GA=8 + adapter=32
#   3. ECL H=384 GA=8 + adapter=64
#   4. Traffic H=384 无 GA 无 adapter (baseline with monitoring)
#   5. Traffic H=384 无 GA + adapter=32 (adapter 作为显式正则替代 BS=1 隐式正则)
#
# 所有实验使用已有的 pretrain 权重（不重新 pretrain）
# 只跑 streaming（adapter 只影响在线更新阶段）
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/phase2
mkdir -p logs/monitor

echo "================================================================="
echo "[*] Phase 2: Bottleneck Adapter Experiment"
echo "================================================================="

# ==============================================================================
# Exp 1: ECL H=384 GA=8 无 adapter (baseline with monitoring)
# 使用 v2 的 GA=8 权重
# ==============================================================================

echo ""
echo "[1/5] ECL H=384 | GA=8 | no adapter (baseline + monitoring)"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --bottleneck_dim 0 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag ECL_H384_GA8_noadapter \
    > logs/phase2/streaming_ECL_H384_GA8_noadapter.log 2>&1
echo "    done"

# ==============================================================================
# Exp 2: ECL H=384 GA=8 + adapter=32
# ==============================================================================

echo ""
echo "[2/5] ECL H=384 | GA=8 | adapter=32"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --bottleneck_dim 32 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag ECL_H384_GA8_adapter32 \
    > logs/phase2/streaming_ECL_H384_GA8_adapter32.log 2>&1
echo "    done"

# ==============================================================================
# Exp 3: ECL H=384 GA=8 + adapter=64
# ==============================================================================

echo ""
echo "[3/5] ECL H=384 | GA=8 | adapter=64"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --bottleneck_dim 64 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag ECL_H384_GA8_adapter64 \
    > logs/phase2/streaming_ECL_H384_GA8_adapter64.log 2>&1
echo "    done"

# ==============================================================================
# Exp 4: Traffic H=384 无 GA 无 adapter (baseline with monitoring)
# 使用原始权重 (accum_steps=1)
# ==============================================================================

echo ""
echo "[4/5] Traffic H=384 | no GA | no adapter (baseline + monitoring)"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --bottleneck_dim 0 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag Traffic_H384_noadapter \
    > logs/phase2/streaming_Traffic_H384_noadapter.log 2>&1
echo "    done"

# ==============================================================================
# Exp 5: Traffic H=384 无 GA + adapter=32
# adapter 作为显式正则化，替代 BS=1 的隐式正则
# ==============================================================================

echo ""
echo "[5/5] Traffic H=384 | no GA | adapter=32 (explicit regularization)"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --bottleneck_dim 32 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag Traffic_H384_adapter32 \
    > logs/phase2/streaming_Traffic_H384_adapter32.log 2>&1
echo "    done"

# ==============================================================================
# 汇总
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] Phase 2 完成！"
echo "================================================================="
echo ""
echo "=== Streaming Results ==="
grep "Completed" logs/phase2/streaming_*.log
echo ""
echo "=== Monitor Logs ==="
ls -la logs/monitor/monitor_*.json 2>/dev/null
