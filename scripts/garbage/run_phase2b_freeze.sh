#!/bin/bash

# ==============================================================================
# Phase 2b: Adapter Ablation — Freeze Prompt
# ==============================================================================
# P1: freeze prompt, no adapter → 确认 prompt 在线更新的重要性
# P2: freeze prompt, adapter=32  → adapter 能否替代 prompt 成为唯一在线通道
#
# 对照组：
#   ECL H=384 原始权重, prompt 正常更新 → streaming MSE ~0.289
#   ECL H=384 GA=8 权重, prompt 正常更新 → streaming MSE ~0.217
#   Traffic H=384 原始权重, prompt 正常更新 → streaming MSE ~0.422
#
# 不用 GA（5090 显存够）
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/phase2b
mkdir -p logs/monitor

echo "================================================================="
echo "[*] Phase 2b: Freeze Prompt Ablation"
echo "================================================================="

# ==============================
# ECL — 用 GA=8 权重 (优化修复后的好权重)
# ==============================

echo ""
echo "--- ECL H=384 (GA=8 weights) ---"

# P1: freeze prompt, no adapter
echo "[1/6] ECL | freeze_prompt | no adapter"
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --freeze_prompt \
    --bottleneck_dim 0 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag ECL_GA8_freeze_noadapter \
    > logs/phase2b/ECL_GA8_freeze_noadapter.log 2>&1
echo "    done"

# P2: freeze prompt, adapter=32
echo "[2/6] ECL | freeze_prompt | adapter=32"
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --freeze_prompt \
    --bottleneck_dim 32 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag ECL_GA8_freeze_adapter32 \
    > logs/phase2b/ECL_GA8_freeze_adapter32.log 2>&1
echo "    done"

# ==============================
# Traffic — 用原始权重 (accum_steps=1)
# ==============================

echo ""
echo "--- Traffic H=384 (original weights) ---"

# P1: freeze prompt, no adapter
echo "[3/6] Traffic | freeze_prompt | no adapter"
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --freeze_prompt \
    --bottleneck_dim 0 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag Traffic_freeze_noadapter \
    > logs/phase2b/Traffic_freeze_noadapter.log 2>&1
echo "    done"

# P2: freeze prompt, adapter=32
echo "[4/6] Traffic | freeze_prompt | adapter=32"
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --freeze_prompt \
    --bottleneck_dim 32 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag Traffic_freeze_adapter32 \
    > logs/phase2b/Traffic_freeze_adapter32.log 2>&1
echo "    done"

# ==============================
# 额外对照: adapter=64 (看 bottleneck 宽度在 freeze 模式下的影响)
# ==============================

echo ""
echo "--- Extra: adapter=64 ---"

echo "[5/6] ECL | freeze_prompt | adapter=64"
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 8 \
    --freeze_prompt \
    --bottleneck_dim 64 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag ECL_GA8_freeze_adapter64 \
    > logs/phase2b/ECL_GA8_freeze_adapter64.log 2>&1
echo "    done"

echo "[6/6] Traffic | freeze_prompt | adapter=64"
PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --freeze_prompt \
    --bottleneck_dim 64 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    --experiment_tag Traffic_freeze_adapter64 \
    > logs/phase2b/Traffic_freeze_adapter64.log 2>&1
echo "    done"

# ==============================================================================
# 汇总
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] Phase 2b 完成！"
echo "================================================================="
echo ""
echo "=== Results ==="
grep "Completed" logs/phase2b/*.log
echo ""
echo "=== Reference (Phase 2a, prompt unfrozen) ==="
echo "ECL GA=8 no adapter:    MSE=0.2168"
echo "ECL GA=8 adapter=32:    MSE=0.2168"
echo "Traffic no adapter:     MSE=0.4618"
echo "Traffic adapter=32:     MSE=0.4614"
echo ""
echo "=== Monitor Logs ==="
ls -la logs/monitor/monitor_*freeze*.json 2>/dev/null
