#!/bin/bash

# ==============================================================================
# Traffic Phase 1 - 剩余实验 (3/5, 4/5, 5/5)
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
EPOCHS=10    

WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/traffic_phase1_complete

echo "================================================================="
echo "[*] 继续执行剩余实验: Exp 3, Exp 4, Exp 5"
echo "================================================================="

# ==============================================================================
# Exp 3: H=192 GA=8 (最优配置+GA，看能否进一步提升)
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
# 汇总 (也会把前两个实验的日志一起抓出来)
# ==============================================================================
echo ""
echo "================================================================="
echo "[*] Phase 1 剩余部分完成！"
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