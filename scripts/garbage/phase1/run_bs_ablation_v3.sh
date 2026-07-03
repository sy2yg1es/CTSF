#!/bin/bash

# ==============================================================================
# GA Ablation v3 — RTX 5090 大显存直接加大 BS（不用 GA）
# ==============================================================================
# GA 和大 BS 效果完全等价，5090 显存够用时直接加大 BS 更简洁
#
# 关键参数：
#   --max_effective_bs 20000   允许 batch_size * C 最多 20000
#   --batch_size 32            请求 BS=32
#   --accum_steps 1            不用 GA
#
# 实际 adaptive BS：
#   ECL (C=321):     min(32, 20000//321) = min(32, 62) = 32   ← 原来是 3
#   Traffic (C=862): min(32, 20000//862) = min(32, 23) = 23   ← 原来是 1
#
# epochs 统一用原始的 10（不需要 sqrt 缩放，因为没有 GA）
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512
MAX_EBS=20000   # 5090 有 32GB，20000 足够覆盖所有数据集

WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

mkdir -p logs/ECL_bs_ablation_v3
mkdir -p logs/traffic_ga_ablation_v3

echo "================================================================="
echo "[*] BS Ablation v3: RTX 5090 直接大 BS，不用 GA"
echo "[*] max_effective_bs=${MAX_EBS}"
echo "================================================================="

# ==============================================================================
# ECL: forecast_H=384, D_model=512, 大 BS
# ==============================================================================

echo ""
echo "=== ECL forecast_H=384 | D_model=512 | BS=32 (large VRAM) ==="

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 \
    --D_model ${D_MODEL} \
    --batch_size 32 --accum_steps 1 --max_effective_bs ${MAX_EBS} \
    --epochs 10 \
    > logs/ECL_bs_ablation_v3/pretrain_H384.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    > logs/ECL_bs_ablation_v3/static_H384.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/ECL_bs_ablation_v3/streaming_H384_E${EXPERTS}.log 2>&1
echo "    streaming done"

# Control: ECL H=96 大 BS
echo ""
echo "--- ECL Control: forecast_H=96 | BS=32 ---"
PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --batch_size 32 --accum_steps 1 --max_effective_bs ${MAX_EBS} \
    --epochs 10 \
    > logs/ECL_bs_ablation_v3/pretrain_H96.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 1 \
    > logs/ECL_bs_ablation_v3/static_H96.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/ECL_bs_ablation_v3/streaming_H96_E${EXPERTS}.log 2>&1
echo "    streaming done"

# ==============================================================================
# Traffic: forecast_H=192 和 384, D_model=512, 大 BS
# ==============================================================================

echo ""
echo "=== Traffic forecast_H=192 | D_model=512 | BS=23 (max with C=862) ==="

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --batch_size 32 --accum_steps 1 --max_effective_bs ${MAX_EBS} \
    --epochs 10 \
    > logs/traffic_ga_ablation_v3/pretrain_H192.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 1 \
    > logs/traffic_ga_ablation_v3/static_H192.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_ga_ablation_v3/streaming_H192_E${EXPERTS}.log 2>&1
echo "    streaming done"

echo ""
echo "--- Traffic forecast_H=384 | BS=23 ---"
PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --batch_size 32 --accum_steps 1 --max_effective_bs ${MAX_EBS} \
    --epochs 10 \
    > logs/traffic_ga_ablation_v3/pretrain_H384.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    > logs/traffic_ga_ablation_v3/static_H384.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 1 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_ga_ablation_v3/streaming_H384_E${EXPERTS}.log 2>&1
echo "    streaming done"

echo ""
echo "================================================================="
echo "[*] 完成！"
echo "================================================================="
echo ""
echo "=== ECL v3 ==="
grep "Adaptive Batch Size" logs/ECL_bs_ablation_v3/pretrain_H384.log
grep "Completed" logs/ECL_bs_ablation_v3/static_H*.log
grep "Completed" logs/ECL_bs_ablation_v3/streaming_H*_E${EXPERTS}.log
echo ""
echo "=== Traffic v3 ==="
grep "Adaptive Batch Size" logs/traffic_ga_ablation_v3/pretrain_H192.log
grep "Completed" logs/traffic_ga_ablation_v3/static_H*.log
grep "Completed" logs/traffic_ga_ablation_v3/streaming_H*_E${EXPERTS}.log
