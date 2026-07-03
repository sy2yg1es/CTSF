#!/bin/bash

# ==============================================================================
# ECL + Traffic GA Ablation v2 — 修正版
# ==============================================================================
# 问题回顾：
#   v1 错误地将 D_model 设为 forecast_H (192/384)
#   原始实验 D_MODEL=512 固定，forecast_H 才是 96/192/384
#   v1 同时改变了两个变量（D_model + accum_steps），结论被污染
#
# v2 修正：
#   固定 D_model=512（与原始实验一致）
#   只变 accum_steps（才是真正的 GA ablation）
#   forecast_H 固定为各数据集原始最差配置：
#     ECL: forecast_H=384 (原来 H=384 最差)
#     Traffic: forecast_H=192 (原来 H=192 最优，用于验证 GA 能否进一步提升)
#              forecast_H=384 (原来 H=384 次于 H=192)
# ==============================================================================

if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

D_MODEL=512   # 固定！与原始实验一致

WINDOW_K=12
TAU=0.1
PATIENCE=2
TOP_K=2
LR=0.0001
L_AUX=0.0
WORKERS=4
EXPERTS=32

mkdir -p logs/ECL_bs_ablation_v2
mkdir -p logs/traffic_ga_ablation_v2

echo "================================================================="
echo "[*] GA Ablation v2 (修正版): D_model=512 固定"
echo "================================================================="

# ==============================================================================
# Part 1: ECL — forecast_H=384, D_model=512, accum=1/8/16
# 对比对象：原始实验 ECL H=384 (forecast_H=384, D_model=512, accum=1)
# ==============================================================================

echo ""
echo "=== ECL: forecast_H=384 | D_model=512 ==="

for ACCUM in 1 8 16; do
    if [ ${ACCUM} -eq 1 ]; then
        EPOCHS=10
    elif [ ${ACCUM} -eq 8 ]; then
        EPOCHS=30
    else
        EPOCHS=40
    fi
    EFF_BS=$((3 * ACCUM))
    echo ""
    echo "--- ECL GA=${ACCUM} | Eff BS=${EFF_BS} | Epochs=${EPOCHS} ---"

    PYTHONPATH=. python pretrain.py \
        --root_path ${ROOT_PATH} \
        --data_path ECL.csv \
        --features M \
        --seq_len 96 \
        --forecast_H 384 \
        --D_model ${D_MODEL} \
        --accum_steps ${ACCUM} \
        --epochs ${EPOCHS} > logs/ECL_bs_ablation_v2/pretrain_H384_GA${ACCUM}.log 2>&1
    echo "    pretrain done"

    PYTHONPATH=. python eval_static.py \
        --root_path ${ROOT_PATH} \
        --data_path ECL.csv \
        --features M \
        --seq_len 96 \
        --forecast_H 384 \
        --D_model ${D_MODEL} \
        --accum_steps ${ACCUM} > logs/ECL_bs_ablation_v2/static_H384_GA${ACCUM}.log 2>&1
    echo "    static done"

    PYTHONPATH=. python main.py \
        --root_path ${ROOT_PATH} \
        --data_path ECL.csv \
        --features M \
        --seq_len 96 \
        --forecast_H 384 \
        --D_model ${D_MODEL} \
        --accum_steps ${ACCUM} \
        --window_K ${WINDOW_K} \
        --threshold_tau ${TAU} \
        --patience_C ${PATIENCE} \
        --num_experts ${EXPERTS} \
        --top_k ${TOP_K} \
        --learning_rate ${LR} \
        --l_aux_weight ${L_AUX} \
        --num_workers ${WORKERS} > logs/ECL_bs_ablation_v2/streaming_H384_GA${ACCUM}_E${EXPERTS}.log 2>&1
    echo "    streaming done"
done

# Control: ECL H=96 GA=16 (原始最优配置 + GA，看 GA 是否有额外收益)
echo ""
echo "--- ECL Control: forecast_H=96 | GA=16 | Epochs=40 ---"
PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 16 --epochs 40 > logs/ECL_bs_ablation_v2/pretrain_H96_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 16 > logs/ECL_bs_ablation_v2/static_H96_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} \
    --data_path ECL.csv \
    --features M --seq_len 96 --forecast_H 96 --D_model ${D_MODEL} \
    --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/ECL_bs_ablation_v2/streaming_H96_GA16_E${EXPERTS}.log 2>&1
echo "    streaming done"

# ==============================================================================
# Part 2: Traffic — D_model=512, accum=1(已有)/16
# 对比对象：原始实验 Traffic H=192 / H=384 (D_model=512, accum=1)
# ==============================================================================

echo ""
echo "=== Traffic: D_model=512 ==="

# Traffic forecast_H=192 GA=16（原始最优 + GA）
echo ""
echo "--- Traffic forecast_H=192 | GA=16 | Eff BS=16 | Epochs=40 ---"
PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 16 --epochs 40 > logs/traffic_ga_ablation_v2/pretrain_H192_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 16 > logs/traffic_ga_ablation_v2/static_H192_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 192 --D_model ${D_MODEL} \
    --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_ga_ablation_v2/streaming_H192_GA16_E${EXPERTS}.log 2>&1
echo "    streaming done"

# Traffic forecast_H=384 GA=16（原始次优 + GA，对应 ECL 的主实验）
echo ""
echo "--- Traffic forecast_H=384 | GA=16 | Eff BS=16 | Epochs=40 ---"
PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 16 --epochs 40 > logs/traffic_ga_ablation_v2/pretrain_H384_GA16.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 16 > logs/traffic_ga_ablation_v2/static_H384_GA16.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model ${D_MODEL} \
    --accum_steps 16 \
    --window_K ${WINDOW_K} --threshold_tau ${TAU} --patience_C ${PATIENCE} \
    --num_experts ${EXPERTS} --top_k ${TOP_K} \
    --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS} \
    > logs/traffic_ga_ablation_v2/streaming_H384_GA16_E${EXPERTS}.log 2>&1
echo "    streaming done"

# ==============================================================================
# 汇总
# ==============================================================================

echo ""
echo "================================================================="
echo "[*] v2 全部完成！"
echo "================================================================="
echo ""
echo "=== ECL v2 结果 ==="
grep "Completed\|Average Loss" logs/ECL_bs_ablation_v2/static_H384_GA1.log 2>/dev/null || echo "(GA1 静态: 见原始 logs/ECL/static_H384.log)"
grep "Completed" logs/ECL_bs_ablation_v2/static_H384_GA8.log
grep "Completed" logs/ECL_bs_ablation_v2/static_H384_GA16.log
grep "Completed" logs/ECL_bs_ablation_v2/streaming_H384_GA*_E${EXPERTS}.log
echo ""
echo "=== Traffic v2 结果 ==="
grep "Completed" logs/traffic_ga_ablation_v2/static_H*.log
grep "Completed" logs/traffic_ga_ablation_v2/streaming_H*_E${EXPERTS}.log
