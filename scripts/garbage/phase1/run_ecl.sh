#!/bin/bash

# ==============================================================================
# ContinualPromptTSF - ECL Dataset Ablation Pipeline
# ==============================================================================

# 基础数据路径与超参设定 (优先检测 ./data 目录，兼容 ./dataset)
if [ -d "./data" ]; then
    ROOT_PATH="./data"
else
    ROOT_PATH="./dataset"
fi

DATA_PATH="ECL.csv"
SEQ_LEN=96
D_MODEL=512

# 动态在线微调的核心参数
WINDOW_K=12
TAU=0.1
PATIENCE=2
TOP_K=2
LR=0.0001
L_AUX=0.0
WORKERS=4

# 实验维度遍历数组
HORIZONS=(96 192 384)
EXPERTS=(32 64)

# 创建 logs/ecl 文件夹用于存放所有的 ECL 实验战报
mkdir -p logs/ecl

echo "================================================================="
echo "[*] 开始执行 ECL 数据集全量自动化实验 pipeline"
echo "[*] 使用数据路径: ${ROOT_PATH}/${DATA_PATH}"
echo "================================================================="

# ------------------------------------------------------------------------------
# 阶段 1：预训练 (Pre-training)
# ------------------------------------------------------------------------------
echo ">>> [Phase 1] 启动预训练 (Pre-training)..."
for H in "${HORIZONS[@]}"; do
    echo "[*] Pre-training ECL H=${H} ..."
    python pretrain.py \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --features M \
        --seq_len ${SEQ_LEN} \
        --forecast_H ${H} \
        --D_model ${D_MODEL} > logs/ecl/pretrain_H${H}.log 2>&1
    echo "    - H=${H} 预训练完成"
done

# ------------------------------------------------------------------------------
# 阶段 2：静态基线 (Static Baseline)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Phase 2] 启动静态基线评测 (Static Evaluation)..."
for H in "${HORIZONS[@]}"; do
    echo "[*] Static Eval ECL H=${H} ..."
    python eval_static.py \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --features M \
        --seq_len ${SEQ_LEN} \
        --forecast_H ${H} \
        --D_model ${D_MODEL} > logs/ecl/static_H${H}.log 2>&1
    echo "    - H=${H} 静态基线评估完成，日志已存入 logs/ecl/static_H${H}.log"
done

# ------------------------------------------------------------------------------
# 阶段 3：动态流式微调 (Streaming Continual Fine-tuning)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Phase 3] 启动动态流式评测 (Streaming Evaluation)..."
for H in "${HORIZONS[@]}"; do
    for E in "${EXPERTS[@]}"; do
        echo "[*] Streaming Eval ECL H=${H} | Experts=${E} ..."
        python main.py \
            --root_path ${ROOT_PATH} \
            --data_path ${DATA_PATH} \
            --features M \
            --seq_len ${SEQ_LEN} \
            --forecast_H ${H} \
            --D_model ${D_MODEL} \
            --window_K ${WINDOW_K} \
            --threshold_tau ${TAU} \
            --patience_C ${PATIENCE} \
            --num_experts ${E} \
            --top_k ${TOP_K} \
            --learning_rate ${LR} \
            --l_aux_weight ${L_AUX} \
            --num_workers ${WORKERS} > logs/ecl/streaming_H${H}_E${E}.log 2>&1
        echo "    - H=${H}, Experts=${E} 流式评估完成，日志已存入 logs/ecl/streaming_H${H}_E${E}.log"
    done
done

echo "================================================================="
echo "[*] 🎉 ECL 所有实验全部执行完毕！请前往 ./logs/ecl 目录查看战报。"
echo "================================================================="
