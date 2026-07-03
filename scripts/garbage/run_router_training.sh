#!/bin/bash

# ==============================================================================
# Stage 1+2 Router Training + 最终评估
# 数据集: ECL H=96 / Traffic H=1 / ETTh1 H=24
# ==============================================================================

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512; EXPERTS=32; TOP_K=2; WINDOW_K=12; WORKERS=4
TEMP_T=1.0; SEED=2026

LOGDIR="logs/router_training"
mkdir -p ${LOGDIR}

BASE_ARGS="--D_model ${D_MODEL} --num_experts ${EXPERTS} --top_k ${TOP_K}
           --window_K ${WINDOW_K} --num_workers ${WORKERS}
           --temp_T ${TEMP_T} --seed ${SEED}
           --root_path ${ROOT_PATH} --features M --seq_len 96
           --router_hidden 256 --router_hist_hidden 64
           --lambda_kl 2.0 --lambda_noop 0.5 --lambda_smooth 0.1
           --oracle_temp 1.0 --log_interval 500"

train_and_eval() {
    local DATASET=$1; local H=$2; local DATA="${DATASET}.csv"
    local TAG="${DATASET}_H${H}"

    echo ""
    echo "================================================================="
    echo " Router Training: ${TAG}"
    echo "================================================================="

    # Stage 1: Oracle Distillation (3 epochs)
    echo "[1/4] Stage 1 distillation..."
    PYTHONPATH=. python pretrain_router.py ${BASE_ARGS} \
        --data_path ${DATA} --forecast_H ${H} \
        --stage 1 --epochs 3 --router_lr 1e-3 \
        > ${LOGDIR}/stage1_${TAG}.log 2>&1
    echo "  done (noop_ratio: $(grep 'noop_ratio' ${LOGDIR}/stage1_${TAG}.log | tail -1))"

    # Stage 2: Joint fine-tuning (1 epoch, 10x smaller expert LR)
    echo "[2/4] Stage 2 joint tuning..."
    PYTHONPATH=. python pretrain_router.py ${BASE_ARGS} \
        --data_path ${DATA} --forecast_H ${H} \
        --stage 2 --epochs 2 --router_lr 1e-4 --expert_lr 1e-5 \
        --router_weights ./weights/router_stage1_${DATASET}_H${H}.pth \
        > ${LOGDIR}/stage2_${TAG}.log 2>&1
    echo "  done"

    # TODO: main.py needs --load_prompt_memory flag to load trained weights
    # Ablation will be run after main.py is updated
}

# ==============================================================================
# Run training for 3 key datasets
# ==============================================================================
train_and_eval ECL 96
train_and_eval Traffic 1
train_and_eval ETTh1 24

echo ""
echo "================================================================="
echo " Training complete. Weights saved to ./weights/"
echo " Next: update main.py to load prompt_memory weights, then eval."
echo "================================================================="
ls -la weights/router_stage*.pth weights/prompt_memory_stage*.pth 2>/dev/null
