#!/bin/bash
# Router sanity overfit: can the Stage3 router fit oracle labels on a tiny subset?

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512; EXPERTS=32; TOP_K=2; WINDOW_K=12; WORKERS=4
TEMP_T=1.0; SEED=2026
SAMPLES=${SANITY_SAMPLES:-512}
EPOCHS=${SANITY_EPOCHS:-20}

LOGDIR="logs/router_sanity"
mkdir -p ${LOGDIR}

BASE_ARGS="--D_model ${D_MODEL} --num_experts ${EXPERTS} --top_k ${TOP_K}
           --window_K ${WINDOW_K} --num_workers ${WORKERS}
           --temp_T ${TEMP_T} --seed ${SEED}
           --root_path ${ROOT_PATH} --features M --seq_len 96
           --router_hidden 256 --router_hist_hidden 64
           --oracle_temp 1.0 --log_interval 1
           --prompt_memory_weights auto --router_stage 2
           --sanity_samples ${SAMPLES}"

run_case() {
    local DATASET=$1; local H=$2; local DATA="${DATASET}.csv"
    local TAG="${DATASET}_H${H}_S${SAMPLES}"
    echo "================================================================="
    echo " Router sanity overfit: ${TAG}"
    echo "================================================================="
    PYTHONPATH=. python pretrain_router.py ${BASE_ARGS} \
        --data_path ${DATA} --forecast_H ${H} \
        --stage sanity --epochs ${EPOCHS} --router_lr 1e-3 \
        > ${LOGDIR}/${TAG}.log 2>&1
    grep "\[Sanity\] Final" ${LOGDIR}/${TAG}.log | tail -1
}

run_case ECL 96
run_case Traffic 1
run_case ETTh1 24

echo ""
echo "Logs saved under ${LOGDIR}"
