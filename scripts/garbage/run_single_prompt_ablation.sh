#!/bin/bash

# ==============================================================================
# Single Prompt Ablation: MoE 是否必要？
#
# 对比：
#   frozen              — 无更新
#   full_ft             — backbone 全量微调
#   ours_improved       — 当前 MoE (E=32, K=2) + improved_detector
#   single_prompt       — E=1, K=1, same detector/channel mask/SGD
#   e32_random          — E=32, K=2, random init, no pretrained router
#   stage3b             — E=32, pretrained router (已证明无效)
#
# 数据集: ECL H=96 / Traffic H=1 / ETTh1 H=24 / ETTh1 H=1 / WTH H=24
# ==============================================================================

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512; WINDOW_K=12; TAU=0.1; PATIENCE=2; WORKERS=4
LR=0.0001; L_AUX=0.0

LOGDIR="logs/single_prompt_ablation"
mkdir -p ${LOGDIR}

BASE_ARGS="--D_model ${D_MODEL} --window_K ${WINDOW_K} --threshold_tau ${TAU}
           --patience_C ${PATIENCE} --learning_rate ${LR} --l_aux_weight ${L_AUX}
           --num_workers ${WORKERS} --root_path ${ROOT_PATH} --features M
           --seq_len 96 --backbone patchtst"

run() {
    local TAG=$1; local DATA=$2; local H=$3; local MODE=$4; shift 4
    echo -n "  [${TAG}] ..."
    PYTHONPATH=. python main.py ${BASE_ARGS} \
        --data_path ${DATA} --forecast_H ${H} \
        --streaming_mode ${MODE} \
        --experiment_tag ${TAG} \
        "$@" \
        > ${LOGDIR}/${TAG}.log 2>&1
    MSE=$(grep "Completed" ${LOGDIR}/${TAG}.log | grep -oP 'MSE: \K[0-9.]+' | tail -1)
    UPD=$(grep "Updates:" ${LOGDIR}/${TAG}.log | grep -oP 'Updates: \K[0-9/]+' | tail -1)
    echo " MSE=${MSE:-FAIL}  Updates=${UPD:-N/A}"
}

for DS_H in "ECL.csv:96" "Traffic.csv:1" "ETTh1.csv:24" "ETTh1.csv:1" "WTH.csv:24"; do
    DATA=$(echo $DS_H | cut -d: -f1)
    H=$(echo $DS_H | cut -d: -f2)
    DSNAME=$(echo $DATA | sed 's/\.csv//')

    echo ""
    echo "================================================================="
    echo " ${DSNAME} H=${H}"
    echo "================================================================="

    # frozen baseline
    run "${DSNAME}_H${H}_frozen" ${DATA} ${H} frozen

    # full_ft
    run "${DSNAME}_H${H}_full_ft" ${DATA} ${H} full_ft

    # ours_improved (E=32, K=2, improved_detector)
    run "${DSNAME}_H${H}_ours_e32" ${DATA} ${H} ours \
        --num_experts 32 --top_k 2

    # SINGLE PROMPT (E=1, K=1, same detector)
    run "${DSNAME}_H${H}_single" ${DATA} ${H} ours \
        --num_experts 1 --top_k 1

    # stage3b (E=32 + pretrained router) — only if weights exist
    PM_PATH="./weights/prompt_memory_stage2_${DSNAME}_H${H}.pth"
    if [ -f "$PM_PATH" ]; then
        run "${DSNAME}_H${H}_stage3b" ${DATA} ${H} ours \
            --num_experts 32 --top_k 2 \
            --load_prompt_memory auto --router_stage 2 --router_calibration
    else
        echo "  [${DSNAME}_H${H}_stage3b] SKIP (no weights: ${PM_PATH})"
    fi
done


echo ""
echo "================================================================="
echo " SUMMARY TABLE"
echo "================================================================="
printf "%-30s %8s\n" "Experiment" "MSE"
printf "%-30s %8s\n" "------------------------------" "--------"

for f in ${LOGDIR}/*.log; do
    TAG=$(basename $f .log)
    MSE=$(grep "Completed" "$f" 2>/dev/null | grep -oP 'MSE: \K[0-9.]+' | tail -1)
    printf "%-30s %8s\n" "$TAG" "${MSE:-FAIL}"
done
