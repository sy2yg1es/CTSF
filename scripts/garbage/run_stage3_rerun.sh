#!/bin/bash
# Stage 3 eval: only stage3a and stage3b (re-run after noop OOB fix)

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512; EXPERTS=32; TOP_K=2; WINDOW_K=12; TAU=0.1; PATIENCE=2; WORKERS=4
LR=0.0001; L_AUX=0.0

LOGDIR="logs/stage3_eval"
mkdir -p ${LOGDIR}

extract_mse() {
    sed -n 's/.*MAE: [0-9.]*, MSE: \([0-9.]*\), RMSE:.*/\1/p' "$1" | tail -1
}

BASE_ARGS="--D_model ${D_MODEL} --window_K ${WINDOW_K} --threshold_tau ${TAU}
           --patience_C ${PATIENCE} --num_experts ${EXPERTS} --top_k ${TOP_K}
           --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS}
           --root_path ${ROOT_PATH} --features M --seq_len 96 --backbone patchtst"

run() {
    local TAG=$1; local DATA=$2; local H=$3; local MODE=$4; shift 4
    echo "  [${TAG}] ..."
    PYTHONPATH=. python main.py ${BASE_ARGS} \
        --data_path ${DATA} --forecast_H ${H} \
        --streaming_mode ${MODE} \
        --experiment_tag ${TAG} \
        $@ \
        > ${LOGDIR}/${TAG}.log 2>&1
    MSE=$(extract_mse ${LOGDIR}/${TAG}.log)
    UPD=$(grep "Updates:" ${LOGDIR}/${TAG}.log | grep -oP 'Updates: \K[0-9/]+' | tail -1)
    echo "    MSE=${MSE}  Updates=${UPD}"; echo ""
}

for DS_H in "ECL.csv:96" "Traffic.csv:1" "ETTh1.csv:24"; do
    DATA=$(echo $DS_H | cut -d: -f1)
    H=$(echo $DS_H | cut -d: -f2)
    DSNAME=$(echo $DATA | sed 's/\.csv//')
    TAG="${DSNAME}_H${H}"

    run "${DSNAME}${H}_stage3a" ${DATA} ${H} ours \
        --load_prompt_memory auto --router_stage 2

    run "${DSNAME}${H}_stage3b" ${DATA} ${H} ours \
        --load_prompt_memory auto --router_stage 2 --router_calibration
done

echo "=== STAGE 3 RESULTS ==="
printf "%-28s %8s %8s\n" "Experiment" "MSE" "Δ%vs_ours"

# Reference: ours MSE from oracle logs
declare -A OURS_MSE=([ECL96]=0.2659 [Traffic1]=0.2325 [ETTh1_24]=0.3554)

for TAG in ECL96_stage3a ECL96_stage3b Traffic1_stage3a Traffic1_stage3b ETTh1_24_stage3a ETTh1_24_stage3b; do
    LOG="${LOGDIR}/${TAG}.log"
    MSE=$(extract_mse "$LOG")
    if [ -z "$MSE" ]; then
        printf "%-28s %8s\n" "$TAG" "FAIL"; continue
    fi
    DS=$(echo $TAG | sed 's/_stage3[ab]//')
    OMSE=${OURS_MSE[$DS]:-0}
    DELTA=$(python3 -c "print(f'{(float(\"$MSE\")-float(\"$OMSE\"))/float(\"$OMSE\")*100:+.1f}%')" 2>/dev/null || echo "N/A")
    printf "%-28s %8s %8s\n" "$TAG" "$MSE" "$DELTA"
done

# Also print reference lines
echo ""
echo "Reference (oracle logs):"
printf "%-28s %8s\n" "ECL96   frozen=0.3215" "ours_improved=0.2659"
printf "%-28s %8s\n" "Traffic1 frozen=0.2322" "ours_improved=0.2325"
printf "%-28s %8s\n" "ETTh1_24 frozen=0.3628" "ours=0.3554"
