#!/bin/bash
# Stage 3 diagnostics: router learnability + posterior/causal oracle gap.

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512; EXPERTS=32; TOP_K=2; WINDOW_K=12; TAU=0.1; PATIENCE=2; WORKERS=4
TEMP_T=1.0; SEED=2026
LR=0.0001; L_AUX=0.0

LOGDIR="logs/stage3_diagnostics"
mkdir -p ${LOGDIR}

extract_mse() {
    sed -n 's/.*MAE: [0-9.]*, MSE: \([0-9.]*\), RMSE:.*/\1/p' "$1" | tail -1
}

BASE_ARGS="--D_model ${D_MODEL} --window_K ${WINDOW_K} --threshold_tau ${TAU}
           --patience_C ${PATIENCE} --num_experts ${EXPERTS} --top_k ${TOP_K}
           --learning_rate ${LR} --l_aux_weight ${L_AUX} --num_workers ${WORKERS}
           --temp_T ${TEMP_T} --seed ${SEED}
           --root_path ${ROOT_PATH} --features M --seq_len 96 --backbone patchtst"

run() {
    local TAG=$1; local DATA=$2; local H=$3; local MODE=$4; shift 4
    echo "  [${TAG}] ..."
    PYTHONPATH=. python main.py ${BASE_ARGS} \
        --data_path ${DATA} --forecast_H ${H} \
        --streaming_mode ${MODE} \
        --experiment_tag ${TAG} \
        "$@" \
        > ${LOGDIR}/${TAG}.log 2>&1
    MSE=$(extract_mse ${LOGDIR}/${TAG}.log)
    NOOP=$(grep "no-op\|noop\|p_noop" ${LOGDIR}/${TAG}.log | tail -3)
    echo "    MSE=${MSE}"
    [ -n "$NOOP" ] && echo "$NOOP" | sed 's/^/    /'
    echo ""
}

run_case() {
    local TAG=$1; local DATA=$2; local H=$3

    echo "================================================================="
    echo " Diagnostics: ${TAG}"
    echo "================================================================="

    run "${TAG}_router_learnability" ${DATA} ${H} router_learnability \
        --load_prompt_memory auto --router_stage 2 --router_calibration \
        --diagnostics_dir ${LOGDIR}

    run "${TAG}_frozen" ${DATA} ${H} frozen
    run "${TAG}_ours_improved_detector" ${DATA} ${H} ours
    run "${TAG}_stage3b" ${DATA} ${H} ours \
        --load_prompt_memory auto --router_stage 2 --router_calibration
    run "${TAG}_posterior_oracle_routing" ${DATA} ${H} posterior_oracle_routing \
        --load_prompt_memory auto --router_stage 2

    for K in 12 24 48; do
        run "${TAG}_causal_oracle_routing_K${K}" ${DATA} ${H} causal_oracle_routing \
            --load_prompt_memory auto --router_stage 2 \
            --causal_oracle_K ${K}
    done
}

run_case "ECL_H96" ECL.csv 96
run_case "Traffic_H1" Traffic.csv 1
run_case "ETTh1_H24" ETTh1.csv 24

echo "================================================================="
echo " SUMMARY"
echo "================================================================="
printf "%-42s %10s\n" "Experiment" "MSE"

for TAG in \
    ECL_H96_frozen ECL_H96_ours_improved_detector ECL_H96_stage3b ECL_H96_posterior_oracle_routing ECL_H96_causal_oracle_routing_K12 ECL_H96_causal_oracle_routing_K24 ECL_H96_causal_oracle_routing_K48 \
    Traffic_H1_frozen Traffic_H1_ours_improved_detector Traffic_H1_stage3b Traffic_H1_posterior_oracle_routing Traffic_H1_causal_oracle_routing_K12 Traffic_H1_causal_oracle_routing_K24 Traffic_H1_causal_oracle_routing_K48 \
    ETTh1_H24_frozen ETTh1_H24_ours_improved_detector ETTh1_H24_stage3b ETTh1_H24_posterior_oracle_routing ETTh1_H24_causal_oracle_routing_K12 ETTh1_H24_causal_oracle_routing_K24 ETTh1_H24_causal_oracle_routing_K48; do
    LOG="${LOGDIR}/${TAG}.log"
    MSE=$(extract_mse "$LOG")
    if [ -z "$MSE" ]; then
        printf "%-42s %10s\n" "$TAG" "FAIL"
    else
        printf "%-42s %10s\n" "$TAG" "$MSE"
    fi
done

echo ""
echo "Router learnability JSON/CSV:"
ls ${LOGDIR}/router_learnability_*.json ${LOGDIR}/router_learnability_*.csv 2>/dev/null
