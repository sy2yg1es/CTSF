#!/bin/bash

# ==============================================================================
# Stage 3 Evaluation: Pretrained RichMLPRouter vs Baselines
# 数据集: ECL H=96 / Traffic H=1 / ETTh1 H=24
#
# 对照组:
#   frozen                    — no online update
#   ours_old                  — legacy detector + original linear router
#   ours_improved_detector    — improved detector + original linear router
#   posterior_oracle_routing  — posterior expert oracle, includes no-op/frozen
#   stage3a           — pretrained router, FROZEN online (只更新 prompt experts)
#   stage3b           — pretrained router, calibration-only online
# ==============================================================================

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512; EXPERTS=32; TOP_K=2; WINDOW_K=12; TAU=0.1; PATIENCE=2; WORKERS=4
TEMP_T=1.0; SEED=2026
LR=0.0001; L_AUX=0.0

LOGDIR="logs/stage3_eval"
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
        $@ \
        > ${LOGDIR}/${TAG}.log 2>&1
    MSE=$(extract_mse ${LOGDIR}/${TAG}.log)
    UPD=$(grep "Updates:" ${LOGDIR}/${TAG}.log | grep -oP 'Updates: \K[0-9/]+' | tail -1)
    NOOP=$(grep "noop\|no-op\|no.op\|noop_ratio\|p_noop" ${LOGDIR}/${TAG}.log | tail -1)
    echo "    MSE=${MSE} Updates=${UPD}"
    [ -n "$NOOP" ] && echo "    ${NOOP}"
    echo ""
}

echo "================================================================="
echo " ECL H=96"
echo "================================================================="
run "ECL96_frozen"                    ECL.csv 96 frozen
run "ECL96_ours_old"                  ECL.csv 96 ours --legacy_detector
run "ECL96_ours_improved_detector"    ECL.csv 96 ours
run "ECL96_posterior_oracle_routing"  ECL.csv 96 posterior_oracle_routing --load_prompt_memory auto --router_stage 2
run "ECL96_stage3a"                   ECL.csv 96 ours --load_prompt_memory auto --router_stage 2
run "ECL96_stage3b"                   ECL.csv 96 ours --load_prompt_memory auto --router_stage 2 --router_calibration

echo "================================================================="
echo " Traffic H=1"
echo "================================================================="
run "Traffic1_frozen"                    Traffic.csv 1 frozen
run "Traffic1_ours_old"                  Traffic.csv 1 ours --legacy_detector
run "Traffic1_ours_improved_detector"    Traffic.csv 1 ours
run "Traffic1_posterior_oracle_routing"  Traffic.csv 1 posterior_oracle_routing --load_prompt_memory auto --router_stage 2
run "Traffic1_stage3a"                   Traffic.csv 1 ours --load_prompt_memory auto --router_stage 2
run "Traffic1_stage3b"                   Traffic.csv 1 ours --load_prompt_memory auto --router_stage 2 --router_calibration

echo "================================================================="
echo " ETTh1 H=24"
echo "================================================================="
run "ETTh1_24_frozen"                    ETTh1.csv 24 frozen
run "ETTh1_24_ours_old"                  ETTh1.csv 24 ours --legacy_detector
run "ETTh1_24_ours_improved_detector"    ETTh1.csv 24 ours
run "ETTh1_24_posterior_oracle_routing"  ETTh1.csv 24 posterior_oracle_routing --load_prompt_memory auto --router_stage 2
run "ETTh1_24_stage3a"                   ETTh1.csv 24 ours --load_prompt_memory auto --router_stage 2
run "ETTh1_24_stage3b"                   ETTh1.csv 24 ours --load_prompt_memory auto --router_stage 2 --router_calibration

echo "================================================================="
echo " SUMMARY"
echo "================================================================="
printf "%-25s %8s %8s\n" "Experiment" "MSE" "vs_frozen"
printf "%-25s %8s %8s\n" "-------------------------" "--------" "--------"

declare -A FROZEN_MSE=( [ECL96]=0.3215 [Traffic1]=0.2322 [ETTh1_24]=0.3628 )

for TAG in \
    ECL96_frozen ECL96_ours_old ECL96_ours_improved_detector ECL96_posterior_oracle_routing ECL96_stage3a ECL96_stage3b \
    Traffic1_frozen Traffic1_ours_old Traffic1_ours_improved_detector Traffic1_posterior_oracle_routing Traffic1_stage3a Traffic1_stage3b \
    ETTh1_24_frozen ETTh1_24_ours_old ETTh1_24_ours_improved_detector ETTh1_24_posterior_oracle_routing ETTh1_24_stage3a ETTh1_24_stage3b; do

    LOG="${LOGDIR}/${TAG}.log"
    MSE=$(extract_mse "$LOG")
    if [ -z "$MSE" ]; then
        printf "%-25s %8s\n" "$TAG" "FAIL"
        continue
    fi
    # Find frozen baseline for this dataset
    DS=$(echo $TAG | sed 's/_frozen\|_ours_old\|_ours_improved_detector\|_posterior_oracle_routing\|_stage3a\|_stage3b//')
    FMSE=${FROZEN_MSE[$DS]:-0}
    if [ "$FMSE" != "0" ]; then
        DELTA=$(python3 -c "print(f'{(float(\"$MSE\")-float(\"$FMSE\"))/float(\"$FMSE\")*100:+.1f}%')")
    else
        DELTA="N/A"
    fi
    printf "%-25s %8s %8s\n" "$TAG" "$MSE" "$DELTA"
done
