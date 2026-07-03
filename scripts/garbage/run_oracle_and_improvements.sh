#!/bin/bash

# ==============================================================================
# Oracle Experiments + 两种改进方法对比
# 数据集：ECL H=96 (PatchTST，之前 ours=0.2687, frozen=0.3217)
#         Traffic H=1 (ours比frozen差，排查误触发)
#         ETTh1 H=24 (ours比frozen好)
# ==============================================================================

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

D_MODEL=512
WINDOW_K=12; TAU=0.1; PATIENCE=2; TOP_K=2
LR=0.0001; L_AUX=0.0; WORKERS=4; EXPERTS=32

LOGDIR="logs/oracle"
mkdir -p ${LOGDIR}

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
    grep -E "Completed|Updates" ${LOGDIR}/${TAG}.log | tail -2
    echo ""
}

echo "================================================================="
echo " Oracle Experiments — ECL H=96 / Traffic H=1 / ETTh1 H=24"
echo "================================================================="

echo ""
echo "=== ECL H=96 (baseline: frozen=0.3217, ours=0.2687) ==="
run "ECL96_frozen"          ECL.csv 96 frozen
run "ECL96_ours"            ECL.csv 96 ours
run "ECL96_oracle_detector" ECL.csv 96 oracle_detector
run "ECL96_oracle_channel"  ECL.csv 96 oracle_channel
run "ECL96_oracle_routing"  ECL.csv 96 oracle_routing
run "ECL96_segment_adapt"   ECL.csv 96 segment_adapt --segment_size 500 --adapt_steps 10

echo ""
echo "=== Traffic H=1 (baseline: frozen=0.2286, ours比frozen差) ==="
run "Traffic1_frozen"          Traffic.csv 1 frozen
run "Traffic1_ours"            Traffic.csv 1 ours
run "Traffic1_oracle_detector" Traffic.csv 1 oracle_detector
run "Traffic1_oracle_channel"  Traffic.csv 1 oracle_channel
run "Traffic1_oracle_routing"  Traffic.csv 1 oracle_routing
run "Traffic1_segment_adapt"   Traffic.csv 1 segment_adapt --segment_size 200 --adapt_steps 5

echo ""
echo "=== ETTh1 H=24 (baseline: frozen=0.3625, ours=0.3550) ==="
run "ETTh1_24_frozen"          ETTh1.csv 24 frozen
run "ETTh1_24_ours"            ETTh1.csv 24 ours
run "ETTh1_24_oracle_detector" ETTh1.csv 24 oracle_detector
run "ETTh1_24_oracle_channel"  ETTh1.csv 24 oracle_channel
run "ETTh1_24_oracle_routing"  ETTh1.csv 24 oracle_routing
run "ETTh1_24_segment_adapt"   ETTh1.csv 24 segment_adapt --segment_size 300 --adapt_steps 10

echo ""
echo "================================================================="
echo " Improved Methods: ImprovedDetector + QueryMLP"
echo "================================================================="

echo ""
echo "=== ECL H=96 — improved methods ==="
run "ECL96_improved_det"     ECL.csv 96 ours --improved_detector
run "ECL96_improved_qmlp"    ECL.csv 96 ours --use_query_mlp
run "ECL96_improved_both"    ECL.csv 96 ours --improved_detector --use_query_mlp

echo ""
echo "=== Traffic H=1 — improved methods ==="
run "Traffic1_improved_det"  Traffic.csv 1 ours --improved_detector
run "Traffic1_improved_qmlp" Traffic.csv 1 ours --use_query_mlp
run "Traffic1_improved_both" Traffic.csv 1 ours --improved_detector --use_query_mlp

echo ""
echo "=== ETTh1 H=24 — improved methods ==="
run "ETTh1_24_improved_det"  ETTh1.csv 24 ours --improved_detector
run "ETTh1_24_improved_qmlp" ETTh1.csv 24 ours --use_query_mlp
run "ETTh1_24_improved_both" ETTh1.csv 24 ours --improved_detector --use_query_mlp

echo ""
echo "================================================================="
echo " SUMMARY"
echo "================================================================="
printf "%-30s %8s %8s\n" "Experiment" "MSE" "Verdict"
printf "%-30s %8s %8s\n" "$(printf '%0.s-' {1..30})" "--------" "-------"

for TAG in \
    ECL96_frozen ECL96_ours ECL96_oracle_detector ECL96_oracle_channel ECL96_oracle_routing ECL96_segment_adapt \
    ECL96_improved_det ECL96_improved_qmlp ECL96_improved_both \
    Traffic1_frozen Traffic1_ours Traffic1_oracle_detector Traffic1_oracle_channel Traffic1_oracle_routing Traffic1_segment_adapt \
    Traffic1_improved_det Traffic1_improved_qmlp Traffic1_improved_both \
    ETTh1_24_frozen ETTh1_24_ours ETTh1_24_oracle_detector ETTh1_24_oracle_channel ETTh1_24_oracle_routing ETTh1_24_segment_adapt \
    ETTh1_24_improved_det ETTh1_24_improved_qmlp ETTh1_24_improved_both; do
    LOG="${LOGDIR}/${TAG}.log"
    MSE=$(grep "Completed" "$LOG" 2>/dev/null | grep -oP 'MSE: \K[0-9.]+' | tail -1)
    if [ -z "$MSE" ]; then
        printf "%-30s %8s\n" "$TAG" "FAIL"
    else
        printf "%-30s %8s\n" "$TAG" "$MSE"
    fi
done
