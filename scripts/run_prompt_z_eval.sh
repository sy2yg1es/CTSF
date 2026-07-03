#!/bin/bash

# ==============================================================================
# Prompt-Z Streaming Evaluation
# ==============================================================================
# Baselines:
#   frozen            — No adaptation
#   mode0             — PromptZ frozen, residual_tracker updates only
#   mode1             — Mode 0 + gamma bias calibration
#   no_prompt_z       — Same backbone, but no PromptZ (=frozen baseline)
# ==============================================================================

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

LOGDIR="logs/prompt_z"
mkdir -p ${LOGDIR}

run() {
    local TAG=$1; shift
    echo -n "  [${TAG}] ..."
    PYTHONPATH=. python main_prompt_z.py "$@" \
        --experiment_tag ${TAG} \
        > ${LOGDIR}/${TAG}.log 2>&1
    MSE=$(grep "Final MSE" ${LOGDIR}/${TAG}.log | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    echo " MSE=${MSE:-FAIL}"
}

for DS_H in "ECL.csv:96:321" "Traffic.csv:1:862" "ETTh1.csv:24:7"; do
    DATA=$(echo $DS_H | cut -d: -f1)
    H=$(echo $DS_H | cut -d: -f2)
    ENC_IN=$(echo $DS_H | cut -d: -f3)
    DSNAME=$(echo $DATA | sed 's/\.csv//')

    echo ""
    echo "================================================================="
    echo " ${DSNAME} H=${H}"
    echo "================================================================="

    # Find pretrained backbone weights
    WEIGHTS=""
    for pat in "weights/pretrain_${DSNAME}_H${H}.pth" \
               "weights/${DSNAME}_H${H}_best.pth" \
               "checkpoints/pretrain_${DSNAME}_H${H}/checkpoint.pth"; do
        if [ -f "$pat" ]; then WEIGHTS="--pretrained_weights $pat"; break; fi
    done

    # Find trained PromptZ weights
    PZ_WEIGHTS=""
    for pat in "weights/prompt_z/prompt_z_${DSNAME}_H${H}.pth" \
               "weights/prompt_z/prompt_z_${DSNAME}_H${H}_final.pth"; do
        if [ -f "$pat" ]; then PZ_WEIGHTS="--prompt_z_weights $pat"; break; fi
    done

    MAX_DELTA_RATIO=${MAX_DELTA_RATIO:-0.05}
    BASE_ARGS="--root_path ${ROOT_PATH} --data_path ${DATA} --forecast_H ${H} \
               --enc_in ${ENC_IN} --backbone patchtst --D_model 512 --d_ff 512 --e_layers 3 \
               --max_delta_ratio ${MAX_DELTA_RATIO} \
               ${WEIGHTS}"

    # 1. Frozen baseline
    run "${DSNAME}_H${H}_frozen" ${BASE_ARGS} --streaming_mode frozen

    # 2. PromptZ Mode 0 (no trained weights — random init)
    run "${DSNAME}_H${H}_pz_random_mode0" ${BASE_ARGS} --streaming_mode mode0

    # 3. PromptZ Mode 0 (trained weights)
    if [ -n "${PZ_WEIGHTS}" ]; then
        run "${DSNAME}_H${H}_pz_mode0" ${BASE_ARGS} ${PZ_WEIGHTS} --streaming_mode mode0

        # 4. PromptZ Mode 1 (trained + calibration)
        run "${DSNAME}_H${H}_pz_mode1" ${BASE_ARGS} ${PZ_WEIGHTS} --streaming_mode mode1
    else
        echo "  [${DSNAME}_H${H}_pz_mode0] SKIP (no trained PromptZ weights)"
        echo "  [${DSNAME}_H${H}_pz_mode1] SKIP (no trained PromptZ weights)"
    fi
done

echo ""
echo "================================================================="
echo " SUMMARY"
echo "================================================================="
printf "%-35s %10s\n" "Experiment" "MSE"
printf "%-35s %10s\n" "-----------------------------------" "----------"
for f in ${LOGDIR}/*.log; do
    TAG=$(basename $f .log)
    # Skip training logs
    [[ $TAG == train_* ]] && continue
    MSE=$(grep "Final MSE" "$f" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "%-35s %10s\n" "$TAG" "${MSE:-FAIL}"
done
