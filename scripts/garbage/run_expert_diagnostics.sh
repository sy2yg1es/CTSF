#!/bin/bash
# Expert Bank Signal Diagnostics
# 回答：oracle label 是否有可学信号？

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

LOGDIR="logs/expert_diag"
mkdir -p ${LOGDIR}

for DS_H in "ECL.csv:96" "Traffic.csv:1" "ETTh1.csv:24"; do
    DATA=$(echo $DS_H | cut -d: -f1)
    H=$(echo $DS_H | cut -d: -f2)
    DSNAME=$(echo $DATA | sed 's/\.csv//')

    echo "============================================="
    echo " ${DSNAME} H=${H}"
    echo "============================================="

    # A) Random init expert bank (no weights loaded)
    echo "--- Random init experts ---"
    PYTHONPATH=. python engine/expert_diagnostics.py \
        --root_path ${ROOT_PATH} --data_path ${DATA} --forecast_H ${H} \
        --n_windows 200 --oracle_temp 1.0 \
        --output ${LOGDIR}/${DSNAME}_H${H}_random.json \
        2>&1 | tail -15

    # B) Stage 2 pretrained expert bank
    PM="./weights/prompt_memory_stage2_${DSNAME}_H${H}.pth"
    if [ -f "$PM" ]; then
        echo ""
        echo "--- Stage2 pretrained experts ---"
        PYTHONPATH=. python engine/expert_diagnostics.py \
            --root_path ${ROOT_PATH} --data_path ${DATA} --forecast_H ${H} \
            --n_windows 200 --oracle_temp 1.0 \
            --pm_weights ${PM} \
            --output ${LOGDIR}/${DSNAME}_H${H}_stage2.json \
            2>&1 | tail -15
    fi

    echo ""
done
