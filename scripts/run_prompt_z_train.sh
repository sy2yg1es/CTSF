#!/bin/bash

# ==============================================================================
# Prompt-Z Stage A: Offline Training
# ==============================================================================

if [ -d "./data" ]; then ROOT_PATH="./data"; else ROOT_PATH="./dataset"; fi

# Auto-detect pretrained weights
find_weights() {
    local DS=$1 H=$2
    # Try common patterns
    for pat in "weights/patchtst_pretrained_${DS}_H${H}.pth" \
               "weights/pretrain_${DS}_H${H}.pth" \
               "weights/${DS}_H${H}_best.pth" \
               "checkpoints/pretrain_${DS}_H${H}/checkpoint.pth"; do
        if [ -f "$pat" ]; then echo "$pat"; return; fi
    done
    echo ""
}

EPOCHS=3
LR=0.001
RANK=8
D_DRIFT=64
LAMBDA_DELTA=0.0002
LAMBDA_MASK=0.0001
LAMBDA_NOOP=0.005
TARGET_MASK_RATIO=0.10
MAX_DELTA_RATIO=0.05
REG_WARMUP_STEPS=2000
NOOP_WARMUP_STEPS=14000
NOOP_RAMP_STEPS=2000
NOOP_MIN_EFFECTIVE_RATIO=0.0001
GAMMA_FLOOR=0.1
GAMMA_FLOOR_STEPS=8000
MASK_FLOOR=0.05
MASK_FLOOR_STEPS=12000

mkdir -p weights/prompt_z logs/prompt_z

for DS_H in "ECL.csv:96:321" "Traffic.csv:1:862" "ETTh1.csv:24:7"; do
    DATA=$(echo $DS_H | cut -d: -f1)
    H=$(echo $DS_H | cut -d: -f2)
    ENC_IN=$(echo $DS_H | cut -d: -f3)
    DSNAME=$(echo $DATA | sed 's/\.csv//')

    echo "================================================================="
    echo " Training Prompt-Z: ${DSNAME} H=${H}"
    echo "================================================================="

    WEIGHTS=$(find_weights ${DSNAME} ${H})
    WEIGHTS_ARG=""
    if [ -n "$WEIGHTS" ]; then
        WEIGHTS_ARG="--pretrained_weights ${WEIGHTS}"
        echo "  Using pretrained: ${WEIGHTS}"
    else
        echo "  No pretrained weights found, using random init"
    fi

    PYTHONPATH=. python train_prompt_z.py \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA} \
        --forecast_H ${H} \
        --enc_in ${ENC_IN} \
        --backbone patchtst \
        --D_model 512 \
        --e_layers 3 \
        ${WEIGHTS_ARG} \
        --d_drift ${D_DRIFT} \
        --rank ${RANK} \
        --gamma_init_bias -3.0 \
        --mask_init_bias -1.5 \
        --max_delta_ratio ${MAX_DELTA_RATIO} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --lambda_delta ${LAMBDA_DELTA} \
        --lambda_mask ${LAMBDA_MASK} \
        --lambda_noop ${LAMBDA_NOOP} \
        --target_mask_ratio ${TARGET_MASK_RATIO} \
        --reg_warmup_steps ${REG_WARMUP_STEPS} \
        --noop_warmup_steps ${NOOP_WARMUP_STEPS} \
        --noop_ramp_steps ${NOOP_RAMP_STEPS} \
        --noop_min_effective_ratio ${NOOP_MIN_EFFECTIVE_RATIO} \
        --gamma_floor ${GAMMA_FLOOR} \
        --gamma_floor_steps ${GAMMA_FLOOR_STEPS} \
        --mask_floor ${MASK_FLOOR} \
        --mask_floor_steps ${MASK_FLOOR_STEPS} \
        --delayed_residual_training \
        --save_dir weights/prompt_z \
        2>&1 | tee logs/prompt_z/train_${DSNAME}_H${H}.log

    echo ""
done

echo "================================================================="
echo " Training Complete"
echo "================================================================="
