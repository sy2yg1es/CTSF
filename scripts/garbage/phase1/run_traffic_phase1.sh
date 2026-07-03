#!/bin/bash
# Traffic H=384 GA=8 — Phase 1 最小验证
# 对比对象：原始 Traffic H=384 static MSE 0.4165 / streaming MSE 0.4224

ROOT_PATH=$([ -d "./data" ] && echo "./data" || echo "./dataset")
mkdir -p logs/traffic_phase1

echo "[*] Traffic H=384 | D_model=512 | GA=8 | Eff BS=8 | Epochs=30"

PYTHONPATH=. python pretrain.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model 512 \
    --accum_steps 8 --epochs 30 \
    > logs/traffic_phase1/pretrain.log 2>&1
echo "    pretrain done"

PYTHONPATH=. python eval_static.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model 512 \
    --accum_steps 8 \
    > logs/traffic_phase1/static.log 2>&1
echo "    static done"

PYTHONPATH=. python main.py \
    --root_path ${ROOT_PATH} --data_path Traffic.csv \
    --features M --seq_len 96 --forecast_H 384 --D_model 512 \
    --accum_steps 8 \
    --window_K 12 --threshold_tau 0.1 --patience_C 2 \
    --num_experts 32 --top_k 2 \
    --learning_rate 0.0001 --l_aux_weight 0.0 --num_workers 4 \
    > logs/traffic_phase1/streaming.log 2>&1
echo "    streaming done"

echo ""
grep "Epoch.*Average Loss" logs/traffic_phase1/pretrain.log | tail -1
grep "Completed" logs/traffic_phase1/static.log
grep "Completed" logs/traffic_phase1/streaming.log
