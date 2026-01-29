#!/bin/bash

# 固定参数
DATASET="credit"
MODEL="mlp"
LR=0.003
DEVICE="cuda:4"
BATCH=5
RATIO=0.1
LOG="--log"

# 搜索空间（按你截图）
TEMPS=(0.01 0.2 0.4)
OPTIMS=("SGD" "SVRG")
SEEDS=(2028 2029 2030 2031 2032 2033)

# 并行数量上限
MAX_JOBS=18

# 控制并行任务数量
function wait_for_free_slot() {
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
}

for T in "${TEMPS[@]}"; do
    for OPT in "${OPTIMS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            wait_for_free_slot

            CMD="python main.py \
                --optimizer $OPT \
                --dataset $DATASET \
                --nn_model $MODEL \
                --lr $LR \
                --log \
                --device $DEVICE \
                --batch_size $BATCH \
                --temperature $T \
                --ratio $RATIO \
                --seed $SEED"

            echo "Running: $CMD"
            eval $CMD &

        done
    done
done

wait
echo "All credit experiments finished!"
