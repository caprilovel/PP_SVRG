#!/bin/bash

# 固定参数
DATASET="CIFAR10"
MODEL="CIFAR10_convnet"
LR=0.005
DEVICE="cuda:5"
BATCH=5
RATIO=0.1
LOG="--log"

# 搜索空间
TEMPS=(20 50 80)
OPTIMS=("SGD" "SVRG")
SEEDS=(2023 2024 2025)

# 并行数量上限（比如你 GPU 够，可以开到 12）
MAX_JOBS=12

# 控制并行任务数量的函数
function wait_for_free_slot() {
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
}

for T in "${TEMPS[@]}"; do
    for OPT in "${OPTIMS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            wait_for_free_slot  # 控制并行数量

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
            eval $CMD &   # 并行执行

        done
    done
done

wait   # 等全部子进程结束
echo "All experiments finished!"
