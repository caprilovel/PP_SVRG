#!/bin/bash

# --- 核心逻辑：Batch Size 越小 -> 梯度方差 sigma 越大 ---
# 我们测试从极小 (1) 到较大 (256) 的 Batch Size
BATCH_SIZES=("1" "16" "64" "256")

# 1. MNIST 实验
echo "Starting MNIST Batch Size Ablation (Variance Test)..."
for bs in "${BATCH_SIZES[@]}"
do
    echo "Running MNIST: Batch Size = $bs"
    # SPRINT (SVRG 优化器) - 理论预估：最终 Loss 平台高度应保持一致
    python main.py --optimizer SVRG --dataset MNIST --nn_model one_layer \
        --lr 0.003 --device 0 --log --temperature 50 --ratio 1 \
        --batch_size $bs

    # SGD-GD (Baseline) - 理论预估：最终 Loss 平台随 bs 减小而升高
    python main.py --optimizer SGD --dataset MNIST --nn_model one_layer \
        --lr 0.003 --device 0 --log --temperature 50 --ratio 1 \
        --batch_size $bs
done

# 2. Credit 实验 (如果你的 credit.py 也支持 --batch_size)
echo "Starting Credit Batch Size Ablation..."
for bs in "${BATCH_SIZES[@]}"
do
    echo "Running Credit: Batch Size = $bs"
    # SPRINT
    python credit.py --dataset credit --nn_model mlp --ratio 1 --log \
        --optimizer SVRG --batch_size $bs 
    # SGD-GD
    python credit.py --dataset credit --nn_model mlp --ratio 1 --log \
        --optimizer SGD --batch_size $bs 
done