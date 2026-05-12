#!/bin/bash

# --- 针对 Q2: 强执行性稳定性 (Temperature/Sensitivity Test) ---
# 审稿人担心 epsilon (在你的代码中对应 ratio 或由 temperature 影响) 较大时 SPRINT 会失效。
# 增加不同的 temperature (例如 20, 50, 80, 100)，观察收敛曲线的震荡情况。
TEMPS=("80" "100")

for t in "${TEMPS[@]}"
do
    echo "Running Q2: temperature=$t"
    # SPRINT (SVRG)
    python main.py --optimizer SVRG --dataset MNIST --nn_model one_layer --lr 0.003 --temperature $t --ratio 1 --batch_size 100 --log 
    # SGD-GD (Baseline)
    python main.py --optimizer SGD --dataset MNIST --nn_model one_layer --lr 0.003 --temperature $t --ratio 1 --batch_size 100 --log 
done

# --- 针对 Q3: 计算效率与样本量消融 (n_samples Ablation) ---
# 证明 SPRINT 不需要全量 n 也能收敛得比 SGD-GD 好，以此回应“计算开销大”的问题。
# 假设全量是 12000，我们测试 100, 500, 1000, 5000。
SAMPLES=("50" "100" "500" "1000")

for ns in "${SAMPLES[@]}"
do
    echo "Running Q3: n_samples=$ns"
    # 重点看 SPRINT 在小样本 Snapshot 下的表现
    python main.py --optimizer SVRG --dataset MNIST --nn_model one_layer --lr 0.003 --temperature 50 --n_samples $ns --batch_size 100 --log 
done