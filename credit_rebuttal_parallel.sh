#!/bin/bash

# 创建专门存放 Credit 实验结果的目录
LOG_DIR="./credit_rebuttal_logs"
mkdir -p $LOG_DIR

echo "Starting Parallel Rebuttal Experiments for Credit Dataset..."

# --- 针对 Q2: 敏感度压力测试 (Temperature Test) ---
# 改变 temperature 观察在高执行性效应下的稳定性
TEMPS=("0.01" "0.2" "0.4" "1.0")
for t in "${TEMPS[@]}"
do
    # SPRINT (SVRG)
    python credit.py dataset=credit nn_model=mlp ratio=1 log=true \
        optimizer=SVRG temperature=$t > "$LOG_DIR/q2_credit_sprint_t$t.log" 2>&1 &
    sleep 3

    # SGD-GD (SGD)
    python credit.py dataset=credit nn_model=mlp ratio=1 log=true \
        optimizer=SGD temperature=$t > "$LOG_DIR/q2_credit_sgd_t$t.log" 2>&1 &
    sleep 3
done

# --- 针对 Q3: 样本效率消融 (n_samples Ablation) ---
# 验证 SPRINT 是否能在远小于全量 (5000) 的 snapshot 下收敛
# 测试 100, 500, 1000 个样本作为 Snapshot
SAMPLES=("100" "500" "1000")
for ns in "${SAMPLES[@]}"
do
    python credit.py dataset=credit nn_model=mlp ratio=1 log=true \
        optimizer=SVRG n_samples=$ns > "$LOG_DIR/q3_credit_sprint_ns$ns.log" 2>&1 &
    sleep 3
done

# --- 针对 Variance Independence: Batch Size 实验 ---
# 核心论证：Batch Size 越小 -> 梯度方差越大。
# 理论声称 SPRINT 的误差邻域 Δ1 与方差无关 [cite: 678, 1434]
BATCH_SIZES=("1" "32" "128")
for bs in "${BATCH_SIZES[@]}"
do
    # SPRINT
    python credit.py dataset=credit nn_model=mlp ratio=1 log=true \
        optimizer=SVRG batch_size=$bs > "$LOG_DIR/var_credit_sprint_bs$bs.log" 2>&1 &
    sleep 3

    # SGD-GD (理论预估其误差会随 bs 减小而增大 [cite: 385])
    python credit.py dataset=credit nn_model=mlp ratio=1 log=true \
        optimizer=SGD batch_size=$bs > "$LOG_DIR/var_credit_sgd_bs$bs.log" 2>&1 &
    sleep 3
done

echo "Credit experiments are running in background. Monitoring process with 'wait'..."
wait
echo "All Credit rebuttal experiments completed. Results saved in $LOG_DIR."