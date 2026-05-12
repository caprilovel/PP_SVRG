#!/bin/bash

LOG_DIR="./rebuttal_results"
mkdir -p $LOG_DIR

echo "Starting Parallel Rebuttal Experiments..."

BASE="python main.py dataset=MNIST nn_model=one_layer lr=0.003 ratio=0.1 batch_size=100 device=cuda:2 log=true"

# --- Q2: Temperature Test ---
TEMPS=("20" "80" "100")
for t in "${TEMPS[@]}"
do
    $BASE optimizer=SVRG temperature=$t > "$LOG_DIR/q2_t$t.log" 2>&1 &
    sleep 2
    $BASE optimizer=SGD temperature=$t > "$LOG_DIR/q2_sgd_t$t.log" 2>&1 &
    sleep 2
done

# --- Q3: n_samples Ablation ---
SAMPLES=("500" "1000")
for ns in "${SAMPLES[@]}"
do
    $BASE optimizer=SVRG temperature=50 n_samples=$ns > "$LOG_DIR/q3_ns$ns.log" 2>&1 &
    sleep 2
done

# --- Variance Independence: Batch Size ---
BATCH_SIZES=("1" "16" "64")
for bs in "${BATCH_SIZES[@]}"
do
    $BASE optimizer=SVRG temperature=50 batch_size=$bs > "$LOG_DIR/var_sprint_bs$bs.log" 2>&1 &
    sleep 2
    $BASE optimizer=SGD temperature=50 batch_size=$bs > "$LOG_DIR/var_sgd_bs$bs.log" 2>&1 &
    sleep 2
done

# --- Q4: Validation Metric ---
$BASE optimizer=SVRG temperature=50 > "$LOG_DIR/q4_val.log" 2>&1 &

echo "All experiments are running in background. Use 'top' or 'ps' to monitor."
wait
echo "All experiments completed."
