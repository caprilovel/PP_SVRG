#!/usr/bin/env bash

# Parallel SVRG runner — launches 12 experiments (3 lr x 4 temperature)
# Usage: ./scripts/run_svrg_parallel.sh

set -euo pipefail

OPT="SVRG"
DATASET="MNIST"
MODEL="one_layer"
DEVICE="0"
RATIO=0.5
BATCH_SIZE=32
LOG_FLAG="--log"

# hyperparameter grid (3 * 4 = 12)
LRS=(0.001 0.003 0.01)
TEMPERATURES=(1 2 5 10)

OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"

echo "Launching SVRG experiments in parallel (total: 12)"

PIDS=()

for LR in "${LRS[@]}"; do
  for TEMP in "${TEMPERATURES[@]}"; do
    EXP_NAME="svrg_lr${LR}_temp${TEMP}"
    CMD=(python main.py --optimizer "$OPT" \
      --dataset "$DATASET" \
      --nn_model "$MODEL" \
      --lr "$LR" \
      --device "$DEVICE" \
      --temperature "$TEMP" \
      --ratio "$RATIO" \
      $LOG_FLAG \
      --batch_size "$BATCH_SIZE" \
      --exp_name "$EXP_NAME")

    LOGFILE="$OUTPUT_DIR/${EXP_NAME}.log"

    echo "Starting $EXP_NAME -> log: $LOGFILE"
    # start in background and capture pid; use nohup so processes survive terminal hangups
    nohup "${CMD[@]}" > "$LOGFILE" 2>&1 &
    PIDS+=("$!")

    # small stagger to reduce timestamp collisions
    sleep 0.2
  done
done

echo "Launched ${#PIDS[@]} jobs. PIDs: ${PIDS[*]}"

echo "Waiting for all jobs to finish..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "Process $pid exited with non-zero status"
done

echo "All SVRG experiments completed. Check $OUTPUT_DIR for logs and generated output folders."
