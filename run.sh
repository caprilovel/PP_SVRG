#!/bin/bash

# 固定参数
OPTIMIZERS=("SGD" "SVRG")
DATASET="MNIST"
MODEL="one_layer"
DEVICE=0
RATIO=0.5
BATCH_SIZE=32
LOG_FLAG="--log"

# 超参数搜索空间
LRS=(0.001 0.003 0.01)
TEMPERATURES=(1 2 5 10)

# 每个实验之间暂停几秒，防止时间戳冲突
SLEEP_TIME=5

for OPT in "${OPTIMIZERS[@]}"; do
  for LR in "${LRS[@]}"; do
    for TEMP in "${TEMPERATURES[@]}"; do
      echo "==========================================="
      echo "Running: optimizer=$OPT, lr=$LR, temperature=$TEMP"
      echo "==========================================="

      CMD="python main.py --optimizer $OPT \
        --dataset $DATASET \
        --nn_model $MODEL \
        --lr $LR \
        --device $DEVICE \
        --temperature $TEMP \
        --ratio $RATIO \
        $LOG_FLAG"

      # SVRG特有参数
      if [ "$OPT" == "SVRG" ]; then
        CMD="$CMD --batch_size $BATCH_SIZE"
      fi

      echo "Command: $CMD"
      eval $CMD

      echo "Sleeping for $SLEEP_TIME seconds..."
      sleep $SLEEP_TIME
    done
  done
done
