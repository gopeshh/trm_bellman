#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

CONFIG=$1

python upi_trm_train.py \
    --config "$CONFIG" \
    --train-steps 200 \
    --batch-size 16 \
    --rollouts-per-step 1 \
    --max-edits 8 \
    --log-interval 10 \
    --eval-interval 50 \
    --eval-episodes 50 \
    --seed 0

