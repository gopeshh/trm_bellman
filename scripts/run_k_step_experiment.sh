#!/usr/bin/env bash
set -euo pipefail

# Adjust the effective K-step horizon via RLConfig.K in rl/config.py before running.
python upi_trm_train.py \
    --train-steps 200 \
    --batch-size 16 \
    --rollouts-per-step 1 \
    --max-edits 8 \
    --log-interval 10 \
    --eval-interval 50 \
    --eval-episodes 50
