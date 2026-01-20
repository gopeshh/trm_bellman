#!/bin/bash
cd ~/fbsource/fbcode

CUDA_VISIBLE_DEVICES=1 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
    -- --config buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml \
    --seed 123 --dataset-paths buiksat_trm/data/sudoku-4x4-trivial --no-wandb \
    > /home/buiksat/trm_bellman/results/table3_baselines/episodic_c_clean_s123.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
    -- --config buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml \
    --seed 456 --dataset-paths buiksat_trm/data/sudoku-4x4-trivial --no-wandb \
    > /home/buiksat/trm_bellman/results/table3_baselines/episodic_c_clean_s456.log 2>&1 &

wait
echo "Both experiments complete"
