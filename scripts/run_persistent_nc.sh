#!/bin/bash
cd ~/fbsource/fbcode

CUDA_VISIBLE_DEVICES=0 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
    -- --config buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml \
    --seed 42 --dataset-paths buiksat_trm/data/sudoku-4x4-trivial --no-wandb \
    > /home/buiksat/trm_bellman/results/table3_baselines/persistent_nc_s42.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
    -- --config buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml \
    --seed 123 --dataset-paths buiksat_trm/data/sudoku-4x4-trivial --no-wandb \
    > /home/buiksat/trm_bellman/results/table3_baselines/persistent_nc_s123.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
    -- --config buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml \
    --seed 456 --dataset-paths buiksat_trm/data/sudoku-4x4-trivial --no-wandb \
    > /home/buiksat/trm_bellman/results/table3_baselines/persistent_nc_s456.log 2>&1 &

wait
echo "All persistent_nc experiments complete"
