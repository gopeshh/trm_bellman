# Hyperparameter Tuning for Baselines on Hard 4×4 Sudoku

## Goal
Historical masked-control sweep for PPO/A2C/DQN on the 6-8 empties dataset. This is not the final locked no-mask Table 1 baseline package.

## Locked No-Mask Table 1 Results
- TRM+PPO: 32.0% ± 15.3% (10 seeds)
- TRM+A2C: 0.0% (10 seeds)
- TRM+DQN: 0.0% (10 seeds)
- UPI-TRM: 57.4% ± 12.2% (10 seeds)

## What This Directory Contains
- A supplementary masked-protocol tuning sweep.
- Single-seed and small multi-seed hyperparameter explorations that all remained at 0% success in that masked control.
- Historical notes that are still useful for ablation context but should not be cited as the final no-mask baseline outcome.

## Tuning Strategy

### Phase 1: Single-seed sweeps (seed 42)
Test most impactful hyperparameters per algorithm.

### Phase 2: Multi-seed validation
Test top configs across seeds 42, 123, 456.

## Hyperparameter Sweep Design

### PPO Sweep
1. Learning rate: 3e-4, 1e-4, 5e-5, 1e-5
2. Entropy coefficient: 0.01, 0.05, 0.1, 0.2
3. PPO epochs: 4, 8, 16
4. Num steps: 64, 128, 256
5. Inner unroll: 2, 4, 8

### A2C Sweep
1. Learning rate: 3e-4, 1e-4, 5e-5
2. Entropy coefficient: 0.01, 0.05, 0.1, 0.2
3. Num steps: 16, 32, 64, 128
4. GAE lambda: 0.9, 0.95, 0.99
5. Inner unroll: 2, 4, 8

### DQN Sweep
1. Learning rate: 1e-3, 5e-4, 1e-4, 5e-5
2. Epsilon decay: 0.1, 0.2, 0.4, 0.6 (fraction of training)
3. Buffer size: 5000, 10000, 50000
4. Target update: 100, 200, 500
5. Inner unroll: 2, 4, 8
