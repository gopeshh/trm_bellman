# Hyperparameter Tuning for Baselines on Hard 4×4 Sudoku

## Goal
Improve PPO/A2C/DQN from 0% to >0% success on 6-8 empties dataset.

## Current Baseline Performance
- PPO: 0% (3 seeds)
- A2C: 0% (3 seeds)
- DQN: 0% (3 seeds)
- UPI-TRM: 37-57% (reference)

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
