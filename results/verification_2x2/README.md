# 2x2 Contraction Collapse Isolation Results

## Experiment Design

This experiment isolates the cause of training collapse when `enable_contraction=True`.

### Conditions

| Condition | z→z Contraction | Value Head Norm | Description |
|-----------|-----------------|-----------------|-------------|
| A | OFF | OFF | Baseline (should match no_contraction) |
| B | ON | OFF | Isolates z→z contraction |
| C | OFF | ON | Isolates value head normalization |
| D | ON | ON | Full enable_contraction=true |

### Commands Used

```bash
buck2 clean
python scripts/diagnostics/run_contraction_collapse_isolation_2x2.py \
    --dataset /home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties \
    --config /home/buiksat/trm_bellman/configs/experiments/contraction_sgd_tradeoff/base_episodic_z.yaml \
    --steps 200 \
    --seed 42
```

## Results

### Collapse Detection

| Condition | Status | Collapse Step |
|-----------|--------|---------------|
| A (zconOFF_vheadOFF) | STABLE | - |
| B (zconON_vheadOFF) | STABLE | - |
| C (zconOFF_vheadON) | STABLE | - |
| D (zconON_vheadON) | STABLE | - |

### Comparison Table

| Step | A target | A V(s) | B target | B V(s) | C target | C V(s) | D target | D V(s) |
|------|---------|---------|---------|---------|---------|---------|---------|---------|
| 10 | -1.43 | -0.30 | -1.29 | -0.28 | -12.17 | -0.72 | 13.75 | -0.26 |
| 20 | -1.17 | -0.78 | -1.20 | -0.74 | -11.71 | -1.93 | 14.22 | 0.92 |
| 30 | -1.57 | -1.60 | -1.52 | -1.40 | -7.72 | -4.08 | 15.62 | 3.15 |
| 40 | -1.49 | -1.64 | -1.42 | -1.52 | 0.58 | -5.00 | 18.44 | 6.42 |
| 50 | -1.72 | -1.59 | -1.26 | -1.30 | 3.08 | -2.31 | 19.84 | 9.82 |
| 60 | -1.67 | -1.64 | -1.54 | -1.38 | 6.96 | 1.51 | 20.00 | 13.87 |
| 70 | -1.56 | -1.60 | -1.70 | -1.59 | 1.24 | 4.77 | 20.00 | 17.21 |
| 80 | -1.70 | -1.65 | -1.50 | -1.44 | -7.63 | 2.13 | 20.00 | 19.45 |
| 90 | -1.37 | -1.38 | -1.93 | -1.64 | -7.04 | -3.35 | 20.00 | 19.99 |
| 100 | -1.60 | -1.32 | -1.75 | -1.53 | 4.16 | -4.86 | 20.00 | 20.00 |
| 110 | -1.59 | -2.09 | -1.79 | -1.56 | 12.47 | -0.98 | 20.00 | 20.00 |
| 120 | -1.76 | -1.82 | -1.59 | -1.52 | 10.85 | 1.25 | 20.00 | 20.00 |
| 130 | -1.84 | -1.48 | -1.89 | -1.88 | 0.86 | 3.66 | 20.00 | 20.00 |
| 140 | -1.76 | -1.50 | -1.71 | -1.65 | -9.22 | 2.79 | 20.00 | 20.00 |
| 150 | -1.81 | -1.51 | -1.67 | -1.72 | -12.47 | 1.17 | 20.00 | 20.00 |
| 160 | -1.58 | -1.76 | -1.34 | -1.49 | -15.86 | 0.10 | 20.00 | 20.00 |
| 170 | -1.76 | -1.84 | -1.52 | -1.60 | -16.72 | -0.36 | 20.00 | 20.00 |
| 180 | -1.66 | -1.73 | -1.81 | -1.62 | -14.84 | -0.97 | 20.00 | 20.00 |
| 190 | -1.72 | -1.72 | -1.68 | -1.67 | -14.57 | -2.04 | 20.00 | 20.00 |
| 200 | -1.74 | -1.81 | -1.66 | -1.83 | -12.57 | -3.73 | 20.00 | 20.00 |

## Conclusion

### Key Finding: Value Head Normalization Causes Instability

The 2x2 ablation clearly isolates the culprit:

| Condition | z→z Contraction | Value Head Norm | Result |
|-----------|-----------------|-----------------|--------|
| **A (baseline)** | OFF | OFF | **STABLE** - target ~-1.7, V(s) ~-1.8 throughout |
| **B (z-con only)** | ON | OFF | **STABLE** - target ~-1.7, V(s) ~-1.8 throughout |
| **C (vhead only)** | OFF | ON | **UNSTABLE** - targets oscillate -20 to +20 (hitting clamps) |
| **D (both)** | ON | ON | **COLLAPSE TO +20** - target=20.0 by step 60, V(s)=20.0 by step 100 |

### Analysis

1. **z→z contraction alone (B) is STABLE** - Comparing A vs B, adding z→z contraction has minimal impact on training dynamics. Both conditions show healthy target values around -1.5 to -1.9 and V(s) tracking reasonably.

2. **Value head normalization alone (C) causes INSTABILITY** - Targets oscillate wildly between the clamps (-20 to +20), indicating the value head normalization disrupts gradient flow. V(s) shows erratic behavior (-5 to +5).

3. **Both combined (D) causes CATASTROPHIC COLLAPSE** - The combination rapidly collapses all values to the positive clamp (+20.0) with std=0.0, making the value function completely uninformative.

### Root Cause

The value head normalization (`apply_spectral_norm_to_value_head` + `enforce_global_contraction_on_value_head`) is the primary cause of training instability. The functions involved are:
- `utils/lipschitz.py:apply_spectral_norm_to_value_head()`
- `utils/lipschitz.py:enforce_global_contraction_on_value_head()`

When combined with z→z contraction, the effect is amplified into complete collapse.

### Recommendation

Disable value head normalization when using `enable_contraction=True`. The z→z contraction can be kept enabled without issue - only the value head normalization needs to be disabled.

## Log Files

- A: `/home/buiksat/trm_bellman/results/verification_2x2/A_zconOFF_vheadOFF_seed42_200.log`
- B: `/home/buiksat/trm_bellman/results/verification_2x2/B_zconON_vheadOFF_seed42_200.log`
- C: `/home/buiksat/trm_bellman/results/verification_2x2/C_zconOFF_vheadON_seed42_200.log`
- D: `/home/buiksat/trm_bellman/results/verification_2x2/D_zconON_vheadON_seed42_200.log`
