# Combined Ablation: Persistent-z + No Contraction

**Experiment Date:** 2026-01-09
**Machine:** Machine 2 (2 GPUs)
**Dataset:** buiksat_trm/data/sudoku-4x4-trivial
**Training Steps:** 5000
**Config:** `configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml`

## Ablation Settings

| Setting | Value |
|---------|-------|
| `episodic_latent` | `false` (persistent-z) |
| `enable_contraction` | `false` (no spectral normalization) |
| `use_feasibility_checker` | `true` |
| `feasibility_violation_weight` | 2.0 |
| `feasibility_zerocand_weight` | 5.0 |
| `stop_action_mode` | `disabled` |
| `K` | 1 |
| `inner_unroll_n` | 2 |

## Results Summary

| Seed | Final Success | Peak Success | Peak Step | Mean Score (final) |
|------|---------------|--------------|-----------|-------------------|
| 42   | 96.0%         | 96.0%        | 5000      | 15.960            |
| 123  | 92.0%         | 92.0%        | 5000      | 15.920            |
| 456  | 92.0%         | 92.0%        | 5000      | 15.880            |

### Aggregate Statistics

| Metric | Mean | Std |
|--------|------|-----|
| Final Success Rate | 93.3% | 2.3% |
| Final Mean Score | 15.920 | 0.040 |

## Notes

- All seeds achieved their peak success rate at or near the final step (5000)
- This ablation combines two modifications from the baseline:
  1. Persistent latent state (z is initialized once per episode, not per step)
  2. No contraction enforcement (spectral normalization disabled)
- Success is defined as: `filled == 16 AND violations == 0` (solution-independent)
- Checker score: `score = filled - 2*violations - 5*zeroCand`

## Log Files

- `42_20260109_141330.log`
- `123_20260109_141330.log`
- `456_20260109_144028.log`
