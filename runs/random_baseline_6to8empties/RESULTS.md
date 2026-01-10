# Random Baseline Results: 4x4 Sudoku (6-8 Empties)

**Date**: 2026-01-09
**Dataset**: `data/sudoku-4x4-easy_6to8empties` (1000 puzzles, 8-10 clues = 6-8 empties)
**Evaluation**: 200 episodes per seed, max 16 edits

## Results

| Seed | Success Rate | Mean Score | Filled (mean) | Violations (mean) | Zero Cand (mean) |
|------|--------------|------------|---------------|-------------------|------------------|
| 42   | 0.000        | -3.59      | 16.00         | 9.79              | 0.00             |
| 123  | 0.000        | -3.59      | 16.00         | 9.79              | 0.00             |
| 456  | 0.000        | -3.59      | 16.00         | 9.79              | 0.00             |
| **Mean** | **0.000** | **-3.59**  | **16.00**     | **9.79**          | **0.00**         |

## Interpretation

Random success dropped dramatically compared to the trivial 4x4 setting (1-4 empties). With 6-8 empty cells, the random policy:

1. **Achieves 0% success rate** - Unable to solve any puzzles by chance
2. **Fills all cells** (16.00 filled) but with **~10 constraint violations on average**
3. **Negative feasibility score** (-3.59) due to violation penalty: score = filled - 2*violations - 5*zeroCand

This validates the harder dataset as a meaningful benchmark where success rate can discriminate between random and learned policies.
