# Success Rate Discrepancy Reconciliation

**Date:** 2026-01-19
**Root Cause Analysis Complete:** Yes

## Summary

The Exp5 "trivial suite success" rate (~5%) appears inconsistent with baseline success rates (~93%). This document explains the root cause and confirms this is **NOT a bug** but rather an expected consequence of training methodology differences.

## Root Cause

The nc_rdis checkpoints (used in Exp5) and Phase 4 checkpoints were trained using an evaluation dataset where **puzzles were already solved at episode start** (`initial=16.00/16`).

### Evidence from Training Logs

| Experiment | Initial Score | Eval Success | Issue |
|------------|---------------|--------------|-------|
| `training_exp2_lz095_s42.log` | 13.52/16 | 86-92% | ✅ Correct - uses unsolved puzzles |
| `training_nc_rdis_s41.log` | **16.00/16** | 100% | ⚠️ Trivial - uses solved puzzles |
| `training_phase4_*.log` | **16.00/16** | 100% | ⚠️ Trivial - uses solved puzzles |

### Why This Happened

1. **Evaluation dataset loader bug:** The nc_rdis and Phase 4 training configs use an evaluation dataset that loads `all__labels.npy` (solutions) instead of `all__inputs.npy` (puzzles).

2. **Models never learned to solve:** Since puzzles appeared "already solved" during evaluation, the 100% success rate was trivial and didn't indicate actual solving capability.

3. **Exp5 tests on real puzzles:** When Exp5 evaluates these checkpoints on actual unsolved puzzles (with 1-4 empty cells), the models perform poorly (~5%) because they never learned to fill empty cells.

## Comparison Table

| Setting | n_eval | R | Initial Score | Success | Explanation |
|---------|--------|---|---------------|---------|-------------|
| Exp2 training eval | 4 | 10 | 13.52 | 86-92% | Real puzzles, model learns |
| nc_rdis training eval | 2 | 0 | **16.00** | 100% | Already-solved puzzles |
| Exp5 (nc_rdis checkpoints) | 2 | 0 | ~12-14 | **~5%** | Real puzzles, model didn't learn |
| Phase 4 eval script | N/A | N/A | N/A | 0% | Placeholder function |

## Implications for Paper

### What to Report

1. **For stability metrics (primary focus):** The nc_rdis and Phase 4 checkpoints are valid. Argmax agreement, ΔV, L_preproj are measured on the z→z dynamics, not task success.

2. **For success rate claims:**
   - Do NOT claim "trivial suite success" for Exp5 without qualification
   - The ~5% represents models that were not trained for task success
   - If success rate is needed, use Exp2-style checkpoints (trained on real puzzles)

3. **Exp5 labeling fix:** The CLAIMS.md should note:
   - "Success rate measured for checkpoints trained with evaluation on solved puzzles"
   - "Low success (~5%) is expected as models did not learn to fill empty cells"

### Recommended Actions

1. ✅ **Keep stability metrics:** argmax agreement, ΔV, L_preproj are valid
2. ⚠️ **Remove or qualify success claims:** for Exp5/Phase 4
3. 🔧 **Fix training eval dataset:** for future experiments (use `all__inputs.npy` not `all__labels.npy`)

## Conclusion

The discrepancy is explained by:
- **Different training:** Exp2 models were trained with real puzzles; nc_rdis/Phase 4 used solved puzzles
- **Different evaluation:** Exp5 evaluates on real puzzles, revealing the training gap

This is **not a bug in Exp5 evaluation** but rather **correct behavior** exposing that nc_rdis checkpoints were trained on a trivial task.

---

## Technical Details

### Location of Issue

The evaluation dataset setup in training configs needs to ensure:
```yaml
# Correct: use inputs (puzzles with empty cells)
eval_dataset: data/sudoku-4x4-trivial/test/all__inputs.npy

# WRONG: using labels (solved puzzles)
eval_dataset: data/sudoku-4x4-trivial/test/all__labels.npy
```

### Verification Commands

Check initial scores in any training log:
```bash
grep "initial=" training_*.log | head -5
# initial=16.00 = already solved (bad)
# initial=13.52 = real puzzles (good)
```
