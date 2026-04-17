# Phase 4: 2×2 Norm Ablation - Audit Report

## Summary

**Status:** ALL CHECKS PASSED ✓

**Total Checks:** 11
**Passed:** 11
**Failed:** 0

## Detailed Results

### Check 1: Config Integrity

**Status:** ✓ PASS

**Details:** All 4 condition configs exist and are valid YAML

### Check 2: Correct Toggles

**Status:** ✓ PASS

**Details:** All conditions have correct enable_contraction/disable_value_head_norm values

### Check 3: Seeds Present

**Status:** ✓ PASS

**Details:** All 12 condition/seed combinations found in summary

### Check 4: No Mislabeled

**Status:** ✓ PASS

**Details:** All checkpoint directories follow expected naming convention

### Check 5: Summary Exists

**Status:** ✓ PASS

**Details:** summary.json exists with correct structure

### Check 6: All Runs Present

**Status:** ✓ PASS

**Details:** All 12 condition×seed combinations present

### Check 7: CLAIMS.md Exists

**Status:** ✓ PASS

**Details:** CLAIMS.md exists with 771 chars

### Check 8: PROVENANCE.md Exists

**Status:** ✓ PASS

**Details:** PROVENANCE.md exists and references condition configs

### Check 9: Metric Bounds

**Status:** ✓ PASS

**Details:** All metrics within expected bounds

### Check 10: No NaN/Inf

**Status:** ✓ PASS

**Details:** No NaN or Inf values found

### Check 11: Statistical Validity

**Status:** ✓ PASS

**Details:** All conditions have 3 seeds for statistics

