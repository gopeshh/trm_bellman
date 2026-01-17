# Experiment 2 Audit Report

**Generated**: 2026-01-17T10:55:48.323985
**Git Commit**: 1debd375c159b8e8de62ed734f0d7efd117760ca

## Overall Result: ✗ FAIL (2/3 checks passed)

## Detailed Results

### File Existence: ✓ PASS

```
OK: fig_exp2_stability_dial.pdf (24132 bytes)
OK: table_exp2_contraction_sweep.tex (797 bytes)
OK: CLAIMS.md (1438 bytes)
OK: PROVENANCE.md (1477 bytes)
```

### Claims vs Table: ✓ PASS

```
target_Lz=0.9 achieved_lz: OK (diff=0.0000)
target_Lz=0.9 success_rate: OK (diff=0.0000)
target_Lz=0.9 delta_V: OK (diff=0.0000)
target_Lz=0.95 achieved_lz: OK (diff=0.0000)
target_Lz=0.95 success_rate: OK (diff=0.0000)
target_Lz=0.95 delta_V: OK (diff=0.0000)
target_Lz=0.99 achieved_lz: OK (diff=0.0000)
target_Lz=0.99 success_rate: OK (diff=0.0000)
target_Lz=0.99 delta_V: OK (diff=0.0000)
target_Lz=0.999 achieved_lz: OK (diff=0.0000)
target_Lz=0.999 success_rate: OK (diff=0.0000)
target_Lz=0.999 delta_V: OK (diff=0.0000)
```

### Dial Monotonicity: ✗ FAIL

```
OK: Δ_V(0.9)=0.172 > Δ_V(0.95)=0.114 (stronger contraction = more stable)
OK: Δ_V(0.95)=0.114 > Δ_V(0.99)=0.001 (stronger contraction = more stable)
WARN: Δ_V(0.99)=0.001 <= Δ_V(0.999)=0.094 (non-monotonic)
FAIL: Overall trend reversed (Δ_V: 0.172 → 0.094)
```

## Regeneration Command

```bash
buck2 run //buiksat_trm:make_paper_figures_exp2
buck2 run //buiksat_trm:audit_exp2_paper_ready
```

## Files in Bundle

- `CLAIMS.md` (1438 bytes)
- `PROVENANCE.md` (1477 bytes)
- `fig_exp2_stability_dial.pdf` (24132 bytes)
- `table_exp2_contraction_sweep.tex` (797 bytes)
