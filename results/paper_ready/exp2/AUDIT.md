# Experiment 2 Audit Report

**Generated**: 2026-01-17T15:21:27.936332
**Git Commit**: 8c6994a44bbe7204d1c883b9e93a3c4b1f2c3cd9

## Overall Result: ✓ PASS (3/3 checks passed)

## Detailed Results

### File Existence: ✓ PASS

```
OK: fig_exp2_stability_dial.pdf (25082 bytes)
OK: table_exp2_contraction_sweep.tex (797 bytes)
OK: CLAIMS.md (1837 bytes)
OK: PROVENANCE.md (2067 bytes)
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

### Dial Behavior: ✓ PASS

```
OK: Δ_V(0.9)=0.382 > Δ_V(0.95)=0.140 (expected relationship)
OK: Δ_V(0.95)=0.140 > Δ_V(0.99)=0.067 (expected relationship)
NOTE: Δ_V(0.99)=0.067 <= Δ_V(0.999)=0.145 (non-monotonic)
OK: Overall trend as expected (Δ_V: 0.382 → 0.145)
INFO: Achieved Lz range: 0.217 to 0.242 (span: 0.025)
INFO: Achieved Lz shows saturation (< 0.05 variation)
```

## Regeneration Command

```bash
buck2 run //buiksat_trm:make_paper_figures_exp2
buck2 run //buiksat_trm:audit_exp2_paper_ready
```

## Files in Bundle

- `AUDIT.md` (1645 bytes)
- `CLAIMS.md` (1837 bytes)
- `PROVENANCE.md` (2067 bytes)
- `exp2_paper_ready_bundle.zip` (18932 bytes)
- `fig_exp2_stability_dial.pdf` (25082 bytes)
- `table_exp2_contraction_sweep.tex` (797 bytes)
