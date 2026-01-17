# Experiment 1 Audit Report

**Generated**: 2026-01-17T00:49:17.975502
**Git Commit**: 77f3c72f2f11102d58af5210f1f967b70774177e

## Overall Result: ✓ PASS (4/4 checks passed)

## Detailed Results

### Claims vs Tables Consistency: ✓ PASS

```
Δ_V No Contraction: claim=0.1560, table=0.1560, diff=0.0000
Δ_V Contraction: claim=0.0380, table=0.0380, diff=0.0000
Δ_π No Contraction: claim=0.0063, table=0.0063, diff=0.0000
Δ_π Contraction: claim=0.0002, table=0.0002, diff=0.0000
R=0 Δ_V No Contraction: claim=1.0840, table=1.0840, diff=0.0000
R=0 Δ_V Contraction: claim=0.2250, table=0.2250, diff=0.0000
```

### Saturation Metric Sanity: ✓ PASS

```
OK: R=0 model_a saturation n=0 (correctly N/A)
OK: R=0 model_b saturation n=0 (correctly N/A)
OK: R=10 model_a saturation=1.00, n=900
OK: R=10 model_b saturation=1.00, n=900
OK: R=100 model_a saturation=0.00, n=900
OK: R=100 model_b saturation=0.00, n=900
```

### Label Correctness: ✓ PASS

```
OK: table_exp1_unroll_sensitivity.tex uses correct labels
OK: table_exp1_radius_sweep_main.tex uses correct labels
OK: table_exp1_radius_sweep_appendix.tex uses correct labels
```

### B0 Scoping: ✓ PASS

```
OK: R=0 claim includes B0 scoping
OK: B1 warning present
```

## Regeneration Verification

The paper-ready bundle was regenerated using:
```bash
buck2 run //buiksat_trm:make_paper_figures_exp1_final
```

All artifacts were validated for:
1. Numeric consistency between CLAIMS.md and LaTeX tables
2. Saturation metric sanity (R=0→N/A, R=10→100%, R=100→0%)
3. Correct labeling ("No Contraction" / "Contraction")
4. B0 scoping for R=0 claims with B1 warning

## Files in Bundle

- `AUDIT.md` (2443 bytes)
- `CLAIMS.md` (1386 bytes)
- `PROVENANCE.md` (2137 bytes)
- `exp1_paper_ready_bundle.zip` (136250 bytes)
- `fig_exp1_radius_sweep.pdf` (31860 bytes)
- `fig_exp1_radius_sweep_appendix.pdf` (31459 bytes)
- `fig_exp1_radius_sweep_main.pdf` (31988 bytes)
- `fig_exp1_unroll_sensitivity.pdf` (38090 bytes)
- `fig_exp1_unroll_sensitivity_appendix.pdf` (34668 bytes)
- `fig_exp1_unroll_sensitivity_main.pdf` (34249 bytes)
- `table_exp1_radius_sweep.md` (1528 bytes)
- `table_exp1_radius_sweep_appendix.md` (1077 bytes)
- `table_exp1_radius_sweep_appendix.tex` (877 bytes)
- `table_exp1_radius_sweep_main.md` (1085 bytes)
- `table_exp1_radius_sweep_main.tex` (909 bytes)
- `table_exp1_unroll_sensitivity.md` (948 bytes)
- `table_exp1_unroll_sensitivity.tex` (636 bytes)
