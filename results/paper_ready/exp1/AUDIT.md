# Experiment 1 Audit Report

**Generated**: 2026-04-05T21:53:48.786449
**Git Commit**: 931d94c696c909841f3f864ae8b1fa4bb50839cf

## Overall Result: ✓ PASS (5/5 checks passed)

## Detailed Results

### Claims vs Tables Consistency: ✓ PASS

```
Δ_V No Contraction: claim=0.1560, table=0.1560, diff=0.0000
Δ_V Contraction: claim=0.0380, table=0.0380, diff=0.0000
Δ_π No Contraction: claim=0.0063, table=0.0063, diff=0.0000
Δ_π Contraction: claim=0.0002, table=0.0002, diff=0.0000
R=0 Δ_V No Contraction: claim=1.0780, table=1.0780, diff=0.0000
R=0 Δ_V Contraction: claim=0.2400, table=0.2400, diff=0.0000
```

### Saturation Metric Sanity: ✓ PASS

```
OK: R=0 model_a saturation n=0 (correctly N/A)
OK: R=0 model_b saturation n=0 (correctly N/A)
OK: R=10 model_a saturation=1.00, n=300
OK: R=10 model_b saturation=1.00, n=300
OK: R=100 model_a saturation=0.00, n=300
OK: R=100 model_b saturation=0.00, n=300
```

### Label Correctness: ✓ PASS

```
OK: table_exp1_radius_sweep_appendix.tex uses correct labels
OK: table_exp1_radius_sweep_main.tex uses correct labels
OK: table_exp1_unroll_sensitivity.tex uses correct labels
```

### Paper Repo Table Sync: ✓ PASS

```
OK: paper repo table matches for table_exp1_unroll_sensitivity.tex
OK: paper repo table matches for table_exp1_radius_sweep_main.tex
OK: paper repo table matches for table_exp1_radius_sweep_appendix.tex
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
4. Canonical LaTeX tables mirrored into the paper repo
5. B0 scoping for R=0 claims with B1 warning

## Files in Bundle

- `AUDIT.md` (2742 bytes)
- `CLAIMS.md` (1471 bytes)
- `PAPER_INSERT_SNIPPET.tex` (2871 bytes)
- `PROVENANCE.md` (2137 bytes)
- `exp1_paper_ready_bundle.zip` (137907 bytes)
- `fig_exp1_radius_sweep.pdf` (31911 bytes)
- `fig_exp1_radius_sweep_appendix.pdf` (31581 bytes)
- `fig_exp1_radius_sweep_main.pdf` (32083 bytes)
- `fig_exp1_unroll_sensitivity.pdf` (38090 bytes)
- `fig_exp1_unroll_sensitivity_appendix.pdf` (34668 bytes)
- `fig_exp1_unroll_sensitivity_main.pdf` (34249 bytes)
- `table_exp1_radius_sweep.md` (1652 bytes)
- `table_exp1_radius_sweep_appendix.md` (1186 bytes)
- `table_exp1_radius_sweep_appendix.tex` (877 bytes)
- `table_exp1_radius_sweep_main.md` (1212 bytes)
- `table_exp1_radius_sweep_main.tex` (953 bytes)
- `table_exp1_unroll_sensitivity.md` (948 bytes)
- `table_exp1_unroll_sensitivity.tex` (636 bytes)
