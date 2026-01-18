# Experiment 2 Provenance

**Generated**: 2026-01-17T15:21:07.643874
**Git Commit**: 8c6994a44bbe7204d1c883b9e93a3c4b1f2c3cd9

## Sweep Configuration

| Target $L_z$ | Config | Seeds |
|--------------|--------|-------|
| 0.9 | `configs/exp2_contraction_sweep/target_lz_090.yaml` | [41, 42, 43] |
| 0.95 | `configs/exp2_contraction_sweep/target_lz_095.yaml` | [41, 42, 43] |
| 0.99 | `configs/exp2_contraction_sweep/target_lz_099.yaml` | [41, 42, 43] |
| 0.999 | `configs/exp2_contraction_sweep/target_lz_0999.yaml` | [41, 42, 43] |

## Checkpoints

### Target $L_z$ = 0.9

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0900/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0900/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0900/seed43/model_step_5000.pt`

### Target $L_z$ = 0.95

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed43/model_step_5000.pt`

### Target $L_z$ = 0.99

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_099/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_099/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_099/seed43/model_step_5000.pt`

### Target $L_z$ = 0.999

- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0999/seed41/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0999/seed42/model_step_5000.pt`
- `/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_0999/seed43/model_step_5000.pt`

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n_eval | 16 (8× depth) |
| Batch | B0 (initial states) |

## Regeneration Command

```bash
buck2 run //buiksat_trm:make_paper_figures_exp2
```

## Exp2b Diagnostics

**Generated**: 2026-01-17

### Purpose

Investigate WHY achieved $\hat{L}_z$ saturates at ~0.23 across all target $L_z$ values.

### Method

For each checkpoint, compute:
- $\hat{L}_{pre-proj}$: Lipschitz constant WITHOUT projection
- $\hat{L}_{post-proj}$: Lipschitz constant WITH projection at radius R
- Projection active rate: fraction where $\|f(z)\| > R$

Evaluated at R ∈ {10, 100, 1000, ∞ (disabled)}.

### Regeneration Command

```bash
buck2 run //buiksat_trm:diagnose_contraction_saturation
```

### Output Files

- `DIAGNOSTICS_saturation.json`: Full results
- `DIAGNOSTICS_saturation.md`: Summary table

## Exp2c-Lite: Unmasking Evaluation

**Generated**: 2026-01-17

### Purpose

Test whether the contraction "dial" works when projection is not dominating by evaluating existing checkpoints at different radii.

### Method

For each checkpoint, evaluate at R_eval ∈ {10, 100, disabled}:
- $\hat{L}_{pre-proj}$: Lipschitz constant WITHOUT projection
- $\hat{L}_{post-proj}$: Lipschitz constant WITH projection at R_eval
- Projection active rate
- Unroll sensitivity metrics (ΔV, Δπ, argmax agreement) at n_train=2 vs n_eval={4, 8, 16}

Decision gates:
- G1: projection_active_rate < 20%
- G2: L_preproj spread ≥ 0.08
- G3: Lower L_preproj ⇒ lower ΔV/Δπ, higher argmax agreement

### Results

| Gate | R=100 | R=disabled |
|------|-------|------------|
| G1 | ✅ PASSED (0%) | ✅ PASSED (0%) |
| G2 | ❌ FAILED (0.064) | ❌ FAILED (0.064) |
| G3 | INCONCLUSIVE | INCONCLUSIVE |

**Path**: B (Negative result - dial has insufficient range)

### Regeneration Command

```bash
buck2 run //buiksat_trm:eval_exp2c_lite
```

### Output Files

- `../exp2c/DIAGNOSTICS_exp2c_lite.json`: Full results
- `../exp2c/DIAGNOSTICS_exp2c_lite.md`: Summary table
- `../exp2c/GATES.md`: Decision gate analysis

