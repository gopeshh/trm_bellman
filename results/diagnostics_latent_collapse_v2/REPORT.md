# Latent Collapse Diagnostic V2 Report

## Experiment Overview

**Goal**: Isolate the effect of z→z contraction from value-head normalization, and measure
pre-projection vs post-projection latents.

**2×2 Factorial Design**:
- **Axis 1**: z→z contraction (zcon_OFF vs zcon_ON with target_Lz=0.9)
- **Axis 2**: value-head normalization (vhead_OFF vs vhead_ON with target_Lv=1.0)

**Conditions**:
| Condition | z→z Contraction | Value Head Norm |
|-----------|-----------------|-----------------|
| A         | OFF             | OFF             |
| B         | ON              | OFF             |
| C         | OFF             | ON              |
| D         | ON              | ON              |

**Setup**:
- Seed: 42
- Dataset: 4×4 Sudoku (trivial, 1–4 empties)
- Batch size: 128 states
- Projection radius: 10.0

---

## Pre-Projection Latent Norms (z_H)

Shows ||z|| BEFORE applying the ball projection. If contraction causes collapse,
we expect lower norms / less diversity here.

| n | Condition | mean ||z|| | std ||z|| | Total Var | Cos Sim |
|---|-----------|-----------|---------|-----------|---------|
| 1 | A | 31.9999 | 0.0000 | 131.9405 | 0.8693 |
| 1 | B | 31.9998 | 0.0000 | 170.4656 | 0.8313 |
| 1 | C | 31.9999 | 0.0000 | 131.9405 | 0.8674 |
| 1 | D | 31.9998 | 0.0000 | 170.4656 | 0.8341 |
| 2 | A | 31.9999 | 0.0000 | 125.7685 | 0.8823 |
| 2 | B | 31.9998 | 0.0000 | 337.1213 | 0.6679 |
| 2 | C | 31.9999 | 0.0000 | 125.7685 | 0.8776 |
| 2 | D | 31.9998 | 0.0000 | 337.1213 | 0.6706 |
| 4 | A | 31.9999 | 0.0000 | 171.4404 | 0.8158 |
| 4 | B | 31.9998 | 0.0000 | 479.5629 | 0.5267 |
| 4 | C | 31.9999 | 0.0000 | 171.4404 | 0.8347 |
| 4 | D | 31.9998 | 0.0000 | 479.5629 | 0.5295 |
| 8 | A | 31.9999 | 0.0000 | 356.9307 | 0.6554 |
| 8 | B | 31.9998 | 0.0000 | 578.7440 | 0.4246 |
| 8 | C | 31.9999 | 0.0000 | 356.9307 | 0.6349 |
| 8 | D | 31.9998 | 0.0000 | 578.7440 | 0.4338 |
| 16 | A | 31.9999 | 0.0000 | 814.0424 | 0.1958 |
| 16 | B | 31.9998 | 0.0000 | 608.6821 | 0.4034 |
| 16 | C | 31.9999 | 0.0000 | 814.0424 | 0.1997 |
| 16 | D | 31.9998 | 0.0000 | 608.6821 | 0.3955 |

## Post-Projection Latent Norms (z_H)

Shows ||z|| AFTER applying the ball projection (radius=10.0).

| n | Condition | mean ||z|| | std ||z|| | Total Var | Cos Sim |
|---|-----------|-----------|---------|-----------|---------|
| 1 | A | 10.0000 | 0.0000 | 12.8849 | 0.8677 |
| 1 | B | 10.0000 | 0.0000 | 16.6472 | 0.8324 |
| 1 | C | 10.0000 | 0.0000 | 12.8849 | 0.8701 |
| 1 | D | 10.0000 | 0.0000 | 16.6472 | 0.8312 |
| 2 | A | 10.0000 | 0.0000 | 12.2822 | 0.8777 |
| 2 | B | 10.0000 | 0.0000 | 32.9223 | 0.6706 |
| 2 | C | 10.0000 | 0.0000 | 12.2822 | 0.8786 |
| 2 | D | 10.0000 | 0.0000 | 32.9223 | 0.6652 |
| 4 | A | 10.0000 | 0.0000 | 16.7423 | 0.8353 |
| 4 | B | 10.0000 | 0.0000 | 46.8328 | 0.5361 |
| 4 | C | 10.0000 | 0.0000 | 16.7423 | 0.8249 |
| 4 | D | 10.0000 | 0.0000 | 46.8328 | 0.5259 |
| 8 | A | 10.0000 | 0.0000 | 34.8568 | 0.6411 |
| 8 | B | 10.0000 | 0.0000 | 56.5185 | 0.4328 |
| 8 | C | 10.0000 | 0.0000 | 34.8568 | 0.6574 |
| 8 | D | 10.0000 | 0.0000 | 56.5185 | 0.4365 |
| 16 | A | 10.0000 | 0.0000 | 79.4969 | 0.2099 |
| 16 | B | 10.0000 | 0.0000 | 59.4422 | 0.3991 |
| 16 | C | 10.0000 | 0.0000 | 79.4969 | 0.1899 |
| 16 | D | 10.0000 | 0.0000 | 59.4422 | 0.4010 |

## Value Head Statistics

Shows V(z) statistics. Comparing B vs A isolates z→z effect on value (same value head).

| n | Condition | V_mean | V_std | V_min | V_max | V_var |
|---|-----------|--------|-------|-------|-------|-------|
| 1 | A | -0.0997 | 0.1182 | -0.3823 | 0.1721 | 0.013963 |
| 1 | B | -0.0928 | 0.1224 | -0.4007 | 0.1744 | 0.014978 |
| 1 | C | 2.0961 | 49.8556 | -50.0000 | 50.0000 | 2485.580322 |
| 1 | D | 10.4156 | 48.6821 | -50.0000 | 50.0000 | 2369.944092 |
| 2 | A | -0.0919 | 0.1153 | -0.3845 | 0.1800 | 0.013297 |
| 2 | B | -0.0859 | 0.1231 | -0.4142 | 0.1904 | 0.015152 |
| 2 | C | 9.4692 | 49.1644 | -50.0000 | 50.0000 | 2417.141113 |
| 2 | D | 12.0900 | 48.2905 | -50.0000 | 50.0000 | 2331.972656 |
| 4 | A | -0.0984 | 0.1180 | -0.3914 | 0.1979 | 0.013923 |
| 4 | B | -0.0807 | 0.1233 | -0.4179 | 0.1966 | 0.015212 |
| 4 | C | 2.3821 | 50.1017 | -50.0000 | 50.0000 | 2510.184814 |
| 4 | D | 11.4166 | 47.8462 | -50.0000 | 50.0000 | 2289.262207 |
| 8 | A | -0.0797 | 0.1193 | -0.3599 | 0.2257 | 0.014242 |
| 8 | B | -0.0754 | 0.1256 | -0.4169 | 0.2045 | 0.015787 |
| 8 | C | 6.9144 | 49.5965 | -50.0000 | 50.0000 | 2459.811279 |
| 8 | D | 9.1162 | 49.0152 | -50.0000 | 50.0000 | 2402.494141 |
| 16 | A | -0.0808 | 0.1247 | -0.3772 | 0.2274 | 0.015561 |
| 16 | B | -0.0764 | 0.1272 | -0.4202 | 0.2099 | 0.016188 |
| 16 | C | 3.8112 | 49.9295 | -50.0000 | 50.0000 | 2492.957031 |
| 16 | D | 9.1971 | 49.1102 | -50.0000 | 50.0000 | 2411.812012 |

---

## Analysis (at n=16)

### Effect of z→z Contraction (B vs A, holding value head constant = OFF)

| Metric | A (zcon OFF) | B (zcon ON) | B/A Ratio |
|--------|--------------|-------------|-----------|
| pre ||z_H|| mean | 31.9999 | 31.9998 | 1.000 |
| pre Total Var | 814.0424 | 608.6821 | 0.748 |
| pre Cos Sim | 0.1958 | 0.4034 | +0.2077 (diff) |
| V_var | 0.015561 | 0.016188 | 1.040 |

### Effect of z→z Contraction (D vs C, holding value head constant = ON)

| Metric | C (zcon OFF) | D (zcon ON) | D/C Ratio |
|--------|--------------|-------------|-----------|
| pre ||z_H|| mean | 31.9999 | 31.9998 | 1.000 |
| pre Total Var | 814.0424 | 608.6821 | 0.748 |
| pre Cos Sim | 0.1997 | 0.3955 | +0.1959 (diff) |
| V_var | 2492.957031 | 2411.812012 | 0.967 |

### Effect of Value Head Norm (C vs A, holding z→z constant = OFF)

| Metric | A (vhead OFF) | C (vhead ON) | C/A Ratio |
|--------|---------------|--------------|-----------|
| V_var | 0.015561 | 2492.957031 | 160208.716 |

### Projection Saturation Check

How much does projection clip the norm? (pre vs post at n=16)

| Condition | pre ||z_H|| | post ||z_H|| | Saturation |
|-----------|------------|-------------|------------|
| A | 31.9999 | 10.0000 | YES |
| B | 31.9998 | 10.0000 | YES |
| C | 31.9999 | 10.0000 | YES |
| D | 31.9998 | 10.0000 | YES |

---

## Conclusions

**MIXED/INCONCLUSIVE RESULTS**

z→z contraction has inconsistent effects on latent variance:
- B/A variance ratio: 0.748
- D/C variance ratio: 0.748

The effect of contraction on latent diversity is unclear.

**PROJECTION SATURATION DETECTED**

Pre-projection norms exceed the ball radius (10.0), so projection is active.
This explains why post-projection ||z|| = 10.0 uniformly.

---

*Report generated by scripts/diagnostics/diagnose_latent_collapse_v2.py*
