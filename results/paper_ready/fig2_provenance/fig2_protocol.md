# Figure 2 Protocol Summary (Paper-Safe)

**Figure label:** `fig:baseline_curves`
**PDF file:** `figures/trivial_baselines_vs_no_contraction_success_vs_steps.pdf`

---

## Dataset / Suite Definition

- **Task:** 4x4 Sudoku
- **Difficulty:** Trivial (1-4 empty cells)
- **Dataset path:** `data/sudoku-4x4-trivial`
- **Grid size:** 4x4 (16 cells total)
- **Action space:** 97 discrete edit actions (16 cells x 6 tokens + 1 STOP, but STOP disabled)

---

## Training Protocol

| Parameter | Value | Source |
|-----------|-------|--------|
| Training steps | 5000 | Config `num_train_steps: 5000` |
| Batch size | 32 (UPI-TRM) / 256 (baselines) | Per-method config |
| Discount (gamma) | 0.99 | Config `gamma: 0.99` |
| K-step horizon | 1 (UPI-TRM) / 5 (PPO) | Config `K` |
| Episode horizon (max_edits) | 16 | Config `max_edits: 16` |
| Evaluation frequency | Every 100 steps | Config `eval_interval: 100` |
| Evaluation episodes | 50 per checkpoint | Config `eval_num_episodes: 50` |

---

## Reward Shaping

- **Checker:** Feasibility checker (`use_feasibility_checker: true`)
- **Score formula:** `filled - 2.0 * violations - 5.0 * zero_cand`
- **Max score:** 16.0 (all cells filled, no violations)
- **Terminal reward (success):** +1.0
- **Terminal reward (failure):** -16.0

---

## Success Metric Definition

An episode is **successful** if:
1. All 16 cells are filled (no zeros remain), AND
2. Zero constraint violations (no row/column/box duplicates)

Equivalently: `sudoku_is_solved(final_plan)` returns True.

**Evaluation mode:** Greedy (argmax) action selection with action masking.

---

## Evaluation Compute Protocol

| Parameter | Value | Notes |
|-----------|-------|-------|
| Unroll depth (n_eval) | 2 | Same as training `inner_unroll_n` |
| Projection radius (R) | **VARIES BY METHOD** | See table below |
| Greedy action selection | Yes | Deterministic evaluation |
| Action masking | Yes | Only valid edits allowed |

### Projection Radius by Method

| Method | Config | Projection R | Notes |
|--------|--------|--------------|-------|
| persistent_nc | `ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml` | 10.0 (default) | Projection ON |
| episodic_nc | `ablations/upi_trm_feasibility_no_contraction.yaml` | 10.0 (default) | Projection ON |
| episodic_c_clean | `exp3_projection_ablation/c_rdis.yaml` | 0 (disabled) | Projection OFF |
| ppo | `baselines/ppo_trm_feasibility.yaml` | 0 (disabled) | Projection OFF |
| a2c | `baselines/a2c_trm_feasibility.yaml` | 0 (disabled) | Projection OFF |
| dqn | `baselines/dqn_trm_feasibility.yaml` | 0 (disabled) | Projection OFF |

**IMPORTANT:** The no-contraction UPI-TRM variants use projection R=10, while the contraction variant and all baselines use R=0. This is an intentional protocol asymmetry matching historical Table 3 setup. For fully controlled comparisons with identical projection settings, use the `exp3_projection_ablation/` configs.

---

## Curve Aggregation

- **Seeds per method:** 3 (seeds 42, 123, 456)
- **Individual curves:** Thin, semi-transparent lines (alpha=0.18)
- **Mean curve:** Thick line with markers, labeled with seed count `(S=3)`
- **Legend format:** `{method_name} (S={num_seeds})`
- **Random baseline:** Horizontal dashed line at 52%

---

## Method-Specific Settings

### UPI-TRM Variants

| Setting | persistent_nc | episodic_nc | episodic_c_clean |
|---------|---------------|-------------|------------------|
| `episodic_latent` | false | true | true |
| `enable_contraction` | false | false | true |
| `target_Lz` | 0.9 (ignored) | 0.9 (ignored) | 0.9 |
| `latent_ball_radius` | 10.0 | 10.0 | 0.0 |
| `disable_value_head_norm` | not set | not set | true |

### Baselines (PPO/A2C/DQN)

All baselines use:
- `enable_contraction: false`
- `latent_ball_radius: 0.0`
- Algorithm-specific hyperparameters (PPO clip, A2C num_steps, DQN buffer, etc.)

---

## Consistency with Table 3

Figure 2 learning curves and Table 3 values are derived from **the same runs**:
- Both read from `results/table3_baselines/*.log`
- Table 3 reports final success at step 5000
- Figure 2 shows success vs. training steps

Final values from curves match Table 3 exactly (see `fig2_table3_consistency.md`).

---

## Known Protocol Asymmetries

1. **Projection radius:** No-contraction variants use R=10; contraction variant and baselines use R=0.
2. **Value-head normalization:** Only `episodic_c_clean` explicitly sets `disable_value_head_norm: true`.
3. **K-step horizon:** UPI-TRM uses K=1; PPO uses K=5 (algorithm difference).

These asymmetries are intentional for Table 3 (matching historical experimental setup). The paper notes these differences where relevant.
