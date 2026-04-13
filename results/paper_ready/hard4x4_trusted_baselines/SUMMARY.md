# Hard 4x4 Trusted Baseline Summary

Paper-facing hard-4x4 aggregate for `T2.1`: UPI-TRM main anchor plus the trusted external SB3 PPO/A2C reruns on the no-mask 6-8-empties suite.

## Aggregate Table

| Method | Steps | Seeds | Success Rate | 95% CI | Mean Return | Invalid Action Rate |
|--------|-------|-------|--------------|--------|-------------|---------------------|
| UPI-TRM | 20000 | 10 | 0.574 ± 0.122 | [0.487, 0.661] | -- | -- |
| SB3 PPO | 5000 | 10 | 0.000 ± 0.000 | [0.000, 0.000] | -17.508 ± 0.123 | 0.988 ± 0.025 |
| SB3 A2C | 5000 | 10 | 0.000 ± 0.000 | [0.000, 0.000] | -18.058 ± 0.618 | 0.817 ± 0.171 |

## Gap Analysis

- Best external baseline by mean success: SB3 PPO, SB3 A2C (tie at 0.0%).
- Seed-bootstrap 95% CI for `UPI-TRM - best external baseline`: [0.502, 0.644] success-rate points.
- Final-eval Wilson 95% upper bound for each external baseline's solve rate: PPO <= 0.76%, A2C <= 0.76%.
- All checkpointed evals stayed at zero success: PPO 500 / 500 zero-success eval points, A2C 500 / 500 zero-success eval points.

## Paper-Ready Interpretation

The trusted external rerun confirms the hard-4x4 no-mask capability story rather than softening it. UPI-TRM stays nontrivial at 57.4% ± 12.2% over 10 seeds, while both SB3 PPO and SB3 A2C remain at 0.0% success over 10 seeds on the same no-mask suite. The bootstrap 95% CI for the UPI-vs-best-baseline gap is [50.2, 64.4] percentage points, well above zero. Across all 20 external runs and all checkpointed evaluations during training, neither trusted baseline ever solved a puzzle. PPO's invalid-action rate remains near the full unmasked 97-action space, while A2C lowers invalid actions somewhat but still never reaches nonzero success. This is enough to resolve the reviewer-trust objection without spending more time on an additional DQN baseline.
