# CLAUDE.md

Repo-local Markdown mirrors for experiment planning and manuscript guidance live in this repository.

Canonical paper source tree:
`/home/buiksat/UPI_TRM/UPI_TRM_NIPS`

Canonical Markdown mirror for paper planning:
`/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS/NIPS_PLAN.md`

Use this repository for code, configs, scripts, generated artifacts, and Markdown planning/provenance mirrors.

Detailed orientation report for future Claude/Codex sessions:
- `/home/buiksat/trm_bellman/documents/CLAUDE_PROJECT_REPORT.md`

Current locked experiment status (2026-04-10):
- Main hard-4x4 capability anchor is the no-mask protocol, not the masked one.
- `M1` no-mask hard-4x4 is complete: UPI `0.574` mean (`std 0.122`) over seeds `0..9`; in-house A2C `0.000` over seeds `0..3`.
- Controlled no-mask hard-4x4 2x2 is complete: `nc_r0 = 0.350`, `nc_r10 = 0.482`, `c_r0 = 0.374`, `c_r10 = 0.502`.
- Interpretation: projection is the primary stabilizer on hard 4x4 (`+0.13` main effect); contraction adds only a small average lift (`+0.02`) and does not reduce variance under projection.
- Next empirical blocker: `PF2` / `PF3` in `/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS/PHASE0_EXPERIMENT_QUEUE.md` for the `exp1_v4` depth-mismatch package.
