# Frozen replay comparison rules

Frozen before the first checkpoint replay on 2026-09-14.

1. Input files, checkpoint files, model-state dictionaries, census ordering, and replay binaries compare by exact byte SHA-256 identity. No tolerance applies.
2. Every exported floating-point value uses canonical Python binary64 `float.hex()` text under schema `float_hex_v1`. Integer and boolean tensor elements remain exact integers and booleans. Tensor dtype and shape are part of the record.
3. The consumer traverses seed position 0 through 7, census record order 0 through 127, fields in its declared order, and vector elements by ascending action index. That order defines the first divergent quantity.
4. The consumer independently recomputes each base-policy Bellman backup with Python 3.12 builtin `sum` over allowed actions in ascending action-index order. It then recomputes signed and absolute residuals, endpoint discrepancy, first-occurrence maxima, `R_hat_n`, `B_hat`, and `G` in binary64. These values and witness identities must match the corresponding sealed payload values bit for bit, including signed zero.
5. The eight-gap mean uses exact rational addition of the eight binary64 inputs and one final correctly rounded binary64 division. It must match the published result bit for bit.
6. The bootstrap uses 10,000 eight-draw replicates, namespace `upi-trm-exp1b-seed-bootstrap-v1`, seed `3246702300714487323`, SHA-256 counter coordinates, low-three-bit sampling, and Hyndman-Fan type 7 endpoints at 0.025 and 0.975. Replicate values, vector digest, plan digest, and interval endpoints must match the published result bit for bit.
7. For raw paired quantities, the report records absolute difference, relative difference `abs(a-b)/max(abs(a),abs(b))` with zero/zero defined as zero, and ordered-binary64 ULP distance. Exact identities produced by the same deterministic arithmetic require zero ULP difference.
8. Existing registered diagnostic thresholds remain fixed: probability mass `1e-10` absolute; mixture identity `1e-6`; constructed centering `1e-6`; trainer centering defect `1e-6`; trainer-versus-constructed parity `1e-6 + (4 * 2^-23) * 10 = 5.76837158203125e-6`. No threshold will be widened.
9. The model, frozen-base, candidate, and target state-dictionary identities must equal their sealed pre-replay values before evaluation and remain identical after evaluation.
10. Any authentication failure or unexplained mismatch stops the replay. The harness will not retry a failed seed to seek a favorable result. `REPLAY_DONE` is created only after complete agreement or after the first divergence has been isolated and written down.
