# TRM Artifact Reproducibility Notes

This artifact bundles the code, configs, and scripts needed to reproduce the ICML submission results for **UPI–TRM: Unrolled Policy Iteration in Plan Space for Tiny Recursive Models**. The focus is on fast verification, deterministic reruns, and minimal dependencies.

For the latest experiment summaries and pointers to current plot/CSV artifacts, see:
- `EXPERIMENT_RESULTS_4x4_FEASIBILITY.md`

## Environment Setup
1. Use Python 3.10+ on Linux or macOS.
2. (Recommended) Create a clean virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install pinned dependencies:
   ```bash
   pip install -r artifact/requirements-artifact.txt
   ```
4. If you need GPU acceleration, install the CUDA-appropriate PyTorch wheel before the remaining requirements.

All experiments rely on reproducibility hooks in `utils/seeding.py`, and `upi_trm_train.py` calls `set_global_seed()` automatically whenever `--seed` is provided.

## Running Tests
```
./artifact/run_all_tests.sh
```
This executes the curated smoke/unit suite (~8 minutes on a 16-core CPU workstation). Failures usually indicate missing dependencies or environment drift.

## Running Ablations
```
./artifact/run_all_ablations.sh
```
This launches the ICML ablation sweep (`K=1`, `K=3`, unroll=2). Expected runtimes:
- CPU-only: 2–3 hours total
- 1×A100 / L40 / A10 GPU: ~60–90 minutes total

Each run consumes the seeded dummy dataset for determinism.

## Expected Runtime Summary
- Environment setup: <5 minutes
- Test suite: ~8 minutes (CPU)
- Single ablation: 15–35 minutes (GPU)
- Full ablation sweep: ~1–1.5 hours (GPU)

## Determinism & Seeds
- Default seed is `42`. Override via `python upi_trm_train.py --seed 0`.
- Replay and action sampling consume the PyTorch RNG; CPU/GPU behavior matches upstream PyTorch determinism settings.
- For improved determinism, also set:
  ```bash
  export PYTHONHASHSEED=0
  ```

## Citation
```
@inproceedings{upi_trm_2025,
  title     = {Unrolled Policy Iteration in Plan Space for Tiny Recursive Models},
  author    = {Anonymous Authors},
  booktitle = {ICML 2025 Submission},
  year      = {2025}
}
```

For questions, contact the authors through the ICML submission portal.

---

## Appendix: Contraction fix (operator-norm clamping)

This section is merged from the former `docs/contraction_fix.md` so the artifact has a single place that covers both **reproducibility** and the **critical implementation detail** behind contraction enforcement.

### Problem: `spectral_norm` causes \(L_z\) explosion

The original implementation used PyTorch’s `torch.nn.utils.spectral_norm` to enforce per-layer Lipschitz bounds. Diagnostics showed it can be **numerically unstable** in this TRM architecture:

| Variant | Measured \(L_z\) | Status |
|---------|------------------:|--------|
| OFF (no contraction) | ~1.0 | OK |
| SN-only (`spectral_norm`) | 7000 – 600000+ | **EXPLODED** |
| SN+SCALE (`spectral_norm` + scaling) | 4000 – 65000+ | **EXPLODED** |

### Root cause (high level)

`spectral_norm` reparameterizes weights and updates them via hooks + power iteration. With inner-loop recursion (`L_cycles > 1`) and residual/normalization structure, this interacts poorly and produces pathological effective operator norms.

### Solution: operator-norm clamping

We replaced `spectral_norm` with a more direct, stable approach:

- Estimate spectral norm via power iteration in float32
- If `||W|| > max_norm`, rescale weights in-place: \(W \leftarrow W \cdot (max\_norm / ||W||)\)
- Apply selectively to the **z→z path** layers (reasoning stack), not embeddings/heads

### Results after fix (validated Jan 2026)

**With `target_Lz = 0.9` (eps=1e-3):**

| Variant | Measured \(L_z\) | max σ(W) | Status |
|---------|------------------:|---------:|--------|
| OFF (no contraction) | 1.39 | 3.81 | OK |
| CLAMP-only | **0.70** | 1.06 | **Contractive** |
| SCALE-only | 1.27 | 3.82 | OK |
| CLAMP+SCALE | **0.70** | 1.03 | **Contractive** |
| SN-only | 50257 | 3.65 | **EXPLODED** |
| SN+SCALE | 54055 | 3.81 | **EXPLODED** |

### Where to look in code

- `utils/lipschitz.py`: power-iteration + clamping utilities
- `models/recursive_reasoning/trm.py`: contraction enforcement uses opnorm clamp (not `spectral_norm`)
- `rl/upi_trm_trainer.py`: optional periodic re-clamping during training

### Diagnostic commands (optional)

- `scripts/diagnose_contraction_components.py` (old SN behavior)
- `scripts/diagnose_contraction_fix.py` (validates clamp vs SN)
