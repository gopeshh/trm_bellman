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
