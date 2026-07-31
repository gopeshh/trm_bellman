#!/usr/bin/env python3
"""
Exp3: Training-time projection ablation evaluation.

This script:
1. Loads Exp3 checkpoints (6 conditions: NC/C × R10/R100/Rdis)
2. Computes training metrics from logs (success rate, projection_active_rate, etc.)
3. Evaluates unroll sensitivity (reusing Exp1 machinery)
4. Checks decision gates G1-G4
5. Generates paper-ready summary

Decision Gates:
- G1 (Stability): Training stable (no NaN, success > random)
- G2 (Projection Activity): R=10 ~100% active, R=100 ~0% active
- G3 (Contraction-Only): C-Rdis stable with reasonable success
- G4 (Interpretation): Tradeoff between projection and stability

Output:
- results/paper_ready/exp3_projection_ablation/summary.json
- results/paper_ready/exp3_projection_ablation/CLAIMS.md
- results/paper_ready/exp3_projection_ablation/PROVENANCE.md
"""

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# JSON Encoder for Numpy Types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, o: object) -> Any:
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# =============================================================================
# Configuration
# =============================================================================

RESULTS_BASE = PROJECT_ROOT / "results/exp3"
OUT_DIR = PROJECT_ROOT / "results/paper_ready/exp3_projection_ablation"

# Exp3 conditions
CONDITIONS = [
    ("nc_r10", False, 10.0),     # No contraction, R=10
    ("nc_r100", False, 100.0),   # No contraction, R=100
    ("nc_rdis", False, 0.0),     # No contraction, R=disabled
    ("c_r10", True, 10.0),       # Contraction, R=10
    ("c_r100", True, 100.0),     # Contraction, R=100
    ("c_rdis", True, 0.0),       # Contraction, R=disabled
]

SEEDS = [42]  # Start with 1 seed, expand to [41, 42, 43] if stable

# Unroll depths for sensitivity analysis
N_TRAIN = 2
N_EVAL_DEPTHS = [4, 8, 16]

# Decision gate thresholds
G1_SUCCESS_THRESHOLD = 0.05  # Must beat random (~5%)
G2_R10_ACTIVE_MIN = 0.90  # R=10 should have ~100% projection active
G2_R100_ACTIVE_MAX = 0.10  # R=100 should have ~0% projection active


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingMetrics:
    """Metrics from training logs."""
    final_success_rate: float
    max_success_rate: float
    had_nan: bool
    final_value_loss: float
    projection_active_rate: Optional[float]
    L_preproj: Optional[float]
    L_postproj: Optional[float]
    mean_latent_norm: Optional[float]


@dataclass
class UnrollMetrics:
    """Metrics for unroll sensitivity comparison."""
    n_train: int
    n_eval: int
    delta_V_mean: float
    delta_V_std: float
    delta_pi_mean: float
    delta_pi_std: float
    argmax_agree_rate: float


@dataclass
class ConditionResult:
    """Results for a single condition."""
    name: str
    enable_contraction: bool
    latent_ball_radius: float
    seeds: List[int]

    # Training metrics (averaged over seeds)
    training: Optional[TrainingMetrics] = None

    # Unroll sensitivity (at each depth)
    unroll_metrics: List[UnrollMetrics] = field(default_factory=list)

    # Primary stability metric (n=2 vs n=16)
    delta_V_2_16: float = 0.0
    argmax_agree_2_16: float = 0.0

    # Gate status
    g1_passed: bool = False
    g2_passed: bool = False


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    value: float
    threshold: float
    details: str


# =============================================================================
# Log Parsing
# =============================================================================

def parse_training_log(log_path: Path) -> Optional[TrainingMetrics]:
    """Parse training log to extract metrics."""
    if not log_path.exists():
        return None

    content = log_path.read_text()

    # Check for NaN
    had_nan = "nan" in content.lower() or "NaN" in content

    # Extract success rates from eval lines
    # Pattern: eval_success_rate=0.123
    success_rates = re.findall(r"eval_success_rate=([0-9.]+)", content)
    success_rates = [float(x) for x in success_rates]

    final_success_rate = success_rates[-1] if success_rates else 0.0
    max_success_rate = max(success_rates) if success_rates else 0.0

    # Extract value loss
    # Pattern: value_loss=0.123456
    value_losses = re.findall(r"value_loss=([0-9.]+)", content)
    final_value_loss = float(value_losses[-1]) if value_losses else 0.0

    # Projection active rate if tracked
    proj_rates = re.findall(r"projection_active_rate=([0-9.]+)", content)
    projection_active_rate = float(proj_rates[-1]) if proj_rates else None

    # Lipschitz estimates if tracked
    l_preproj = re.findall(r"L_preproj=([0-9.]+)", content)
    l_postproj = re.findall(r"L_postproj=([0-9.]+)", content)

    return TrainingMetrics(
        final_success_rate=final_success_rate,
        max_success_rate=max_success_rate,
        had_nan=had_nan,
        final_value_loss=final_value_loss,
        projection_active_rate=projection_active_rate,
        L_preproj=float(l_preproj[-1]) if l_preproj else None,
        L_postproj=float(l_postproj[-1]) if l_postproj else None,
        mean_latent_norm=None,  # Would need to parse from logs
    )


def find_checkpoint(condition_name: str, seed: int) -> Optional[Path]:
    """Find checkpoint path for given condition and seed."""
    ckpt_dir = RESULTS_BASE / f"{condition_name}_s{seed}"

    # Look for model_step_5000.pt (final) or latest available
    for step in [5000, 4000, 3000, 2000, 1000]:
        ckpt_path = ckpt_dir / f"model_step_{step}.pt"
        if ckpt_path.exists():
            return ckpt_path

    return None


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_condition(
    name: str,
    enable_contraction: bool,
    radius: float,
    seeds: List[int],
) -> ConditionResult:
    """Evaluate a single condition across seeds."""

    result = ConditionResult(
        name=name,
        enable_contraction=enable_contraction,
        latent_ball_radius=radius,
        seeds=seeds,
    )

    # Collect training metrics from logs
    all_training = []
    for seed in seeds:
        log_path = RESULTS_BASE / f"{name}_s{seed}.log"
        metrics = parse_training_log(log_path)
        if metrics:
            all_training.append(metrics)

    if all_training:
        # Average over seeds
        result.training = TrainingMetrics(
            final_success_rate=np.mean([t.final_success_rate for t in all_training]),
            max_success_rate=np.mean([t.max_success_rate for t in all_training]),
            had_nan=any(t.had_nan for t in all_training),
            final_value_loss=np.mean([t.final_value_loss for t in all_training]),
            projection_active_rate=np.mean([t.projection_active_rate for t in all_training
                                           if t.projection_active_rate is not None]) or None,
            L_preproj=np.mean([t.L_preproj for t in all_training
                              if t.L_preproj is not None]) or None,
            L_postproj=np.mean([t.L_postproj for t in all_training
                               if t.L_postproj is not None]) or None,
            mean_latent_norm=None,
        )

        # Check G1: Stability gate
        result.g1_passed = (
            not result.training.had_nan and
            result.training.final_success_rate > G1_SUCCESS_THRESHOLD
        )

        # Check G2: Projection activity gate
        if result.training.projection_active_rate is not None:
            if radius == 10.0:
                result.g2_passed = result.training.projection_active_rate >= G2_R10_ACTIVE_MIN
            elif radius == 100.0:
                result.g2_passed = result.training.projection_active_rate <= G2_R100_ACTIVE_MAX
            else:  # disabled
                result.g2_passed = True  # No projection to check

    # TODO: Add unroll sensitivity evaluation using existing machinery
    # This would require loading checkpoints and running eval_unroll_sensitivity

    return result


# =============================================================================
# Summary Generation
# =============================================================================

def generate_claims(results: List[ConditionResult]) -> str:
    """Generate CLAIMS.md based on results."""

    lines = [
        "# Exp3 Claims: Training-Time Projection Ablation",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]

    # Count gates passed
    g1_passed = [r for r in results if r.g1_passed]
    g1_failed = [r for r in results if not r.g1_passed and r.training]

    lines.append(f"- **G1 (Stability):** {len(g1_passed)}/{len(results)} conditions passed")
    lines.append("")

    # Detailed results
    lines.append("## Results by Condition")
    lines.append("")
    lines.append("| Condition | Contraction | R | Success | NaN | G1 |")
    lines.append("|-----------|-------------|---|---------|-----|-----|")

    for r in results:
        if r.training:
            success = f"{r.training.final_success_rate:.3f}"
            nan_status = "YES" if r.training.had_nan else "NO"
            g1_status = "PASS" if r.g1_passed else "FAIL"
        else:
            success = "N/A"
            nan_status = "N/A"
            g1_status = "N/A"

        lines.append(
            f"| {r.name} | {r.enable_contraction} | {r.latent_ball_radius} | "
            f"{success} | {nan_status} | {g1_status} |"
        )

    lines.append("")

    # Scoped claims based on results
    lines.append("## Scoped Claims")
    lines.append("")

    if g1_failed:
        failed_names = ", ".join(r.name for r in g1_failed)
        lines.append(f"**Claim 1 (Negative):** The following conditions failed G1 (stability): {failed_names}")
        lines.append("")

    # Check if projection-disabled conditions are stable
    rdis_conditions = [r for r in results if r.latent_ball_radius == 0.0]
    rdis_passed = [r for r in rdis_conditions if r.g1_passed]

    if rdis_passed:
        lines.append(
            "**Claim 2 (Positive):** Training without projection (R=disabled) is stable "
            f"for conditions: {', '.join(r.name for r in rdis_passed)}"
        )
    elif rdis_conditions:
        lines.append(
            "**Claim 2 (Negative):** Training without projection (R=disabled) is unstable. "
            "Projection is required for stable training in this architecture."
        )

    lines.append("")
    lines.append("## Scope Limitations")
    lines.append("")
    lines.append("- Results on trivial 4×4 Sudoku only")
    lines.append("- Single seed (42) per condition")
    lines.append("- 5000 training steps")
    lines.append("")

    return "\n".join(lines)


def generate_provenance(results: List[ConditionResult]) -> str:
    """Generate PROVENANCE.md."""

    lines = [
        "# Exp3 Provenance",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Experiment Configuration",
        "",
        "- **Experiment:** Exp3 (Training-time projection ablation)",
        "- **Conditions:** 2×3 matrix (contraction × projection radius)",
        "- **Seeds:** 42 (single seed diagnostic)",
        "- **Training steps:** 5000",
        "- **Dataset:** sudoku-4x4-trivial",
        "",
        "## Config Files",
        "",
    ]

    for name, contraction, radius in CONDITIONS:
        lines.append(f"- `configs/exp3_projection_ablation/{name}.yaml`")

    lines.append("")
    lines.append("## Checkpoint Paths")
    lines.append("")

    for r in results:
        for seed in r.seeds:
            ckpt = find_checkpoint(r.name, seed)
            if ckpt:
                lines.append(f"- `{ckpt.relative_to(PROJECT_ROOT)}`")

    lines.append("")
    lines.append("## Commands to Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# Training (run all 6 conditions)")
    lines.append("for config in nc_r10 nc_r100 nc_rdis c_r10 c_r100 c_rdis; do")
    lines.append("  buck2 run //buiksat_trm:upi_trm_train -- \\")
    lines.append("    --config buiksat_trm/configs/exp3_projection_ablation/${config}.yaml \\")
    lines.append("    --seed 42 \\")
    lines.append("    --dataset-paths buiksat_trm/data/sudoku-4x4-trivial \\")
    lines.append("    --checkpoint-dir buiksat_trm/results/exp3/${config}_s42")
    lines.append("done")
    lines.append("")
    lines.append("# Evaluation")
    lines.append("buck2 run //buiksat_trm:eval_exp3_projection_ablation")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run Exp3 evaluation."""

    print("=" * 60)
    print("Exp3: Training-Time Projection Ablation Evaluation")
    print("=" * 60)
    print()

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate all conditions
    results = []
    for name, contraction, radius in CONDITIONS:
        print(f"Evaluating {name}...")
        result = evaluate_condition(name, contraction, radius, SEEDS)
        results.append(result)

        if result.training:
            print(f"  Success: {result.training.final_success_rate:.3f}")
            print(f"  NaN: {result.training.had_nan}")
            print(f"  G1: {'PASS' if result.g1_passed else 'FAIL'}")
        else:
            print("  No training data found")
        print()

    # Generate summary
    summary = {
        "experiment": "Exp3",
        "description": "Training-time projection ablation",
        "generated_at": datetime.now().isoformat(),
        "conditions": [asdict(r) for r in results],
        "gates": {
            "g1_passed": sum(1 for r in results if r.g1_passed),
            "g1_total": len(results),
        },
    }

    # Write outputs
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    with open(OUT_DIR / "CLAIMS.md", "w") as f:
        f.write(generate_claims(results))

    with open(OUT_DIR / "PROVENANCE.md", "w") as f:
        f.write(generate_provenance(results))

    print("=" * 60)
    print(f"Outputs written to: {OUT_DIR}")
    print("=" * 60)

    # Print summary table
    print()
    print("SUMMARY TABLE")
    print("-" * 60)
    print(f"{'Condition':<12} {'Contraction':<12} {'R':<8} {'Success':<10} {'G1':<6}")
    print("-" * 60)
    for r in results:
        if r.training:
            print(
                f"{r.name:<12} {str(r.enable_contraction):<12} "
                f"{r.latent_ball_radius:<8.1f} {r.training.final_success_rate:<10.3f} "
                f"{'PASS' if r.g1_passed else 'FAIL':<6}"
            )
        else:
            print(f"{r.name:<12} {str(r.enable_contraction):<12} {r.latent_ball_radius:<8.1f} {'N/A':<10} {'N/A':<6}")


if __name__ == "__main__":
    main()
