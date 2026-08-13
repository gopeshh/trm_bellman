#!/usr/bin/env python3
"""
Phase 4: Figure Generation for 2×2 Norm Ablation.

Produces:
- fig_phase4_2x2_norm_ablation.pdf: 2×2 grid showing condition effects
- table_phase4_2x2_norm_ablation.tex: LaTeX table

Usage:
    buck2 run //buiksat_trm:make_paper_figures_phase4 -- \
        --summary_json results/paper_ready/phase4_2x2_norm_ablation/v3/summary.json
        --checkpoint_dir results/phase4_2x2_norm_ablation
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

from scripts.phase4_result_schema import (  # noqa: E402
    Phase4SummaryValidationError,
    validate_phase4_summary,
)
from scripts.phase4_checkpoint import (  # noqa: E402
    Phase4CheckpointError,
    verify_phase4_summary_checkpoints,
)
from scripts.phase4_diagnostic_inputs import (  # noqa: E402
    Phase4DiagnosticInputError,
    verify_phase4_diagnostic_inputs,
)
from scripts.phase4_source import (  # noqa: E402
    PHASE4_FIGURE_SOURCE_PROFILE,
    Phase4SourceError,
    phase4_evaluator_source_manifest_sha256,
    require_phase4_runtime_attestation,
    resolve_phase4_source_roots,
    verify_phase4_producer_source,
    verify_phase4_runtime_sources,
)
from utils.run_identity import (  # noqa: E402
    RunIdentityError,
    discover_clean_git_source,
)


def load_summary(
    summary_path: str,
    checkpoint_dir: str,
    data_dir: str,
    project_root: str | Path,
    producer_project_root: str | Path,
    expected_producer_source: Dict[str, Any],
    expected_evaluator_runtime_sha256: str,
) -> Dict[str, Any]:
    """Load a summary and revalidate every checkpoint before publication."""
    with open(summary_path, "r") as f:
        summary = json.load(f)
    validate_phase4_summary(summary)
    source_identity = discover_clean_git_source(project_root)
    if summary["evaluator_git_commit"] != source_identity["git_commit"]:
        raise Phase4SourceError(
            "Summary evaluator commit differs from the clean project checkout."
        )
    expected_source_digest = phase4_evaluator_source_manifest_sha256(
        project_root
    )
    if summary["evaluator_source_manifest_sha256"] != expected_source_digest:
        raise Phase4SourceError(
            "Summary evaluator source digest differs from the project checkout."
        )
    if (
        summary["evaluator_runtime_artifact_sha256"]
        != expected_evaluator_runtime_sha256
    ):
        raise Phase4SourceError(
            "Summary evaluator runtime differs from the authorized PAR digest."
        )
    verify_phase4_summary_checkpoints(
        summary,
        checkpoint_dir,
        (
            Path(producer_project_root)
            / "configs"
            / "phase4_2x2_norm_ablation"
        ),
        expected_producer_source=expected_producer_source,
        device="cpu",
    )
    verify_phase4_diagnostic_inputs(summary, data_dir)
    return summary


def generate_2x2_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate 2×2 grid plot showing condition effects."""
    validate_phase4_summary(summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plot")
        return

    aggregates = summary["aggregates"]

    # Create 2×2 matrix of results
    # Rows: z→z contraction (OFF, ON)
    # Cols: value-head norm (OFF, ON)
    matrix_labels = [
        ["nc_nv", "nc_yv"],  # Contraction OFF
        ["yc_nv", "yc_yv"],  # Contraction ON
    ]

    # Get data by condition
    data_by_cond = {a["condition"]: a for a in aggregates}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Phase 4: 2×2 Norm Ablation", fontsize=14, fontweight='bold')

    metrics = [
        ("var_V_mean", "Var(V)", "steelblue"),
        ("argmax_n4_mean", "Argmax@n=4", "darkgreen"),
    ]

    for i, contraction in enumerate(["OFF", "ON"]):
        for j, vhead in enumerate(["OFF", "ON"]):
            ax = axes[i, j]
            cond = matrix_labels[i][j]

            if cond in data_by_cond:
                data = data_by_cond[cond]

                # Bar chart of metrics
                x = np.arange(len(metrics))
                values = [data[m[0]] for m in metrics]
                stds = [data[m[0].replace("_mean", "_std")] for m in metrics]
                colors = [m[2] for m in metrics]

                ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
                ax.errorbar(x, values, yerr=stds, fmt='none', color='black', capsize=3)

                ax.set_xticks(x)
                ax.set_xticklabels([m[1] for m in metrics], rotation=45, ha='right')
                ax.set_ylim(0, max(1.0, max(values) * 1.2))

            ax.set_title(f"C={contraction}, V={vhead}", fontsize=11)
            ax.grid(True, alpha=0.3)

    # Add row/column labels
    axes[0, 0].set_ylabel("z→z Contraction OFF", fontsize=10)
    axes[1, 0].set_ylabel("z→z Contraction ON", fontsize=10)

    plt.tight_layout()

    pdf_path = out_path / "fig_phase4_2x2_norm_ablation.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    png_path = out_path / "fig_phase4_2x2_norm_ablation.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {png_path}")

    plt.close()


def generate_bar_comparison(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate grouped bar chart comparing conditions."""
    validate_phase4_summary(summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    aggregates = summary["aggregates"]

    conditions = [a["condition"] for a in aggregates]
    labels = [a["label"] for a in aggregates]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Plot 1: Var(V)
    ax = axes[0]
    vals = [a["var_V_mean"] for a in aggregates]
    stds = [a["var_V_std"] for a in aggregates]
    x = np.arange(len(conditions))
    ax.bar(x, vals, yerr=stds, color='steelblue', alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Var(V)")
    ax.set_title("Value Variance")
    ax.grid(True, alpha=0.3)

    # Plot 2: Argmax Agreement
    ax = axes[1]
    vals = [a["argmax_n4_mean"] for a in aggregates]
    stds = [a["argmax_n4_std"] for a in aggregates]
    ax.bar(x, vals, yerr=stds, color='darkgreen', alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Argmax Agreement at n=4")
    ax.set_title("Policy Stability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Phase 4: Norm Ablation Comparison", fontsize=12, fontweight='bold')
    plt.tight_layout()

    pdf_path = out_path / "fig_phase4_bar_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    plt.close()


def generate_latex_table(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate LaTeX table."""
    validate_phase4_summary(summary)
    aggregates = summary["aggregates"]

    content = r"""\begin{table}[t]
\centering
\caption{Phase 4: 2×2 Norm Ablation across 3 seeds. C=z$\to$z contraction, V=value-head spectral norm.}
\label{tab:phase4-2x2-ablation}
\begin{tabular}{lcccc}
\toprule
Condition & C & V & Var($V$) & Argmax at $n=4$ \\
\midrule
"""

    for a in aggregates:
        c_status = "ON" if a["enable_contraction"] else "OFF"
        v_status = "OFF" if a["disable_value_head_norm"] else "ON"
        content += f"{a['label']} & {c_status} & {v_status} & "
        content += f"${a['var_V_mean']:.3f} \\pm {a['var_V_std']:.3f}$ & "
        content += f"${a['argmax_n4_mean']:.3f} \\pm {a['argmax_n4_std']:.3f}$ \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex_path = out_path / "table_phase4_2x2_norm_ablation.tex"
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"Saved: {tex_path}")


def main(*, runtime_attestation: Dict[str, Any] | None = None):
    try:
        attestation = require_phase4_runtime_attestation(
            runtime_attestation,
            PHASE4_FIGURE_SOURCE_PROFILE,
        )
    except Phase4SourceError as error:
        print(f"ERROR: Figure runtime source is not authenticated: {error}")
        return 1
    parser = argparse.ArgumentParser(description="Generate Phase 4 figures")
    parser.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="Exact clean implementation checkout used by the evaluator",
    )
    parser.add_argument(
        "--fbcode_root",
        type=str,
        required=True,
        help="fbcode root whose buiksat_trm cell resolves to project_root",
    )
    parser.add_argument(
        "--producer_project_root",
        type=str,
        required=True,
        help="Exact clean implementation checkout used for Phase 4 training",
    )
    parser.add_argument(
        "--expected_producer_git_commit",
        type=str,
        required=True,
        help="Authorized Phase 4 training commit",
    )
    parser.add_argument(
        "--expected_evaluator_runtime_sha256",
        required=True,
        help="Externally authorized evaluator PAR SHA-256",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path to summary.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (defaults to same as summary.json)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Trusted root containing the 12 full Phase 4 checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Trusted root containing the recorded Phase 4 diagnostic arrays",
    )

    args = parser.parse_args()
    if re.fullmatch(
        r"[0-9a-f]{64}",
        args.expected_evaluator_runtime_sha256,
    ) is None:
        print("ERROR: Expected evaluator runtime SHA-256 is invalid.")
        return 1

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        return 1

    try:
        project_root, _ = resolve_phase4_source_roots(
            args.project_root,
            args.fbcode_root,
        )
        runtime_source_digest = verify_phase4_runtime_sources(
            project_root,
            PHASE4_FIGURE_SOURCE_PROFILE,
        )
        project_identity = discover_clean_git_source(project_root)
        if (
            runtime_source_digest != attestation["source_manifest_sha256"]
            or project_identity["git_commit"]
            != attestation["source_git_commit"]
        ):
            raise Phase4SourceError(
                "Figure checkout differs from the pre-import runtime attestation."
            )
        producer_project_root = Path(
            args.producer_project_root
        ).expanduser().resolve(strict=True)
        expected_producer_source = verify_phase4_producer_source(
            producer_project_root,
            args.expected_producer_git_commit,
        )
        summary = load_summary(
            str(summary_path),
            args.checkpoint_dir,
            args.data_dir,
            project_root,
            producer_project_root,
            expected_producer_source,
            args.expected_evaluator_runtime_sha256,
        )
    except (
        OSError,
        json.JSONDecodeError,
        Phase4CheckpointError,
        Phase4DiagnosticInputError,
        Phase4SourceError,
        Phase4SummaryValidationError,
        RunIdentityError,
    ) as error:
        print(f"ERROR: Summary is not publishable: {error}")
        return 1

    out_path = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures from: {summary_path}")
    print(f"Output directory: {out_path}")
    print()

    generate_2x2_plot(summary, out_path)
    generate_bar_comparison(summary, out_path)
    generate_latex_table(summary, out_path)

    try:
        if verify_phase4_runtime_sources(
            project_root,
            PHASE4_FIGURE_SOURCE_PROFILE,
        ) != runtime_source_digest:
            raise Phase4SourceError(
                "Figure-generator runtime source changed during generation."
            )
        if verify_phase4_producer_source(
            producer_project_root,
            args.expected_producer_git_commit,
        ) != expected_producer_source:
            raise Phase4SourceError(
                "Producer source identity changed during figure generation."
            )
        final_identity = discover_clean_git_source(project_root)
        if final_identity["git_commit"] != summary["evaluator_git_commit"]:
            raise Phase4SourceError(
                "Project source identity changed during figure generation."
            )
    except (OSError, Phase4SourceError, RunIdentityError) as error:
        print(f"ERROR: Figure generation source changed: {error}")
        return 1

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
