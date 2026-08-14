#!/usr/bin/env python3
"""
Phase 4: Figure Generation for 2×2 Norm Ablation.

Produces:
- fig_phase4_2x2_norm_ablation.pdf: 2×2 grid showing condition effects
- table_phase4_2x2_norm_ablation.tex: LaTeX table

Build the figure PAR, freeze its SHA-256 externally, and invoke it only through
``phase4_runtime_launcher --purpose phase4-figure``. Direct PAR or ``buck2 run``
execution fails the required pre-import runtime attestation.
"""

import argparse
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
from scripts.phase4_figure_publication import (  # noqa: E402
    Phase4FigurePublicationError,
    Phase4FigurePublicationIdentity,
    build_phase4_figure_publication_identity,
    publish_phase4_generation,
    stable_regular_file_bytes,
    strict_json_loads,
    verify_runtime_archive_sha256,
)
from scripts.phase4_source import (  # noqa: E402
    PHASE4_FIGURE_SOURCE_PROFILE,
    Phase4SourceError,
    phase4_evaluator_source_manifest_sha256,
    require_phase4_runtime_attestation,
    resolve_phase4_path,
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
    expected_training_runtime_sha256: str,
) -> tuple[Dict[str, Any], str]:
    """Load a summary and revalidate every checkpoint before publication."""
    summary_bytes, summary_sha256 = stable_regular_file_bytes(
        summary_path,
        "Phase 4 summary",
    )
    summary = strict_json_loads(summary_bytes, "Phase 4 summary")
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
        expected_training_runtime_sha256=expected_training_runtime_sha256,
        device="cpu",
    )
    verify_phase4_diagnostic_inputs(summary, data_dir)
    return summary, summary_sha256


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
        "--expected_training_runtime_sha256",
        required=True,
        help="Externally authorized Phase 4 training PAR SHA-256",
    )
    parser.add_argument(
        "--evaluator_runtime_archive",
        required=True,
        help="Absolute path to the authorized evaluator PAR",
    )
    parser.add_argument(
        "--training_runtime_archive",
        required=True,
        help="Absolute path to the authorized Phase 4 training PAR",
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
        required=True,
        help="Dedicated publication root within project_root",
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
    try:
        project_root, _ = resolve_phase4_source_roots(
            args.project_root,
            args.fbcode_root,
        )
        evaluator_runtime_sha256 = verify_runtime_archive_sha256(
            args.evaluator_runtime_archive,
            args.expected_evaluator_runtime_sha256,
            "evaluator runtime archive",
        )
        training_runtime_sha256 = verify_runtime_archive_sha256(
            args.training_runtime_archive,
            args.expected_training_runtime_sha256,
            "training runtime archive",
        )
        summary_path = resolve_phase4_path(args.summary_json, project_root)
        if not summary_path.exists():
            raise Phase4SourceError(
                f"Summary file not found: {summary_path}"
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
        checkpoint_dir = resolve_phase4_path(
            args.checkpoint_dir,
            producer_project_root,
        )
        data_dir = resolve_phase4_path(args.data_dir, project_root)
        summary, summary_sha256 = load_summary(
            str(summary_path),
            str(checkpoint_dir),
            str(data_dir),
            project_root,
            producer_project_root,
            expected_producer_source,
            evaluator_runtime_sha256,
            training_runtime_sha256,
        )
        publication_identity = build_phase4_figure_publication_identity(
            summary,
            summary_sha256=summary_sha256,
            evaluator_runtime_artifact_sha256=evaluator_runtime_sha256,
            training_runtime_artifact_sha256=training_runtime_sha256,
            figure_git_commit=project_identity["git_commit"],
            figure_source_manifest_sha256=runtime_source_digest,
            figure_runtime_artifact_sha256=attestation["runtime_sha256"],
            expected_producer_source=expected_producer_source,
        )
    except (
        OSError,
        Phase4CheckpointError,
        Phase4DiagnosticInputError,
        Phase4FigurePublicationError,
        Phase4SourceError,
        Phase4SummaryValidationError,
        RunIdentityError,
    ) as error:
        print(f"ERROR: Summary is not publishable: {error}")
        return 1

    print(f"Generating figures from: {summary_path}")
    print(f"Publication root: {args.out_dir}")
    print()

    def generate(staging_path: Path) -> None:
        generate_2x2_plot(summary, staging_path)
        generate_bar_comparison(summary, staging_path)
        generate_latex_table(summary, staging_path)

    def revalidate() -> Phase4FigurePublicationIdentity:
        final_evaluator_runtime_sha256 = verify_runtime_archive_sha256(
            args.evaluator_runtime_archive,
            args.expected_evaluator_runtime_sha256,
            "evaluator runtime archive",
        )
        final_training_runtime_sha256 = verify_runtime_archive_sha256(
            args.training_runtime_archive,
            args.expected_training_runtime_sha256,
            "training runtime archive",
        )
        final_runtime_source_digest = verify_phase4_runtime_sources(
            project_root,
            PHASE4_FIGURE_SOURCE_PROFILE,
        )
        if final_runtime_source_digest != runtime_source_digest:
            raise Phase4SourceError(
                "Figure-generator runtime source changed during generation."
            )
        final_producer_source = verify_phase4_producer_source(
            producer_project_root,
            args.expected_producer_git_commit,
        )
        if final_producer_source != expected_producer_source:
            raise Phase4SourceError(
                "Producer source identity changed during figure generation."
            )
        final_summary, final_summary_sha256 = load_summary(
            str(summary_path),
            str(checkpoint_dir),
            str(data_dir),
            project_root,
            producer_project_root,
            expected_producer_source,
            final_evaluator_runtime_sha256,
            final_training_runtime_sha256,
        )
        final_project_identity = discover_clean_git_source(project_root)
        if final_project_identity != project_identity:
            raise Phase4SourceError(
                "Project source identity changed during figure generation."
            )
        return build_phase4_figure_publication_identity(
            final_summary,
            summary_sha256=final_summary_sha256,
            evaluator_runtime_artifact_sha256=(
                final_evaluator_runtime_sha256
            ),
            training_runtime_artifact_sha256=(
                final_training_runtime_sha256
            ),
            figure_git_commit=final_project_identity["git_commit"],
            figure_source_manifest_sha256=final_runtime_source_digest,
            figure_runtime_artifact_sha256=attestation["runtime_sha256"],
            expected_producer_source=final_producer_source,
        )

    try:
        published = publish_phase4_generation(
            project_root,
            args.out_dir,
            publication_identity,
            generate,
            revalidate,
        )
    except (
        OSError,
        Phase4CheckpointError,
        Phase4DiagnosticInputError,
        Phase4FigurePublicationError,
        Phase4SourceError,
        Phase4SummaryValidationError,
        RunIdentityError,
    ) as error:
        print(f"ERROR: Figure generation was not published: {error}")
        return 1

    print(f"Published generation: {published.generation_path}")
    print(f"Current pointer: {published.current_path}")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
