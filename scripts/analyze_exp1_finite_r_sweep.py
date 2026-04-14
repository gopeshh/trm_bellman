#!/usr/bin/env python3
"""
Analyze finite-R eval-only sweeps on exp1_v4_refreeze using fixed (n1, n2) pairs.

This script intentionally reads per-state CSVs directly rather than the legacy
radius aggregate, because the legacy aggregate pools across all depth pairs.
"""

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = ("model_a", "model_b")
PRIMARY_METRICS = ("delta_V", "delta_z", "argmax_agree")
SUPPORTING_METRICS = ("delta_pi",)
MECHANISM_METRICS = ("saturated", "z_pre_norm", "z_post_norm", "proj_disp_proxy")
ALL_METRICS = PRIMARY_METRICS + SUPPORTING_METRICS + MECHANISM_METRICS

LABELS = {
    "model_a": "No Contraction",
    "model_b": "Contraction",
}

COLORS = {
    "model_a": "#E24A33",
    "model_b": "#348ABD",
}

METRIC_LABELS = {
    "delta_V": "Delta_V",
    "delta_z": "Delta_z",
    "argmax_agree": "Argmax Agree",
    "delta_pi": "Delta_pi",
    "saturated": "Saturated Rate",
    "z_pre_norm": "z_pre_norm",
    "z_post_norm": "z_post_norm",
    "proj_disp_proxy": "proj_disp_proxy",
}

REQUIRED_COLUMNS = (
    "state_id",
    "n1",
    "n2",
    "delta_V",
    "delta_pi",
    "delta_z",
    "argmax_agree",
    "z_pre_norm",
    "z_post_norm",
    "saturated",
)


@dataclass
class StatSummary:
    mean: float
    std: float
    n: int


@dataclass
class GateResult:
    passed: bool
    details: str


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def radius_dir(radius: float) -> str:
    if float(radius).is_integer():
        return f"R{int(radius)}"
    return f"R{radius}"


def radius_label(radius: float) -> str:
    if radius == 0:
        return "proj. off"
    if float(radius).is_integer():
        return f"R={int(radius)}"
    return f"R={radius}"


def safe_pstdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def safe_stdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty sequence")
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return float(sorted_values[idx])


def find_csv(
    results_roots: Sequence[Path],
    seed: int,
    radius: float,
    model: str,
    batch: str,
) -> Optional[Path]:
    rel = Path(f"seed{seed}") / radius_dir(radius) / f"{model}_{batch}_per_state.csv"
    for root in results_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def load_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def row_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def compute_seed_metrics(
    rows: List[Dict[str, str]],
    radius: float,
    n1: int,
    n2: int,
) -> Tuple[Dict[str, float], List[float]]:
    filtered = [row for row in rows if int(row["n1"]) == n1 and int(row["n2"]) == n2]
    if not filtered:
        raise ValueError(f"No rows found for n1={n1}, n2={n2}")

    metric_values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
    z_pre_values: List[float] = []

    for row in filtered:
        metric_values["delta_V"].append(row_float(row, "delta_V"))
        metric_values["delta_z"].append(row_float(row, "delta_z"))
        metric_values["argmax_agree"].append(row_float(row, "argmax_agree"))
        metric_values["delta_pi"].append(row_float(row, "delta_pi"))

        saturated = row_float(row, "saturated")
        if saturated >= 0:
            z_pre = row_float(row, "z_pre_norm")
            z_post = row_float(row, "z_post_norm")
            metric_values["saturated"].append(saturated)
            metric_values["z_pre_norm"].append(z_pre)
            metric_values["z_post_norm"].append(z_post)
            metric_values["proj_disp_proxy"].append(z_pre - z_post)
            if radius == 10.0:
                z_pre_values.append(z_pre)

    seed_summary: Dict[str, float] = {}
    for metric, values in metric_values.items():
        if values:
            seed_summary[metric] = statistics.mean(values)
        else:
            seed_summary[metric] = float("nan")

    return seed_summary, z_pre_values


def aggregate_seed_means(seed_means: Sequence[float]) -> StatSummary:
    cleaned = [x for x in seed_means if not math.isnan(x)]
    if not cleaned:
        return StatSummary(mean=float("nan"), std=float("nan"), n=0)
    return StatSummary(
        mean=statistics.mean(cleaned),
        std=safe_stdev(cleaned),
        n=len(cleaned),
    )


def format_stat(stat: StatSummary, digits: int = 4) -> str:
    if stat.n == 0 or math.isnan(stat.mean):
        return "N/A"
    return f"{stat.mean:.{digits}f}±{stat.std:.{digits}f} (n={stat.n})"


def render_gate_table(gates: Dict[str, GateResult]) -> str:
    lines = ["| Gate | Status | Details |", "|---|---|---|"]
    for gate_name, gate in gates.items():
        status = "PASS" if gate.passed else "FAIL"
        details = gate.details.replace("\n", " ")
        lines.append(f"| {gate_name} | {status} | {details} |")
    return "\n".join(lines)


def write_csv_summary(
    out_path: Path,
    stats: Dict[str, Dict[float, Dict[str, StatSummary]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "radius", "metric", "mean", "std", "n"])
        for model in MODELS:
            for radius in sorted(stats[model].keys()):
                for metric in ALL_METRICS:
                    stat = stats[model][radius][metric]
                    writer.writerow([model, radius, metric, stat.mean, stat.std, stat.n])


def write_markdown_summary(
    out_path: Path,
    stage_label: str,
    stats: Dict[str, Dict[float, Dict[str, StatSummary]]],
    radii: Sequence[float],
    n1: int,
    n2: int,
    batch: str,
    notes: Sequence[str],
) -> None:
    lines = [
        f"# {stage_label} Fixed-Pair Finite-R Summary",
        "",
        f"- Batch: `{batch}`",
        f"- Fixed pair: `n1={n1}`, `n2={n2}`",
        f"- Radii: `{', '.join(radius_label(r) for r in radii)}`",
        "",
    ]

    for note in notes:
        lines.append(f"- {note}")
    if notes:
        lines.append("")

    for metric in ("delta_V", "delta_z", "argmax_agree", "saturated", "proj_disp_proxy"):
        lines.append(f"## {METRIC_LABELS[metric]}")
        lines.append("")
        lines.append("| Radius | No Contraction | Contraction |")
        lines.append("|---|---|---|")
        for radius in radii:
            a = format_stat(stats["model_a"][radius][metric], digits=4)
            b = format_stat(stats["model_b"][radius][metric], digits=4)
            lines.append(f"| {radius_label(radius)} | {a} | {b} |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def plot_primary_metrics(
    out_path: Path,
    stats: Dict[str, Dict[float, Dict[str, StatSummary]]],
    radii: Sequence[float],
    batch: str,
    n1: int,
    n2: int,
) -> None:
    if plt is None:
        return
    labels = [radius_label(r) for r in radii]
    x = list(range(len(radii)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = (
        ("delta_V", True),
        ("delta_z", True),
        ("argmax_agree", False),
    )

    for ax, (metric, use_log) in zip(axes, metrics):
        for model in MODELS:
            means = []
            errs = []
            for radius in radii:
                stat = stats[model][radius][metric]
                means.append(stat.mean if stat.n > 0 else float("nan"))
                errs.append(stat.std if stat.n > 1 else 0.0)
            ax.errorbar(
                x,
                means,
                yerr=errs,
                marker="o",
                capsize=4,
                linewidth=2.0,
                label=LABELS[model],
                color=COLORS[model],
            )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        if use_log:
            finite = [m for model in MODELS for m in [stats[model][r][metric].mean for r in radii] if not math.isnan(m) and m > 0]
            if finite:
                ax.set_yscale("log")
        else:
            ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)

    axes[-1].legend(loc="lower right")
    fig.suptitle(f"Finite-R Primary Metrics ({batch}, n1={n1}, n2={n2})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_metrics(
    out_path: Path,
    stats: Dict[str, Dict[float, Dict[str, StatSummary]]],
    radii: Sequence[float],
    batch: str,
    n1: int,
    n2: int,
) -> None:
    if plt is None:
        return
    labels = [radius_label(r) for r in radii]
    x = list(range(len(radii)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = ("saturated", "proj_disp_proxy")

    for ax, metric in zip(axes, metrics):
        for model in MODELS:
            means = []
            errs = []
            for radius in radii:
                stat = stats[model][radius][metric]
                means.append(stat.mean if stat.n > 0 else float("nan"))
                errs.append(stat.std if stat.n > 1 else 0.0)
            ax.errorbar(
                x,
                means,
                yerr=errs,
                marker="o",
                capsize=4,
                linewidth=2.0,
                label=LABELS[model],
                color=COLORS[model],
            )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        if metric == "saturated":
            ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    axes[-1].legend(loc="best")
    fig.suptitle(f"Finite-R Mechanism Metrics ({batch}, n1={n1}, n2={n2})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_transition_window(
    out_path: Path,
    z_pre_values: Sequence[float],
    source_radius: float,
) -> Dict[str, float]:
    sorted_values = sorted(z_pre_values)
    payload = {
        "source_radius": source_radius,
        "n": len(sorted_values),
        "min": min(sorted_values),
        "p05": quantile(sorted_values, 0.05),
        "p50": quantile(sorted_values, 0.50),
        "p95": quantile(sorted_values, 0.95),
        "max": max(sorted_values),
        "mean": statistics.mean(sorted_values),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_gate_results(
    stage_label: str,
    stats: Dict[str, Dict[float, Dict[str, StatSummary]]],
    integrity_ok: bool,
    schema_ok: bool,
    z_window: Dict[str, float],
) -> Dict[str, GateResult]:
    gates: Dict[str, GateResult] = {}

    gates["data_integrity"] = GateResult(
        passed=integrity_ok,
        details="All expected fixed-pair per-state CSVs were found." if integrity_ok else "Missing fixed-pair CSV inputs.",
    )

    gates["schema_sanity"] = GateResult(
        passed=schema_ok,
        details="All analyzed CSVs expose the preferred `saturated` schema." if schema_ok else "At least one CSV failed the preferred `saturated` schema check.",
    )

    def endpoint_pass(model: str) -> Tuple[bool, str]:
        active = stats[model][10.0]
        inactive = stats[model][100.0]
        checks = [
            active["delta_V"].mean < inactive["delta_V"].mean,
            active["delta_z"].mean < inactive["delta_z"].mean,
            active["argmax_agree"].mean > inactive["argmax_agree"].mean,
        ]
        details = (
            f"{LABELS[model]}: R=10 vs R=100 "
            f"Delta_V {active['delta_V'].mean:.4f} < {inactive['delta_V'].mean:.4f}, "
            f"Delta_z {active['delta_z'].mean:.4f} < {inactive['delta_z'].mean:.4f}, "
            f"Argmax {active['argmax_agree'].mean:.4f} > {inactive['argmax_agree'].mean:.4f}"
        )
        return all(checks), details

    model_a_ok, model_a_details = endpoint_pass("model_a")
    model_b_ok, model_b_details = endpoint_pass("model_b")
    gates["primary_endpoint_separation"] = GateResult(
        passed=model_a_ok and model_b_ok,
        details=f"{model_a_details}; {model_b_details}",
    )

    window_width = z_window["p95"] - z_window["p05"]
    transition_ok = window_width <= 5.0
    gates["transition_localization"] = GateResult(
        passed=transition_ok,
        details=(
            f"R=10 z_pre_norm width p95-p05 = {window_width:.4f} "
            f"(p05={z_window['p05']:.4f}, p95={z_window['p95']:.4f})"
        ),
    )

    inactive_ok = True
    inactive_details = []
    for model in MODELS:
        off = stats[model][0.0]
        inactive = stats[model][100.0]
        diffs = {
            metric: abs(off[metric].mean - inactive[metric].mean)
            for metric in PRIMARY_METRICS
        }
        model_ok = diffs["delta_V"] <= 1e-6 and diffs["delta_z"] <= 1e-6 and diffs["argmax_agree"] <= 1e-6
        inactive_ok &= model_ok
        inactive_details.append(
            f"{LABELS[model]} off vs R=100 diffs "
            f"(Delta_V={diffs['delta_V']:.6f}, Delta_z={diffs['delta_z']:.6f}, Argmax={diffs['argmax_agree']:.6f})"
        )
    gates["inactive_plateau_check"] = GateResult(
        passed=inactive_ok,
        details="; ".join(inactive_details),
    )

    if stage_label == "stage1":
        intermediate_radii = [r for r in stats["model_a"].keys() if r not in {0.0, 10.0, 100.0}]

        def intermediate_pass(model: str) -> Tuple[bool, str]:
            active = stats[model][10.0]
            inactive = stats[model][100.0]
            hits = []
            for radius in intermediate_radii:
                cur = stats[model][radius]
                between = (
                    active["delta_V"].mean < cur["delta_V"].mean < inactive["delta_V"].mean
                    and active["delta_z"].mean < cur["delta_z"].mean < inactive["delta_z"].mean
                    and inactive["argmax_agree"].mean < cur["argmax_agree"].mean < active["argmax_agree"].mean
                )
                if between:
                    hits.append(radius_label(radius))
            details = f"{LABELS[model]} intermediate hits: {', '.join(hits) if hits else 'none'}"
            return bool(hits), details

        a_ok, a_details = intermediate_pass("model_a")
        b_ok, b_details = intermediate_pass("model_b")
        gates["intermediate_regime_exists"] = GateResult(
            passed=a_ok and b_ok,
            details=f"{a_details}; {b_details}",
        )

        def directional_pass(model: str) -> Tuple[bool, str]:
            ordered = [r for r in sorted(stats[model].keys()) if r != 0.0]
            sat_means = [stats[model][r]["saturated"].mean for r in ordered if stats[model][r]["saturated"].n > 0]
            sat_nonincreasing = all(x >= y - 1e-9 for x, y in zip(sat_means, sat_means[1:]))
            details = f"{LABELS[model]} saturation means over finite R: {', '.join(f'{m:.3f}' for m in sat_means)}"
            return sat_nonincreasing, details

        a_ok, a_details = directional_pass("model_a")
        b_ok, b_details = directional_pass("model_b")
        gates["directional_coherence"] = GateResult(
            passed=a_ok and b_ok,
            details=f"{a_details}; {b_details}",
        )

    return gates


def write_gate_report(
    out_path: Path,
    stage_label: str,
    gates: Dict[str, GateResult],
    transition_window: Dict[str, float],
    note_lines: Sequence[str],
) -> None:
    overall = all(gate.passed for gate in gates.values())
    lines = [
        f"# {stage_label.upper()} Gate Report",
        "",
        f"- Overall: `{'PASS' if overall else 'FAIL'}`",
        "",
        render_gate_table(gates),
        "",
        "## Transition Window",
        "",
        "```json",
        json.dumps(transition_window, indent=2),
        "```",
        "",
        "## Notes",
        "",
    ]
    for note in note_lines:
        lines.append(f"- {note}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def analyze(
    results_roots: Sequence[Path],
    out_dir: Path,
    radii: Sequence[float],
    seeds: Sequence[int],
    batch: str,
    n1: int,
    n2: int,
    stage_label: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_files: List[str] = []
    schema_errors: List[str] = []
    stats: Dict[str, Dict[float, Dict[str, StatSummary]]] = {
        model: {radius: {} for radius in radii} for model in MODELS
    }
    seed_tables: Dict[str, Dict[float, Dict[str, List[float]]]] = {
        model: {radius: {metric: [] for metric in ALL_METRICS} for radius in radii}
        for model in MODELS
    }
    transition_values: List[float] = []

    for radius in radii:
        for model in MODELS:
            for seed in seeds:
                csv_path = find_csv(results_roots, seed, radius, model, batch)
                if csv_path is None:
                    missing_files.append(f"seed{seed}/{radius_dir(radius)}/{model}_{batch}_per_state.csv")
                    continue

                rows, fieldnames = load_rows(csv_path)
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
                has_legacy_only = "saturation" in fieldnames and "saturated" not in fieldnames
                if missing_cols or has_legacy_only:
                    schema_errors.append(
                        f"{csv_path}: missing columns {missing_cols}, fieldnames={tuple(fieldnames)}"
                    )
                    continue

                try:
                    seed_summary, z_pre_values = compute_seed_metrics(rows, radius, n1, n2)
                except ValueError as exc:
                    schema_errors.append(f"{csv_path}: {exc}")
                    continue

                if radius == 10.0 and z_pre_values:
                    transition_values.extend(z_pre_values)

                for metric in ALL_METRICS:
                    seed_tables[model][radius][metric].append(seed_summary[metric])

    integrity_ok = not missing_files
    schema_ok = not schema_errors

    for model in MODELS:
        for radius in radii:
            for metric in ALL_METRICS:
                stats[model][radius][metric] = aggregate_seed_means(seed_tables[model][radius][metric])

    if not transition_values:
        raise RuntimeError("No R=10 z_pre_norm values found for transition-window estimation")

    transition_window = write_transition_window(
        out_dir / "finite_r_transition_window.json",
        transition_values,
        source_radius=10.0,
    )

    notes = [
        "Statistics are computed over seed-level means, not pooled state rows.",
        "proj_disp_proxy = z_pre_norm - z_post_norm is a scalar proxy based on norms, not the true displacement norm.",
        "Rows with saturated < 0 are treated as mechanism-metric N/A.",
    ]
    if plt is None:
        notes.append("matplotlib is unavailable in the current Python environment; figure generation was skipped.")
    if missing_files:
        notes.append(f"Missing files: {len(missing_files)}")
    if schema_errors:
        notes.append(f"Schema errors: {len(schema_errors)}")

    csv_name = f"finite_r_{batch}_n1_{n1}_n2_{n2}.csv"
    md_name = f"finite_r_{batch}_n1_{n1}_n2_{n2}.md"
    primary_png = f"finite_r_primary_{batch}_n1_{n1}_n2_{n2}.png"
    mech_png = f"finite_r_mechanism_{batch}_n1_{n1}_n2_{n2}.png"

    write_csv_summary(out_dir / csv_name, stats)
    write_markdown_summary(out_dir / md_name, stage_label, stats, radii, n1, n2, batch, notes)
    plot_primary_metrics(out_dir / primary_png, stats, radii, batch, n1, n2)
    plot_mechanism_metrics(out_dir / mech_png, stats, radii, batch, n1, n2)

    gates = build_gate_results(stage_label, stats, integrity_ok, schema_ok, transition_window)
    write_gate_report(
        out_dir / f"{stage_label}_gate_report.md",
        stage_label,
        gates,
        transition_window,
        note_lines=notes + missing_files[:10] + schema_errors[:10],
    )

    payload = {
        "stage": stage_label,
        "batch": batch,
        "n1": n1,
        "n2": n2,
        "radii": list(radii),
        "overall_pass": all(gate.passed for gate in gates.values()),
        "gates": {
            name: {"passed": gate.passed, "details": gate.details}
            for name, gate in gates.items()
        },
        "transition_window": transition_window,
    }
    (out_dir / f"{stage_label}_gate_report.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_roots",
        required=True,
        help="Comma-separated list of results roots to search, in priority order.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for summaries/figures.")
    parser.add_argument("--radii", required=True, help="Comma-separated radius list.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed list.")
    parser.add_argument("--batch", default="b0", choices=["b0", "b1"], help="Batch selector.")
    parser.add_argument("--n1", type=int, default=2, help="Training depth in CSV rows.")
    parser.add_argument("--n2", type=int, default=8, help="Eval depth in CSV rows.")
    parser.add_argument("--stage", default="stage0", choices=["stage0", "stage1"], help="Stage label for gate logic.")
    args = parser.parse_args()

    results_roots = [Path(p.strip()) for p in args.results_roots.split(",") if p.strip()]
    radii = parse_float_list(args.radii)
    seeds = parse_int_list(args.seeds)

    return analyze(
        results_roots=results_roots,
        out_dir=Path(args.out_dir),
        radii=radii,
        seeds=seeds,
        batch=args.batch,
        n1=args.n1,
        n2=args.n2,
        stage_label=args.stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
