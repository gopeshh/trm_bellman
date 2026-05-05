#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path(
    os.environ.get("UPI_TRM_PAPER_ROOT", Path.home() / "UPI_TRM" / "UPI_TRM_NIPS")
).expanduser()
FULL_STAGE_ROOT = REPO_ROOT / "build" / "neurips2026_anonymized"
FULL_ZIP_PATH = REPO_ROOT / "artifact" / "neurips2026_anonymized.zip"
SUPPLEMENTARY_STAGE_ROOT = REPO_ROOT / "build" / "neurips2026_supplementary"
SUPPLEMENTARY_ZIP_PATH = REPO_ROOT / "artifact" / "neurips2026_supplementary.zip"
STAGE_ROOT = FULL_STAGE_ROOT
ZIP_PATH = FULL_ZIP_PATH

SUCCESS_RE = re.compile(r"\[step\s+(\d+)\] eval_success_rate=(\d+\.\d+)")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")

TEXT_REPLACEMENTS = {
    "ICML Phase 1 unroll sensitivity experiment": "depth-mismatch sensitivity experiment",
    "This is the recommended configuration per CLAUDE.md.": "This is the recommended locked configuration.",
    "value-head norm: OFF (recommended per CLAUDE.md)": "value-head norm: OFF (locked setting)",
    "per CLAUDE.md non-negotiables": "for the locked protocol",
    "Per CLAUDE.md non-negotiables:": "Locked protocol requirements:",
    "required per CLAUDE.md": "required for the locked protocol",
    "recommended per CLAUDE.md": "recommended for the locked protocol",
    "CLAUDE.md": "submission guidance",
    "ICML": "earlier",
    str(REPO_ROOT) + "/": "project_repo/",
    str(REPO_ROOT): "project_repo",
    "/home/buiksat/trm_bellman/": "project_repo/",
    "/home/buiksat/trm_bellman": "project_repo",
    "<sanitized_user_home>/trm_bellman/": "project_repo/",
    "<sanitized_user_home>/trm_bellman": "project_repo",
    "<sanitized_user_home>/UPI_TRM/UPI_TRM_NIPS/": "paper/",
    "<sanitized_user_home>/UPI_TRM/UPI_TRM_NIPS": "paper",
    str(PAPER_ROOT): "paper",
    str(PAPER_ROOT / "results" / "episodic_z_hard_suite_20k_seed41_50") + "/":
        "results/reproduction_inputs/episodic_z_hard_suite_20k_seed41_50/",
    str(PAPER_ROOT / "results" / "episodic_z_validation") + "/":
        "results/reproduction_inputs/episodic_z_validation/",
    str(PAPER_ROOT / "results" / "closure_batch_seed1729.npz"):
        "results/reproduction_inputs/episodic_z_hard_suite_20k_seed41_50/closure_batch_seed1729.npz",
    str(PAPER_ROOT / "results" / "lv_directions_seed1729_eps1e-4.npz"):
        "results/reproduction_inputs/episodic_z_hard_suite_20k_seed41_50/lv_directions_seed1729_eps1e-4.npz",
    "/home/buiksat/fbsource/fbcode": "<internal_build_root>",
    "/home/buiksat/fbsource/fbcode/buiksat_trm/": "",
    "/data/users/buiksat/fbsource/fbcode": "<internal_build_root>",
    "/data/users/buiksat/fbsource": "<internal_source_tree>",
    str(PAPER_ROOT) + "/": "paper/",
    "/data/repos/fbsource/fbcode/": "<internal_build_root>/",
    "/data/repos/fbsource/fbcode": "<internal_build_root>",
    "/data/repos/fbsource/": "<internal_source_tree>/",
    "/data/repos/fbsource": "<internal_source_tree>",
    "fbcode//buiksat_trm:run_baseline": "artifact_target:run_baseline",
    "fbcode//buiksat_trm:": "artifact_target:",
    "//buiksat_trm:upi_trm_train": "artifact_target:upi_trm_train",
    "//buiksat_trm:eval_unroll_sensitivity": "artifact_target:eval_unroll_sensitivity",
    "//buiksat_trm:exp1_value_head_lipschitz": "artifact_target:exp1_value_head_lipschitz",
    "//buiksat_trm:": "artifact_target:",
    "fbsource//third-party/pypi/": "third_party/",
    "/home/buiksat": "<sanitized_user_home>",
    "fbcode//buiksat_trm": "artifact_target",
    "fbsource//": "<internal_source_tree>/",
    "www.internalfb.com": "<internal_doc_host>",
    "_buiksat_cpu_one_hot_patch": "_artifact_cpu_one_hot_patch",
    str(Path.home()): "<sanitized_user_home>",
    "buiksat_trm": "artifact_project",
    "trm_bellman": "project_repo",
    "fbsource": "<internal_source_tree>",
    "fbcode": "<internal_build_root>",
    "CLAUDE": "submission_notes",
    "buiksat": "anonymous",
}

BANNED_STRINGS = (
    "/home/buiksat",
    str(Path.home()),
    "buiksat",
    "trm_bellman",
    "fbcode//buiksat_trm",
    "buiksat_trm",
    "fbsource",
    "fbcode",
    "fbsource//",
    "www.internalfb.com",
    "_buiksat_cpu_one_hot_patch",
    "CLAUDE",
)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text)


def _copy_binary(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    shutil.copy2(src, dst)


def _sanitize_text(text: str) -> str:
    sanitized = text
    for needle, replacement in TEXT_REPLACEMENTS.items():
        sanitized = sanitized.replace(needle, replacement)
    return sanitized


def _sanitize_string_value(value: str) -> str:
    sanitized = _sanitize_text(value)
    if sanitized.startswith("configs/"):
        return f"code/{sanitized}"
    if sanitized.startswith("external_baselines/"):
        return f"code/{sanitized}"
    if sanitized.startswith("run_baseline.py"):
        return "code/run_baseline.py"
    if sanitized.startswith("results/neurips2026/external_hard4x4_20k/"):
        return sanitized.replace(
            "results/neurips2026/external_hard4x4_20k/",
            "results/reproduction_inputs/trusted_external_raw/",
            1,
        )
    if sanitized.startswith("results/neurips2026/external_hard4x4/"):
        return sanitized.replace(
            "results/neurips2026/external_hard4x4/",
            "results/reproduction_inputs/trusted_external_raw/",
            1,
        )
    if sanitized.endswith(".pt"):
        return f"omitted_from_artifact/{Path(sanitized).name}"
    if sanitized.endswith(".zip"):
        return f"omitted_from_artifact/{Path(sanitized).name}"
    return sanitized


def _sanitize_json_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        sanitized: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"git_sha", "commits"}:
                continue
            sanitized[key] = _sanitize_json_obj(value)
        return sanitized
    if isinstance(obj, list):
        return [_sanitize_json_obj(item) for item in obj]
    if isinstance(obj, str):
        return _sanitize_string_value(obj)
    return obj


def _copy_sanitized_text(src: Path, dst: Path) -> None:
    _write_text(dst, _sanitize_text(src.read_text()))


def _copy_sanitized_json(src: Path, dst: Path) -> None:
    payload = json.loads(src.read_text())
    _write_text(dst, json.dumps(_sanitize_json_obj(payload), indent=2, sort_keys=True) + "\n")


def _copy_sanitized_jsonl(src: Path, dst: Path) -> None:
    lines = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        lines.append(json.dumps(_sanitize_json_obj(payload), sort_keys=True))
    _write_text(dst, "\n".join(lines) + ("\n" if lines else ""))


def _paper_asset_paths() -> list[Path]:
    main_tex = (PAPER_ROOT / "main.tex").read_text()
    asset_paths = {
        PAPER_ROOT / "main.tex",
        PAPER_ROOT / "neurips_2026.sty",
        PAPER_ROOT / "trm_rl.bib",
        PAPER_ROOT / "EXPERIMENTS_PROTOCOL.md",
        PAPER_ROOT / "REEVAL_LOG.md",
        PAPER_ROOT / "EPISODIC_Z_RUN_LOG.md",
        PAPER_ROOT / "TABLE1_PROVENANCE.md",
    }

    for rel_path in GRAPHICS_RE.findall(main_tex):
        candidate = PAPER_ROOT / rel_path
        if candidate.exists():
            asset_paths.add(candidate)

    for rel_path in INPUT_RE.findall(main_tex):
        if not rel_path.endswith(".tex"):
            rel_path = rel_path + ".tex"
        candidate = PAPER_ROOT / rel_path
        if candidate.exists():
            asset_paths.add(candidate)

    return sorted(asset_paths)


def _stage_paper() -> None:
    for src in _paper_asset_paths():
        rel_path = src.relative_to(PAPER_ROOT)
        dst = STAGE_ROOT / "paper" / rel_path
        if src.suffix.lower() in {".tex", ".bib", ".sty", ".md"}:
            _copy_sanitized_text(src, dst)
        else:
            _copy_binary(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No existing path among: {', '.join(str(candidate) for candidate in candidates)}")


def _stage_datasets() -> None:
    for dataset_name in (
        "sudoku-4x4-easy_6to8empties",
        "sudoku-4x4-trivial",
        "sudoku-4x4-ultra-easy",
    ):
        src = REPO_ROOT / "data" / dataset_name
        if not src.exists() and dataset_name == "sudoku-4x4-easy_6to8empties":
            src = REPO_ROOT / "data" / "sudoku-4x4"
        _copy_tree(src, STAGE_ROOT / "data" / dataset_name)


def _patch_hard4x4_aggregate_script(text: str) -> str:
    patched = _sanitize_text(text)
    patched = patched.replace(
        'default="/home/buiksat/trm_bellman/results/table3_hard_6to8"',
        'default="results/reproduction_inputs/hard4x4_upi_nomask_logs"',
    )
    patched = patched.replace(
        'default="results/table3_hard_6to8"',
        'default="results/reproduction_inputs/hard4x4_upi_nomask_logs"',
    )
    patched = patched.replace(
        'default="/home/buiksat/trm_bellman/results/neurips2026/external_hard4x4_20k"',
        'default="results/reproduction_inputs/trusted_external_raw"',
    )
    patched = patched.replace(
        'default="results/neurips2026/external_hard4x4_20k"',
        'default="results/reproduction_inputs/trusted_external_raw"',
    )
    patched = patched.replace(
        'default="/home/buiksat/trm_bellman/results/neurips2026/external_hard4x4"',
        'default="results/reproduction_inputs/trusted_external_raw"',
    )
    patched = patched.replace(
        'default="results/neurips2026/external_hard4x4"',
        'default="results/reproduction_inputs/trusted_external_raw"',
    )
    patched = patched.replace(
        'default="/home/buiksat/trm_bellman/results/paper_ready/hard4x4_trusted_baselines_20k"',
        'default="results/reproduced/hard4x4_trusted_baselines"',
    )
    patched = patched.replace(
        'default="results/paper_ready/hard4x4_trusted_baselines_20k"',
        'default="results/reproduced/hard4x4_trusted_baselines"',
    )
    patched = patched.replace(
        'default="/home/buiksat/trm_bellman/results/paper_ready/hard4x4_trusted_baselines"',
        'default="results/reproduced/hard4x4_trusted_baselines"',
    )
    patched = patched.replace(
        'default="results/paper_ready/hard4x4_trusted_baselines"',
        'default="results/reproduced/hard4x4_trusted_baselines"',
    )
    return patched


def _patch_controlled_2x2_plot_script(text: str) -> str:
    patched = _sanitize_text(text)
    patched = patched.replace("R=0", "proj. off")
    patched = patched.replace("Projection OFF", "Projection off")
    patched = patched.replace("Projection ON", "Projection on")
    patched = patched.replace(
        'results_dir = Path("/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled_nomask")',
        'results_dir = Path("results/reproduction_inputs/hard4x4_controlled_2x2_logs")',
    )
    patched = patched.replace(
        'output_dir = Path("/home/buiksat/UPI_TRM/UPI_TRM_NIPS/figures")',
        'output_dir = Path("results/reproduced/hard4x4_controlled_2x2")',
    )
    patched = patched.replace(
        'print("Run the experiments first: scripts/run_table3_hard_controlled_4gpu.sh")',
        'print("Expected sanitized logs under results/reproduction_inputs/hard4x4_controlled_2x2_logs")',
    )
    return patched


def _patch_exp1_value_head_lipschitz_script(text: str) -> str:
    patched = _sanitize_text(text)
    patched = patched.replace(
        'DEFAULT_CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "exp1_v4_refreeze"',
        'DEFAULT_CHECKPOINT_BASE = PROJECT_ROOT / "omitted_from_artifact" / "exp1_v4_refreeze_checkpoints"',
    )
    patched = patched.replace(
        'DEFAULT_BATCH_PATH = PROJECT_ROOT / "artifacts" / "eval_batches" / "exp1_v4_refreeze" / "b0.pt"',
        'DEFAULT_BATCH_PATH = PROJECT_ROOT / "omitted_from_artifact" / "exp1_v4_refreeze_batches" / "b0.pt"',
    )
    patched = patched.replace(
        'DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "paper_ready" / "exp1_value_head_lipschitz"',
        'DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "reproduced" / "exp1_value_head_lipschitz"',
    )
    patched = patched.replace(
        '"cd /home/buiksat/fbsource/fbcode",',
        '"cd <internal_build_root>",',
    )
    patched = patched.replace(
        '"buck2 run //buiksat_trm:exp1_value_head_lipschitz -- "',
        '"buck2 run artifact_target:exp1_value_head_lipschitz -- "',
    )
    return patched


def _stage_code() -> None:
    _copy_sanitized_text(REPO_ROOT / "run_baseline.py", STAGE_ROOT / "code" / "run_baseline.py")
    _copy_sanitized_text(REPO_ROOT / "requirements.txt", STAGE_ROOT / "code" / "requirements.txt")

    for filename in ("__init__.py", "npy_reader.py", "sudoku4x4_env.py"):
        _copy_sanitized_text(
            REPO_ROOT / "external_baselines" / filename,
            STAGE_ROOT / "code" / "external_baselines" / filename,
        )

    aggregate_text = (REPO_ROOT / "scripts" / "aggregate_hard4x4_trusted_baselines.py").read_text()
    _write_text(
        STAGE_ROOT / "code" / "scripts" / "aggregate_hard4x4_trusted_baselines.py",
        _patch_hard4x4_aggregate_script(aggregate_text),
    )

    _copy_sanitized_text(
        REPO_ROOT / "scripts" / "analyze_exp1_finite_r_sweep.py",
        STAGE_ROOT / "code" / "scripts" / "analyze_exp1_finite_r_sweep.py",
    )
    _copy_sanitized_text(
        REPO_ROOT / "scripts" / "build_neurips2026_anonymized_artifact.py",
        STAGE_ROOT / "code" / "scripts" / "build_neurips2026_anonymized_artifact.py",
    )
    _write_text(
        STAGE_ROOT / "code" / "scripts" / "exp1_value_head_lipschitz.py",
        _patch_exp1_value_head_lipschitz_script(
            (REPO_ROOT / "scripts" / "exp1_value_head_lipschitz.py").read_text()
        ),
    )

    controlled_text = (REPO_ROOT / "scripts" / "plot_table3_hard_controlled.py").read_text()
    _write_text(
        STAGE_ROOT / "code" / "scripts" / "plot_table3_hard_controlled.py",
        _patch_controlled_2x2_plot_script(controlled_text),
    )
    for script_name in (
        "materialize_hard4x4_closure_batch.py",
        "episodic_z_hard_suite_diagnostics.py",
        "run_episodic_z_hard_suite_10seed_2gpu.sh",
    ):
        _copy_sanitized_text(
            REPO_ROOT / "scripts" / script_name,
            STAGE_ROOT / "code" / "scripts" / script_name,
        )

    for config_dir in (
        REPO_ROOT / "configs" / "ablations",
        REPO_ROOT / "configs" / "baselines",
        REPO_ROOT / "configs" / "revision",
        REPO_ROOT / "configs" / "table3_hard_controlled",
        REPO_ROOT / "configs" / "exp2_contraction_sweep",
        REPO_ROOT / "configs" / "exp3_projection_ablation",
    ):
        for src in sorted(config_dir.glob("*.yaml")):
            _copy_sanitized_text(src, STAGE_ROOT / "code" / src.relative_to(REPO_ROOT))


def _extract_success_lines(src: Path) -> list[str]:
    lines = []
    for line in src.read_text().splitlines():
        if SUCCESS_RE.search(line):
            lines.append(line)
    if not lines:
        raise RuntimeError(f"No eval_success_rate lines found in {src}")
    return lines


def _stage_upi_logs() -> None:
    src_root = REPO_ROOT / "results" / "table3_hard_6to8"
    dst_root = STAGE_ROOT / "results" / "reproduction_inputs" / "hard4x4_upi_nomask_logs"
    for seed in range(10):
        src = src_root / f"m1_persistent_nc_nomask_s{seed}.log"
        lines = _extract_success_lines(src)
        dst = dst_root / src.name
        _write_text(dst, "\n".join(lines) + "\n")


def _stage_controlled_2x2_logs() -> None:
    src_root = REPO_ROOT / "results" / "table3_hard_6to8_controlled_nomask"
    dst_root = STAGE_ROOT / "results" / "reproduction_inputs" / "hard4x4_controlled_2x2_logs"
    for cell in ("nc_r0", "nc_r10", "c_r0", "c_r10"):
        for seed in range(10):
            src = src_root / f"{cell}_s{seed}.log"
            lines = _extract_success_lines(src)
            _write_text(dst_root / src.name, "\n".join(lines) + "\n")


def _sanitize_external_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_json_obj(payload)
    sanitized["dataset_dir"] = "data/sudoku-4x4-easy_6to8empties"
    return sanitized


def _sanitize_external_train_summary(payload: dict[str, Any], algo: str, seed: int) -> dict[str, Any]:
    sanitized = _sanitize_json_obj(payload)
    sanitized["dataset_dir"] = "data/sudoku-4x4-easy_6to8empties"
    artifacts = sanitized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["eval_history_jsonl"] = (
            f"results/reproduction_inputs/trusted_external_raw/{algo}/seed{seed}/eval_history.jsonl"
        )
        artifacts["run_config_json"] = (
            f"results/reproduction_inputs/trusted_external_raw/{algo}/seed{seed}/run_config.json"
        )
        artifacts.pop("final_model", None)
    return sanitized


def _stage_external_baseline_inputs() -> None:
    src_root = REPO_ROOT / "results" / "neurips2026" / "external_hard4x4_20k"
    dst_root = STAGE_ROOT / "results" / "reproduction_inputs" / "trusted_external_raw"
    for algo in ("ppo", "a2c", "dqn", "dqn_nstep5"):
        for seed in range(10):
            src_dir = src_root / algo / f"seed{seed}"
            dst_dir = dst_root / algo / f"seed{seed}"
            _copy_binary(src_dir / "eval_history.jsonl", dst_dir / "eval_history.jsonl")

            run_config = json.loads((src_dir / "run_config.json").read_text())
            _write_text(
                dst_dir / "run_config.json",
                json.dumps(_sanitize_external_run_config(run_config), indent=2, sort_keys=True) + "\n",
            )

            train_summary = json.loads((src_dir / "train_summary.json").read_text())
            _write_text(
                dst_dir / "train_summary.json",
                json.dumps(
                    _sanitize_external_train_summary(train_summary, algo=algo, seed=seed),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )


def _stage_appendix_inputs() -> None:
    appendix_root = STAGE_ROOT / "results" / "reproduction_inputs" / "appendix_summaries"
    _copy_sanitized_json(
        REPO_ROOT / "results" / "paper_ready" / "exp2_final" / "summary.json",
        appendix_root / "exp2_final_summary.json",
    )
    _copy_sanitized_json(
        REPO_ROOT / "results" / "paper_ready" / "exp3_projection_ablation" / "summary.json",
        appendix_root / "exp3_projection_ablation_summary.json",
    )
    _copy_sanitized_json(
        REPO_ROOT / "results" / "paper_ready" / "exp4_projection_free_dial_v2" / "summary.json",
        appendix_root / "exp4_projection_free_dial_v2_summary.json",
    )
    _copy_sanitized_json(
        REPO_ROOT / "results" / "paper_ready" / "exp5_tradeoff_curve_v2" / "summary.json",
        appendix_root / "exp5_tradeoff_curve_v2_summary.json",
    )

    batch_src = REPO_ROOT / "artifacts" / "eval_batches" / "exp1_v4_refreeze"
    batch_dst = STAGE_ROOT / "results" / "reproduction_inputs" / "exp1_v4_refreeze_batches"
    for filename in ("b0_metadata.json", "b1_metadata.json"):
        _copy_sanitized_json(batch_src / filename, batch_dst / filename)

    finite_r_src = REPO_ROOT / "results" / "plot_data" / "exp1_v4_refreeze_finite_r_residual_stage1_b0"
    finite_r_dst = STAGE_ROOT / "results" / "reproduction_inputs" / "finite_r_stage1_b0"
    for filename in (
        "finite_r_b0_n1_2_n2_8.csv",
        "finite_r_b0_n1_2_n2_8.md",
        "finite_r_transition_window.json",
        "stage1_gate_report.json",
        "stage1_gate_report.md",
    ):
        src = _first_existing_path(
            finite_r_src / filename,
            REPO_ROOT / "documents" / "results" / "plot_data" / "exp1_v4_refreeze_finite_r_residual_stage1_b0" / filename,
        )
        dst = finite_r_dst / filename
        if src.suffix == ".json":
            _copy_sanitized_json(src, dst)
        else:
            _copy_sanitized_text(src, dst)
    for filename in (
        "finite_r_primary_b0_n1_2_n2_8.png",
        "finite_r_mechanism_b0_n1_2_n2_8.png",
        "finite_r_theory_b0_n1_2_n2_8.png",
    ):
        _copy_binary(finite_r_src / filename, finite_r_dst / filename)

    lv_src = REPO_ROOT / "results" / "paper_ready" / "exp1_value_head_lipschitz"
    lv_dst = STAGE_ROOT / "results" / "paper_ready" / "exp1_value_head_lipschitz"
    for filename in (
        "summary.json",
        "CLAIMS.md",
        "PROVENANCE.md",
        "table_exp1_value_head_lipschitz.tex",
    ):
        src = _first_existing_path(
            lv_src / filename,
            REPO_ROOT / "documents" / "results" / "paper_ready" / "exp1_value_head_lipschitz" / filename,
        )
        dst = lv_dst / filename
        if src.suffix == ".json":
            _copy_sanitized_json(src, dst)
        else:
            _copy_sanitized_text(src, dst)


def _stage_episodic_z_theorem_contact() -> None:
    src_root = PAPER_ROOT / "results" / "episodic_z_hard_suite_20k_seed41_50"
    dst_root = STAGE_ROOT / "results" / "reproduction_inputs" / "episodic_z_hard_suite_20k_seed41_50"

    for filename in (
        "summary.json",
        "diagnostics_summary.json",
        "cpi_penalty_grid.json",
    ):
        _copy_sanitized_json(src_root / filename, dst_root / filename)

    for filename in ("failed_jobs.txt", "job_manifest.tsv"):
        _copy_sanitized_text(src_root / filename, dst_root / filename)

    for filename in ("per_seed_eval.jsonl", "per_seed_diagnostics.jsonl"):
        _copy_sanitized_jsonl(src_root / filename, dst_root / filename)

    for src in sorted((src_root / "diagnostics").glob("seed*/step*.json")):
        _copy_sanitized_json(src, dst_root / src.relative_to(src_root))

    for subdir in ("logs", "diagnostic_logs"):
        for src in sorted((src_root / subdir).glob("*.log")):
            _copy_sanitized_text(src, dst_root / src.relative_to(src_root))

    for filename in (
        "closure_batch_seed1729.npz",
        "lv_directions_seed1729_eps1e-4.npz",
    ):
        _copy_binary(PAPER_ROOT / "results" / filename, dst_root / filename)

    validation_src = PAPER_ROOT / "results" / "episodic_z_validation"
    validation_dst = STAGE_ROOT / "results" / "reproduction_inputs" / "episodic_z_validation"
    _copy_sanitized_json(validation_src / "validation_summary.json", validation_dst / "validation_summary.json")


def _stage_hard4x4_paper_ready_outputs() -> None:
    src_dir = REPO_ROOT / "results" / "paper_ready" / "hard4x4_trusted_baselines_20k"
    dst_dir = STAGE_ROOT / "results" / "paper_ready" / "hard4x4_trusted_baselines"

    for filename in ("SUMMARY.md", "summary.csv", "summary.json", "table_hard4x4_trusted_baselines.tex"):
        src = _first_existing_path(
            src_dir / filename,
            REPO_ROOT / "documents" / "results" / "paper_ready" / "hard4x4_trusted_baselines_20k" / filename,
        )
        if src.suffix == ".json":
            _copy_sanitized_json(src, dst_dir / filename)
        else:
            _copy_sanitized_text(src, dst_dir / filename)

    rows = []
    with (src_dir / "per_seed_metrics.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            seed = int(row["seed"])
            if method == "upi_trm":
                row["source"] = (
                    f"results/reproduction_inputs/hard4x4_upi_nomask_logs/"
                    f"m1_persistent_nc_nomask_s{seed}.log"
                )
            else:
                algo = {
                    "sb3_ppo": "ppo",
                    "sb3_a2c": "a2c",
                    "sb3_dqn": "dqn",
                    "sb3_dqn_nstep5": "dqn_nstep5",
                }[method]
                row["source"] = (
                    f"results/reproduction_inputs/trusted_external_raw/{algo}/seed{seed}/train_summary.json"
                )
            rows.append(row)

    _ensure_parent(dst_dir / "per_seed_metrics.csv")
    with (dst_dir / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme() -> None:
    readme = textwrap.dedent(
        """\
        # NeurIPS 2026 Anonymized Reproduction Artifact

        This artifact is a curated, anonymized subset of the project used for the NeurIPS 2026 submission.
        It is intentionally lightweight: it includes the paper sources, synthetic Sudoku data, configs,
        trusted-baseline code, sanitized evaluation outputs, and small derived artifacts needed to rebuild the
        manuscript and inspect the reported paper-facing tables and figures.

        Large training checkpoints, internal build logs, and git history are intentionally omitted.

        ## Layout

        - `paper/`: manuscript sources and the figure/table assets referenced by `paper/main.tex`
        - `code/`: trusted-baseline harness, aggregation scripts, and experiment configs
        - `data/`: small synthetic 4x4 Sudoku datasets used by the paper
        - `results/reproduction_inputs/`: sanitized result snapshots and derived logs used for inspection
        - `results/reproduction_inputs/finite_r_stage1_b0/`: fixed-pair finite-R appendix summaries and plots
        - `results/reproduction_inputs/episodic_z_hard_suite_20k_seed41_50/`: episodic-$z$ theorem-contact rerun bundle, including the frozen closure batch, aggregate diagnostics, per-seed JSONL summaries, and sanitized logs
        - `results/paper_ready/`: paper-facing hard-4x4 trusted-baseline aggregate outputs
        - `results/paper_ready/exp1_value_head_lipschitz/`: sanitized local `L_V` estimates used by Appendix Table 5

        ## Rebuild The Submission PDF

        The commands below rebuild the manuscript PDF from the included paper sources. This reproduces
        Figure 1 and Table 1 as they appear in the submission PDF.

        ```bash
        cd paper
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        bibtex main
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        ```

        ## Reproduce The Trusted-Baseline Aggregate

        The trusted external baseline table can be regenerated from the sanitized per-seed inputs bundled here.

        ```bash
        cd ..
        python3 code/scripts/aggregate_hard4x4_trusted_baselines.py
        ```

        This writes regenerated files to:

        ```text
        results/reproduced/hard4x4_trusted_baselines/
        ```

        ## Optional: Regenerate The Controlled 2x2 Plots

        If `numpy` and `matplotlib` are available, the controlled hard-4x4 projection x contraction plots can be
        rebuilt from the sanitized log snapshots:

        ```bash
        python3 code/scripts/plot_table3_hard_controlled.py
        ```

        This writes regenerated files to:

        ```text
        results/reproduced/hard4x4_controlled_2x2/
        ```

        ## Notes

        - The dataset directory name `sudoku-4x4-easy_6to8empties` is historical. It is the paper's hard 4x4 no-mask suite.
        - The bundled baseline harness includes the exact PPO/A2C/DQN and DQN (n=5) hyperparameters used in the matched-budget 20k trusted-baseline sweep.
        - The finite-R appendix evidence is bundled as a small fixed-pair analysis package under `results/reproduction_inputs/finite_r_stage1_b0/` plus the source analysis script in `code/scripts/analyze_exp1_finite_r_sweep.py`.
        - The appendix `L_V` diagnostic is bundled as sanitized summaries under `results/paper_ready/exp1_value_head_lipschitz/`, with the associated batch metadata under `results/reproduction_inputs/exp1_v4_refreeze_batches/`. The heavyweight frozen batch tensors are omitted from this lightweight artifact, so the bundled measurement script is included for transparency rather than as a turnkey rerun input.
        - The episodic-$z$ theorem-contact appendix evidence is bundled under `results/reproduction_inputs/episodic_z_hard_suite_20k_seed41_50/`, together with `paper/EPISODIC_Z_RUN_LOG.md`, `paper/EXPERIMENTS_PROTOCOL.md`, and the finite-MDP validation summary under `results/reproduction_inputs/episodic_z_validation/validation_summary.json`.
        - Recomputing the trusted-baseline aggregate does not require Stable-Baselines3; it reads the sanitized JSON histories already included here.
        - Running `code/run_baseline.py` for fresh PPO/A2C/DQN and DQN (n=5) training requires local installs of `torch`, `numpy`, `stable-baselines3`, and either `gym` or `gymnasium`. Those packages are not required to rebuild the paper PDF or the included aggregate outputs.
        """
    )
    _write_text(STAGE_ROOT / "README.md", readme)


def _write_supplementary_readme() -> None:
    readme = textwrap.dedent(
        """\
        # NeurIPS 2026 Supplementary Material

        This zip is the anonymized supplementary-material bundle for the submission. It intentionally
        excludes the manuscript source, bibliography, style files, and written appendices; those belong
        only in the submitted paper PDF. The bundle contains code, small synthetic Sudoku datasets, and
        sanitized result snapshots needed to inspect the reported empirical claims.

        Large training checkpoints, internal build logs, git history, and machine-local paths are omitted.

        ## Layout

        - `code/`: trusted-baseline harness, aggregation scripts, diagnostics scripts, and experiment configs
        - `data/`: small synthetic 4x4 Sudoku datasets used by the paper
        - `results/reproduction_inputs/`: sanitized result snapshots, derived logs, and frozen diagnostic inputs
        - `results/reproduction_inputs/finite_r_stage1_b0/`: fixed-pair finite-R diagnostic summaries and plots
        - `results/reproduction_inputs/episodic_z_hard_suite_20k_seed41_50/`: episodic-$z$ theorem-contact rerun summaries, closure batch, per-seed JSONL files, and sanitized logs
        - `results/paper_ready/`: paper-facing aggregate CSV/JSON/Markdown outputs

        ## Reproduce The Trusted-Baseline Aggregate

        The trusted external baseline table can be regenerated from the sanitized per-seed inputs bundled here.

        ```bash
        python3 code/scripts/aggregate_hard4x4_trusted_baselines.py
        ```

        This writes regenerated files to:

        ```text
        results/reproduced/hard4x4_trusted_baselines/
        ```

        ## Optional: Regenerate The Controlled 2x2 Plots

        If `numpy` and `matplotlib` are available, the controlled hard-4x4 projection x contraction plots can be
        rebuilt from the sanitized log snapshots:

        ```bash
        python3 code/scripts/plot_table3_hard_controlled.py
        ```

        This writes regenerated files to:

        ```text
        results/reproduced/hard4x4_controlled_2x2/
        ```

        ## Notes

        - The dataset directory name `sudoku-4x4-easy_6to8empties` is historical. It is the paper's hard 4x4 no-mask suite.
        - The bundled baseline harness includes the PPO/A2C/DQN and DQN (n=5) hyperparameters used in the matched-budget 20k trusted-baseline sweep.
        - Recomputing the trusted-baseline aggregate does not require Stable-Baselines3; it reads the sanitized JSON histories already included here.
        - Running `code/run_baseline.py` for fresh PPO/A2C/DQN and DQN (n=5) training requires local installs of `torch`, `numpy`, `stable-baselines3`, and either `gym` or `gymnasium`.
        """
    )
    _write_text(STAGE_ROOT / "README.md", readme)


def _prune_supplementary_written_material() -> None:
    paper_dir = STAGE_ROOT / "paper"
    if paper_dir.exists():
        shutil.rmtree(paper_dir)
    for path in sorted(STAGE_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".tex", ".bib", ".sty"}:
            path.unlink()


def _assert_supplementary_only() -> None:
    for path in sorted(STAGE_ROOT.rglob("*")):
        if path.is_dir():
            continue
        rel_path = path.relative_to(STAGE_ROOT)
        if rel_path.parts and rel_path.parts[0] == "paper":
            raise RuntimeError(f"Supplementary bundle still contains paper source: {path}")
        if path.suffix.lower() in {".tex", ".bib", ".sty"}:
            raise RuntimeError(f"Supplementary bundle still contains LaTeX source: {path}")


def _zip_stage_dir() -> None:
    _ensure_parent(ZIP_PATH)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE_ROOT.rglob("*")):
            arcname = Path(STAGE_ROOT.name) / path.relative_to(STAGE_ROOT)
            if path.is_dir():
                continue
            zf.write(path, arcname=arcname)


def _assert_no_banned_strings() -> None:
    for path in sorted(STAGE_ROOT.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() in {".pdf", ".png", ".npy", ".npz", ".pt"}:
            continue
        text = path.read_text(errors="ignore")
        for banned in BANNED_STRINGS:
            if banned in text:
                raise RuntimeError(f"Found banned string {banned!r} in {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supplementary-only",
        action="store_true",
        help="Build the NeurIPS supplementary-material zip without paper sources or written appendices.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    global STAGE_ROOT, ZIP_PATH
    if args.supplementary_only:
        STAGE_ROOT = SUPPLEMENTARY_STAGE_ROOT
        ZIP_PATH = SUPPLEMENTARY_ZIP_PATH
    else:
        STAGE_ROOT = FULL_STAGE_ROOT
        ZIP_PATH = FULL_ZIP_PATH

    if not PAPER_ROOT.exists():
        raise FileNotFoundError(f"Missing paper repo: {PAPER_ROOT}")

    _reset_dir(STAGE_ROOT)
    if not args.supplementary_only:
        _stage_paper()
    _stage_code()
    _stage_datasets()
    _stage_upi_logs()
    _stage_controlled_2x2_logs()
    _stage_external_baseline_inputs()
    _stage_appendix_inputs()
    _stage_episodic_z_theorem_contact()
    _stage_hard4x4_paper_ready_outputs()
    if args.supplementary_only:
        _prune_supplementary_written_material()
        _write_supplementary_readme()
        _assert_supplementary_only()
    else:
        _write_readme()
    _assert_no_banned_strings()
    _zip_stage_dir()

    print(f"Staged artifact directory: {STAGE_ROOT}")
    print(f"Artifact ZIP: {ZIP_PATH}")


if __name__ == "__main__":
    main()
