"""Immutable, checkpoint-bound artifacts for held-out policy evaluation."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.dataset_provenance import ordered_record_sha256
from utils.run_identity import canonical_json_bytes, canonical_json_sha256
from utils.compute_accounting import validate_compute_snapshot


EVALUATION_ARTIFACT_SCHEMA_VERSION = 3
RECORD_LOCAL_SEED_SCHEME = "sha256_base_seed_record_sha256_u32be_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationArtifactError(RuntimeError):
    """Raised when evaluation evidence is incomplete or cannot be published."""


def record_local_evaluation_seed(base_seed: int, record_sha256: str) -> int:
    """Derive the version-1 record-local 32-bit evaluation seed."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= 2**32 - 1:
        raise EvaluationArtifactError("Base evaluation seed must be a 32-bit integer.")
    _require_sha256(record_sha256, field="record_sha256")
    payload = f"{base_seed}:{record_sha256}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise EvaluationArtifactError(f"{field} must be 64 lowercase hex characters.")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationArtifactError(f"{field} must be a non-negative integer.")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationArtifactError(f"{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationArtifactError(f"{field} must be finite.")
    return result


def _sudoku_stats_from_plan(plan: Sequence[int]) -> tuple[tuple[int, int, int, int], bool]:
    total_cells = len(plan)
    if total_cells == 16:
        size, box_size = 4, 2
    elif total_cells == 81:
        size, box_size = 9, 3
    else:
        raise EvaluationArtifactError("Sudoku plan must contain 16 or 81 cells.")
    digits = [token - 1 if 2 <= token <= size + 1 else 0 for token in plan]

    def duplicate_count(values: Sequence[int]) -> int:
        filled = [value for value in values if value > 0]
        return len(filled) - len(set(filled))

    violations = 0
    for row in range(size):
        violations += duplicate_count(digits[row * size : (row + 1) * size])
    for column in range(size):
        violations += duplicate_count(
            [digits[row * size + column] for row in range(size)]
        )
    for box_row in range(0, size, box_size):
        for box_column in range(0, size, box_size):
            violations += duplicate_count(
                [
                    digits[(box_row + row) * size + box_column + column]
                    for row in range(box_size)
                    for column in range(box_size)
                ]
            )

    zero_candidates = 0
    all_digits = set(range(1, size + 1))
    for row in range(size):
        for column in range(size):
            index = row * size + column
            if digits[index] != 0:
                continue
            blocked = set(digits[row * size : (row + 1) * size])
            blocked.update(digits[other_row * size + column] for other_row in range(size))
            box_row = (row // box_size) * box_size
            box_column = (column // box_size) * box_size
            blocked.update(
                digits[(box_row + inner_row) * size + box_column + inner_column]
                for inner_row in range(box_size)
                for inner_column in range(box_size)
            )
            if not (all_digits - blocked):
                zero_candidates += 1

    filled = sum(2 <= token <= size + 1 for token in plan)
    solved = filled == total_cells and violations == 0
    return (total_cells, filled, violations, zero_candidates), solved


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "artifact_schema_version",
        "run_id",
        "algorithm",
        "training_seed",
        "producer_git_commit",
        "effective_config_sha256",
        "dataset_provenance_sha256",
        "checkpoint_sha256",
        "checkpoint_environment_steps",
        "checkpoint_outer_steps",
        "evaluation_seed",
        "evaluation_seed_scheme",
        "policy_mode",
        "reward_definition",
        "dataset",
        "environment",
    }
    if set(metadata) != required:
        raise EvaluationArtifactError(
            "Evaluation metadata field inventory differs from schema 3."
        )
    if metadata["artifact_schema_version"] != EVALUATION_ARTIFACT_SCHEMA_VERSION:
        raise EvaluationArtifactError("Unsupported evaluation artifact schema.")
    for field in ("run_id", "algorithm", "policy_mode", "reward_definition"):
        value = metadata[field]
        if not isinstance(value, str) or not value or not value.isascii():
            raise EvaluationArtifactError(f"{field} must be a nonempty ASCII string.")
    if metadata["evaluation_seed_scheme"] != RECORD_LOCAL_SEED_SCHEME:
        raise EvaluationArtifactError("Unsupported record-local evaluation seed scheme.")
    producer_commit = metadata["producer_git_commit"]
    if (
        not isinstance(producer_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", producer_commit)
    ):
        raise EvaluationArtifactError(
            "producer_git_commit must be 40 lowercase hex characters."
        )
    for field in (
        "effective_config_sha256",
        "dataset_provenance_sha256",
        "checkpoint_sha256",
    ):
        _require_sha256(metadata[field], field=field)
    for field in (
        "training_seed",
        "checkpoint_environment_steps",
        "checkpoint_outer_steps",
        "evaluation_seed",
    ):
        _require_nonnegative_int(metadata[field], field=field)

    dataset = metadata["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "split",
        "manifest_sha256",
        "ordered_record_sha256",
        "record_count",
    }:
        raise EvaluationArtifactError("Evaluation dataset metadata is incomplete.")
    if not isinstance(dataset["split"], str) or not dataset["split"]:
        raise EvaluationArtifactError("Evaluation dataset split must be nonempty.")
    _require_sha256(dataset["manifest_sha256"], field="dataset.manifest_sha256")
    _require_sha256(
        dataset["ordered_record_sha256"],
        field="dataset.ordered_record_sha256",
    )
    count = _require_nonnegative_int(dataset["record_count"], field="dataset.record_count")
    if count == 0:
        raise EvaluationArtifactError("Evaluation dataset record_count must be positive.")
    environment = metadata["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {
        "action_count",
        "plan_length",
        "vocab_size",
        "max_edits",
        "stop_action_id",
        "stop_action_mode",
        "task_name",
        "undo_enabled",
    }:
        raise EvaluationArtifactError("Evaluation environment metadata is incomplete.")
    action_count = _require_nonnegative_int(
        environment["action_count"], field="environment.action_count"
    )
    plan_length = _require_nonnegative_int(
        environment["plan_length"], field="environment.plan_length"
    )
    vocab_size = _require_nonnegative_int(
        environment["vocab_size"], field="environment.vocab_size"
    )
    max_edits = _require_nonnegative_int(
        environment["max_edits"], field="environment.max_edits"
    )
    stop_action_id = _require_nonnegative_int(
        environment["stop_action_id"], field="environment.stop_action_id"
    )
    if min(action_count, plan_length, vocab_size, max_edits) == 0:
        raise EvaluationArtifactError("Evaluation environment dimensions must be positive.")
    if stop_action_id >= action_count:
        raise EvaluationArtifactError("STOP action lies outside the action space.")
    if environment["stop_action_mode"] not in {"terminal", "noop", "disabled"}:
        raise EvaluationArtifactError("Unsupported STOP action mode.")
    if not isinstance(environment["task_name"], str) or not environment["task_name"]:
        raise EvaluationArtifactError("Environment task_name must be nonempty.")
    if not isinstance(environment["undo_enabled"], bool):
        raise EvaluationArtifactError("Environment undo_enabled must be boolean.")
    if environment["undo_enabled"]:
        raise EvaluationArtifactError(
            "Schema-3 evidence does not support history-dependent UNDO actions."
        )
    if action_count != plan_length * vocab_size + 1:
        raise EvaluationArtifactError(
            "Action count does not match edit actions plus one STOP action."
        )
    return dict(metadata)


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        raise EvaluationArtifactError("Evaluation artifact requires per-instance rows.")
    expected_fields = {
        "record_index",
        "record_sha256",
        "evaluation_seed",
        "success",
        "initial_checker_score",
        "final_checker_score",
        "undiscounted_shaped_return",
        "environment_interactions",
        "invalid_action_count",
        "termination_reason",
        "actions",
        "rewards",
        "final_plan",
        "final_plan_sha256",
        "sudoku",
    }
    normalized: list[dict[str, Any]] = []
    for expected_index, source in enumerate(rows):
        if not isinstance(source, Mapping) or set(source) != expected_fields:
            raise EvaluationArtifactError(
                f"Per-instance row {expected_index} has an invalid field inventory."
            )
        row = dict(source)
        index = _require_nonnegative_int(
            row["record_index"], field=f"rows[{expected_index}].record_index"
        )
        if index != expected_index:
            raise EvaluationArtifactError("Per-instance rows must be in dataset order.")
        _require_sha256(
            row["record_sha256"], field=f"rows[{expected_index}].record_sha256"
        )
        _require_sha256(
            row["final_plan_sha256"],
            field=f"rows[{expected_index}].final_plan_sha256",
        )
        _require_nonnegative_int(
            row["evaluation_seed"], field=f"rows[{expected_index}].evaluation_seed"
        )
        if not isinstance(row["success"], bool):
            raise EvaluationArtifactError("Per-instance success must be boolean.")
        for field in (
            "initial_checker_score",
            "final_checker_score",
            "undiscounted_shaped_return",
        ):
            _require_finite_number(row[field], field=f"rows[{expected_index}].{field}")
        steps = _require_nonnegative_int(
            row["environment_interactions"],
            field=f"rows[{expected_index}].environment_interactions",
        )
        invalid = _require_nonnegative_int(
            row["invalid_action_count"],
            field=f"rows[{expected_index}].invalid_action_count",
        )
        if invalid > steps:
            raise EvaluationArtifactError("Invalid-action count exceeds interactions.")
        if not isinstance(row["termination_reason"], str) or not row["termination_reason"]:
            raise EvaluationArtifactError("Termination reason must be nonempty.")
        actions = row["actions"]
        rewards = row["rewards"]
        if (
            not isinstance(actions, list)
            or not isinstance(rewards, list)
            or len(actions) != steps
            or len(rewards) != steps
        ):
            raise EvaluationArtifactError(
                "Action and reward sequences must match environment interactions."
            )
        environment = metadata["environment"]
        action_count = int(environment["action_count"])
        stop_action_id = int(environment["stop_action_id"])
        max_edits = int(environment["max_edits"])
        if steps == 0 or steps > max_edits:
            raise EvaluationArtifactError(
                "Episode interactions must lie in [1, max_edits]."
            )
        if any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or action < 0
            or action >= action_count
            for action in actions
        ):
            raise EvaluationArtifactError("Actions must lie inside the action space.")
        if (
            environment["stop_action_mode"] == "disabled"
            and stop_action_id in actions
        ):
            raise EvaluationArtifactError("Disabled STOP action appears in evaluation.")
        termination_reason = row["termination_reason"]
        if termination_reason not in {"solved", "budget", "stop"}:
            raise EvaluationArtifactError("Unsupported termination reason.")
        if termination_reason == "budget" and steps != max_edits:
            raise EvaluationArtifactError(
                "Budget termination must occur at the registered edit limit."
            )
        if termination_reason == "stop":
            if environment["stop_action_mode"] != "terminal" or actions[-1] != stop_action_id:
                raise EvaluationArtifactError(
                    "STOP termination is inconsistent with the action sequence."
                )
        if row["success"] != (termination_reason == "solved"):
            raise EvaluationArtifactError(
                "Success flag differs from the terminal reason."
            )
        for reward in rewards:
            _require_finite_number(reward, field="per-instance reward")
        if not math.isclose(
            sum(float(reward) for reward in rewards),
            float(row["undiscounted_shaped_return"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise EvaluationArtifactError(
                "Per-instance return differs from its recorded reward sequence."
            )
        final_plan = row["final_plan"]
        if (
            not isinstance(final_plan, list)
            or len(final_plan) != int(environment["plan_length"])
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= int(environment["vocab_size"])
                for token in final_plan
            )
        ):
            raise EvaluationArtifactError("Final plan is outside the registered space.")
        if canonical_json_sha256(final_plan) != row["final_plan_sha256"]:
            raise EvaluationArtifactError("Final-plan hash differs from final_plan.")
        sudoku = row["sudoku"]
        if environment["task_name"] == "sudoku" and sudoku is None:
            raise EvaluationArtifactError(
                "Sudoku evaluation rows require final-plan statistics."
            )
        if sudoku is not None:
            if not isinstance(sudoku, Mapping) or set(sudoku) != {
                "total_cells",
                "filled",
                "violations",
                "zero_candidates",
            }:
                raise EvaluationArtifactError("Sudoku statistics are incomplete.")
            for field, value in sudoku.items():
                _require_nonnegative_int(
                    value, field=f"rows[{expected_index}].sudoku.{field}"
                )
            if sudoku["total_cells"] != len(final_plan):
                raise EvaluationArtifactError(
                    "Sudoku cell count differs from final plan length."
                )
            recomputed_stats, solved_from_plan = _sudoku_stats_from_plan(final_plan)
            if tuple(sudoku[field] for field in (
                "total_cells",
                "filled",
                "violations",
                "zero_candidates",
            )) != recomputed_stats:
                raise EvaluationArtifactError(
                    "Sudoku statistics differ from the recorded final plan."
                )
            if row["success"] != solved_from_plan:
                raise EvaluationArtifactError(
                    "Per-instance success differs from the recorded Sudoku plan."
                )
        expected_episode_seed = record_local_evaluation_seed(
            metadata["evaluation_seed"], row["record_sha256"]
        )
        if row["evaluation_seed"] != expected_episode_seed:
            raise EvaluationArtifactError(
                "Per-instance evaluation seed differs from the registered derivation."
            )
        normalized.append(row)

    dataset = metadata["dataset"]
    if len(normalized) != dataset["record_count"]:
        raise EvaluationArtifactError(
            "Per-instance row count differs from evaluation dataset metadata."
        )
    ordered_hash = ordered_record_sha256(
        [row["record_sha256"] for row in normalized]
    )
    if ordered_hash != dataset["ordered_record_sha256"]:
        raise EvaluationArtifactError(
            "Per-instance record order differs from the registered evaluation pool."
        )
    return normalized


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata_sha256: str,
    per_instance_sha256: str,
    compute_snapshot_sha256: str,
) -> dict[str, Any]:
    final_scores = [float(row["final_checker_score"]) for row in rows]
    initial_scores = [float(row["initial_checker_score"]) for row in rows]
    returns = [float(row["undiscounted_shaped_return"]) for row in rows]
    interactions = [int(row["environment_interactions"]) for row in rows]
    invalid_count = sum(int(row["invalid_action_count"]) for row in rows)
    solved_count = sum(bool(row["success"]) for row in rows)
    sudoku_rows = [row["sudoku"] for row in rows if row["sudoku"] is not None]
    summary: dict[str, Any] = {
        "artifact_schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "metadata_sha256": metadata_sha256,
        "per_instance_sha256": per_instance_sha256,
        "compute_snapshot_sha256": compute_snapshot_sha256,
        "record_count": len(rows),
        "solved_count": solved_count,
        "success_rate": solved_count / len(rows),
        "mean_initial_checker_score": _mean(initial_scores),
        "mean_final_checker_score": _mean(final_scores),
        "min_final_checker_score": min(final_scores),
        "max_final_checker_score": max(final_scores),
        "mean_undiscounted_shaped_return": _mean(returns),
        "total_environment_interactions": sum(interactions),
        "mean_environment_interactions": _mean(interactions),
        "invalid_action_rate": invalid_count / max(sum(interactions), 1),
        "sudoku": None,
    }
    if sudoku_rows:
        if len(sudoku_rows) != len(rows):
            raise EvaluationArtifactError(
                "Sudoku statistics must be present for every row or no rows."
            )
        summary["sudoku"] = {
            "mean_filled": _mean([float(row["filled"]) for row in sudoku_rows]),
            "mean_violations": _mean(
                [float(row["violations"]) for row in sudoku_rows]
            ),
            "mean_zero_candidates": _mean(
                [float(row["zero_candidates"]) for row in sudoku_rows]
            ),
        }
    return summary


def _validate_reported_metrics(
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    exact_fields = {
        "total_episodes": "record_count",
        "solved_count": "solved_count",
    }
    for reported, derived in exact_fields.items():
        if metrics.get(reported) != summary[derived]:
            raise EvaluationArtifactError(
                f"Reported {reported} differs from per-instance rows."
            )
    numeric_fields = {
        "success_rate": "success_rate",
        "mean_score": "mean_final_checker_score",
        "score_min": "min_final_checker_score",
        "score_max": "max_final_checker_score",
        "initial_score_mean": "mean_initial_checker_score",
        "mean_return": "mean_undiscounted_shaped_return",
        "mean_steps": "mean_environment_interactions",
        "invalid_action_rate": "invalid_action_rate",
    }
    for reported, derived in numeric_fields.items():
        value = _require_finite_number(metrics.get(reported), field=reported)
        if not math.isclose(
            value,
            float(summary[derived]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvaluationArtifactError(
                f"Reported {reported} differs from per-instance rows."
            )
    sudoku = summary["sudoku"]
    if sudoku is not None:
        for reported, derived in {
            "final_filled_mean": "mean_filled",
            "final_violations_mean": "mean_violations",
            "final_zero_cand_mean": "mean_zero_candidates",
        }.items():
            value = _require_finite_number(metrics.get(reported), field=reported)
            if not math.isclose(
                value,
                float(sudoku[derived]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise EvaluationArtifactError(
                    f"Reported {reported} differs from per-instance rows."
                )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_no_replace(stage: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise EvaluationArtifactError(
            "This host has no atomic no-replace publication primitive."
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise EvaluationArtifactError("Refusing to overwrite an evaluation artifact.")
    raise EvaluationArtifactError(
        f"Atomic evaluation artifact publication failed: {os.strerror(error_number)}."
    )


def write_evaluation_artifact(
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    compute_snapshot: Mapping[str, Any],
    reported_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one immutable evaluation directory."""

    canonical_metadata = _validate_metadata(metadata)
    canonical_rows = _validate_rows(rows, metadata=canonical_metadata)
    try:
        canonical_compute_snapshot = validate_compute_snapshot(
            dict(compute_snapshot)
        )
    except ValueError as exc:
        raise EvaluationArtifactError(f"Invalid compute snapshot: {exc}") from exc
    progress = canonical_compute_snapshot["progress"]
    if not isinstance(progress, dict):
        raise EvaluationArtifactError("Compute progress must be a dictionary.")
    if (
        progress["environment_interactions"]
        != canonical_metadata["checkpoint_environment_steps"]
        or progress["outer_updates"]
        != canonical_metadata["checkpoint_outer_steps"]
    ):
        raise EvaluationArtifactError(
            "Compute progress differs from the evaluated checkpoint."
        )
    model_work = canonical_compute_snapshot["model_work"]
    if not isinstance(model_work, dict):
        raise EvaluationArtifactError("Model work must be a dictionary.")
    if model_work["uninstrumented_roles"]:
        raise EvaluationArtifactError(
            "Evaluation artifacts cannot contain uninstrumented model roles."
        )
    metadata_bytes = canonical_json_bytes(canonical_metadata) + b"\n"
    per_instance_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in canonical_rows
    )
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    per_instance_sha256 = hashlib.sha256(per_instance_bytes).hexdigest()
    compute_snapshot_bytes = canonical_json_bytes(canonical_compute_snapshot) + b"\n"
    compute_snapshot_sha256 = hashlib.sha256(compute_snapshot_bytes).hexdigest()
    summary = _build_summary(
        canonical_rows,
        metadata_sha256=metadata_sha256,
        per_instance_sha256=per_instance_sha256,
        compute_snapshot_sha256=compute_snapshot_sha256,
    )
    if reported_metrics is not None:
        _validate_reported_metrics(reported_metrics, summary)
    summary_bytes = canonical_json_bytes(summary) + b"\n"

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise EvaluationArtifactError("Refusing to overwrite an evaluation artifact.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    try:
        _write_fsynced(stage / "evaluation_metadata.json", metadata_bytes)
        _write_fsynced(stage / "per_instance.jsonl", per_instance_bytes)
        _write_fsynced(stage / "compute_snapshot.json", compute_snapshot_bytes)
        _write_fsynced(stage / "summary.json", summary_bytes)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _publish_directory_no_replace(stage, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "output_dir": str(output),
        "metadata_sha256": metadata_sha256,
        "per_instance_sha256": per_instance_sha256,
        "compute_snapshot_sha256": compute_snapshot_sha256,
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "summary": summary,
    }
