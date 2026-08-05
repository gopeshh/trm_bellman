"""Tests for immutable held-out evaluation artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

from utils.dataset_provenance import ordered_record_sha256
from utils.compute_accounting import (
    COMPUTE_SNAPSHOT_SCHEMA_VERSION,
    MODEL_COUNTER_DEFINITIONS,
    OPTIMIZER_STEP_FIELDS,
    zero_model_counters,
)
from utils.evaluation_artifacts import (
    EVALUATION_ARTIFACT_SCHEMA_VERSION,
    EvaluationArtifactError,
    RECORD_LOCAL_SEED_SCHEME,
    record_local_evaluation_seed,
    write_evaluation_artifact,
)
from utils.run_identity import canonical_json_sha256


class TestEvaluationArtifacts(unittest.TestCase):
    def _episode_seed(self, record_sha256):
        return record_local_evaluation_seed(1729, record_sha256)

    def _rows(self):
        solved_plan = [
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2,
        ]
        partial_plan = [
            1, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2,
        ]
        return [
            {
                "record_index": 0,
                "record_sha256": "1" * 64,
                "evaluation_seed": self._episode_seed("1" * 64),
                "success": True,
                "initial_checker_score": 1.0,
                "final_checker_score": 5.0,
                "undiscounted_shaped_return": 4.0,
                "environment_interactions": 2,
                "invalid_action_count": 0,
                "termination_reason": "solved",
                "actions": [2, 3],
                "rewards": [1.5, 2.5],
                "final_plan": solved_plan,
                "final_plan_sha256": canonical_json_sha256(solved_plan),
                "sudoku": {
                    "total_cells": 16,
                    "filled": 16,
                    "violations": 0,
                    "zero_candidates": 0,
                },
            },
            {
                "record_index": 1,
                "record_sha256": "2" * 64,
                "evaluation_seed": self._episode_seed("2" * 64),
                "success": False,
                "initial_checker_score": 2.0,
                "final_checker_score": 3.0,
                "undiscounted_shaped_return": 1.0,
                "environment_interactions": 4,
                "invalid_action_count": 1,
                "termination_reason": "budget",
                "actions": [1, 2, 3, 4],
                "rewards": [0.0, 0.0, 0.5, 0.5],
                "final_plan": partial_plan,
                "final_plan_sha256": canonical_json_sha256(partial_plan),
                "sudoku": {
                    "total_cells": 16,
                    "filled": 15,
                    "violations": 0,
                    "zero_candidates": 0,
                },
            },
        ]

    def _metadata(self, rows):
        return {
            "artifact_schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
            "run_id": "c2-ppo-seed-26080311",
            "algorithm": "trm_ppo",
            "training_seed": 26080311,
            "producer_git_commit": "c" * 40,
            "effective_config_sha256": "d" * 64,
            "dataset_provenance_sha256": "e" * 64,
            "checkpoint_sha256": "f" * 64,
            "checkpoint_environment_steps": 10_000,
            "checkpoint_outer_steps": 125,
            "evaluation_seed": 1729,
            "evaluation_seed_scheme": RECORD_LOCAL_SEED_SCHEME,
            "policy_mode": "greedy",
            "reward_definition": "undiscounted_sum_of_shaped_environment_rewards",
            "environment": {
                "action_count": 97,
                "plan_length": 16,
                "vocab_size": 6,
                "max_edits": 4,
                "stop_action_id": 96,
                "stop_action_mode": "disabled",
                "task_name": "sudoku",
                "undo_enabled": False,
            },
            "dataset": {
                "split": "test",
                "manifest_sha256": "0" * 64,
                "ordered_record_sha256": ordered_record_sha256(
                    [row["record_sha256"] for row in rows]
                ),
                "record_count": len(rows),
            },
        }

    def _compute_snapshot(self):
        training = zero_model_counters()
        training.update(
            {
                "policy_api_calls": 12,
                "policy_state_evaluations": 12,
                "recurrent_latent_update_calls": 48,
                "recurrent_latent_state_updates": 48,
                "action_logits_evaluated": 972,
            }
        )
        evaluation = zero_model_counters()
        evaluation.update(
            {
                "policy_api_calls": 6,
                "policy_state_evaluations": 6,
                "recurrent_latent_update_calls": 24,
                "recurrent_latent_state_updates": 24,
                "action_logits_evaluated": 486,
            }
        )
        total = {
            field: training[field] + evaluation[field]
            for field in training
        }
        optimizer_steps = {field: 0 for field in OPTIMIZER_STEP_FIELDS}
        optimizer_steps["combined"] = 4
        return {
            "compute_schema_version": COMPUTE_SNAPSHOT_SCHEMA_VERSION,
            "model_work": {
                "total": total,
                "training": training,
                "evaluation": evaluation,
                "counter_definitions": dict(MODEL_COUNTER_DEFINITIONS),
                "role_groups": [["model"]],
                "uninstrumented_roles": [],
            },
            "progress": {
                "environment_interactions": 10_000,
                "outer_updates": 125,
                "optimizer_steps_total": 4,
                "optimizer_steps_by_kind": optimizer_steps,
            },
            "wall_time_seconds": {"training": 12.5, "evaluation": 1.25},
            "peak_memory_bytes": {
                "cuda_allocated": None,
                "cuda_reserved": None,
                "process_rss": 123_456,
            },
        }

    def test_artifacts_are_deterministic_and_summary_is_row_derived(self):
        rows = self._rows()
        metadata = self._metadata(rows)
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = Path(tmp) / "first"
            second_dir = Path(tmp) / "second"
            first = write_evaluation_artifact(
                first_dir,
                metadata=metadata,
                rows=rows,
                compute_snapshot=self._compute_snapshot(),
            )
            second = write_evaluation_artifact(
                second_dir,
                metadata=metadata,
                rows=rows,
                compute_snapshot=self._compute_snapshot(),
            )

            for filename in (
                "evaluation_metadata.json",
                "per_instance.jsonl",
                "compute_snapshot.json",
                "summary.json",
            ):
                self.assertEqual(
                    (first_dir / filename).read_bytes(),
                    (second_dir / filename).read_bytes(),
                )
            summary = json.loads((first_dir / "summary.json").read_text("ascii"))
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["solved_count"], 1)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(summary["mean_final_checker_score"], 4.0)
            self.assertEqual(summary["total_environment_interactions"], 6)
            self.assertAlmostEqual(summary["invalid_action_rate"], 1.0 / 6.0)
            self.assertEqual(first["metadata_sha256"], second["metadata_sha256"])
            self.assertEqual(
                first["per_instance_sha256"], second["per_instance_sha256"]
            )
            self.assertEqual(first["summary_sha256"], second["summary_sha256"])

    def test_existing_artifact_is_never_overwritten(self):
        rows = self._rows()
        metadata = self._metadata(rows)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            write_evaluation_artifact(
                output,
                metadata=metadata,
                rows=rows,
                compute_snapshot=self._compute_snapshot(),
            )
            before = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            with self.assertRaisesRegex(EvaluationArtifactError, "overwrite"):
                write_evaluation_artifact(
                    output,
                    metadata=metadata,
                    rows=rows,
                    compute_snapshot=self._compute_snapshot(),
                )
            after = {
                path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
            }
            self.assertEqual(before, after)

    def test_wrong_record_order_is_rejected_before_publication(self):
        rows = self._rows()
        metadata = self._metadata(rows)
        metadata["dataset"]["ordered_record_sha256"] = "9" * 64
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            with self.assertRaisesRegex(EvaluationArtifactError, "record order"):
                write_evaluation_artifact(
                    output,
                    metadata=metadata,
                    rows=rows,
                    compute_snapshot=self._compute_snapshot(),
                )
            self.assertFalse(output.exists())

    def test_inconsistent_row_return_is_rejected_before_publication(self):
        rows = self._rows()
        rows[0]["undiscounted_shaped_return"] = 99.0
        metadata = self._metadata(rows)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            with self.assertRaisesRegex(EvaluationArtifactError, "reward sequence"):
                write_evaluation_artifact(
                    output,
                    metadata=metadata,
                    rows=rows,
                    compute_snapshot=self._compute_snapshot(),
                )
            self.assertFalse(output.exists())

    def test_invalid_action_and_termination_semantics_are_rejected(self):
        for mutation, message in (
            (lambda rows: rows[0]["actions"].__setitem__(0, -1), "action space"),
            (
                lambda rows: rows[1].__setitem__("termination_reason", "unknown"),
                "termination reason",
            ),
            (
                lambda rows: rows[1].__setitem__("environment_interactions", 3),
                "match environment interactions",
            ),
        ):
            with self.subTest(message=message):
                rows = self._rows()
                mutation(rows)
                metadata = self._metadata(rows)
                with tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / "artifact"
                    with self.assertRaisesRegex(EvaluationArtifactError, message):
                        write_evaluation_artifact(
                            output,
                            metadata=metadata,
                            rows=rows,
                            compute_snapshot=self._compute_snapshot(),
                        )
                    self.assertFalse(output.exists())

    def test_final_plan_hash_and_sudoku_stats_are_recomputed(self):
        for mutation, message in (
            (
                lambda rows: rows[0].__setitem__("final_plan_sha256", "a" * 64),
                "Final-plan hash",
            ),
            (
                lambda rows: rows[0]["sudoku"].__setitem__("zero_candidates", 7),
                "Sudoku statistics",
            ),
            (
                lambda rows: rows[0].__setitem__("sudoku", None),
                "require final-plan statistics",
            ),
        ):
            with self.subTest(message=message):
                rows = self._rows()
                mutation(rows)
                metadata = self._metadata(rows)
                with tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / "artifact"
                    with self.assertRaisesRegex(EvaluationArtifactError, message):
                        write_evaluation_artifact(
                            output,
                            metadata=metadata,
                            rows=rows,
                            compute_snapshot=self._compute_snapshot(),
                        )
                    self.assertFalse(output.exists())

    def test_inconsistent_compute_total_is_rejected_before_publication(self):
        rows = self._rows()
        metadata = self._metadata(rows)
        compute_snapshot = self._compute_snapshot()
        compute_snapshot["model_work"]["total"]["policy_api_calls"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            with self.assertRaisesRegex(EvaluationArtifactError, "training plus evaluation"):
                write_evaluation_artifact(
                    output,
                    metadata=metadata,
                    rows=rows,
                    compute_snapshot=compute_snapshot,
                )
            self.assertFalse(output.exists())

    def test_schema_versions_reject_bool_float_and_string_before_publication(self):
        rows = self._rows()
        for boundary, expected, mutate in (
            (
                "evaluation",
                EVALUATION_ARTIFACT_SCHEMA_VERSION,
                lambda metadata, compute, invalid: metadata.__setitem__(
                    "artifact_schema_version", invalid
                ),
            ),
            (
                "compute",
                COMPUTE_SNAPSHOT_SCHEMA_VERSION,
                lambda metadata, compute, invalid: compute.__setitem__(
                    "compute_schema_version", invalid
                ),
            ),
        ):
            for invalid in (True, float(expected), str(expected)):
                with self.subTest(boundary=boundary, invalid=invalid):
                    metadata = self._metadata(rows)
                    compute_snapshot = self._compute_snapshot()
                    mutate(metadata, compute_snapshot, invalid)
                    with tempfile.TemporaryDirectory() as tmp:
                        output = Path(tmp) / "artifact"
                        with self.assertRaises(EvaluationArtifactError):
                            write_evaluation_artifact(
                                output,
                                metadata=metadata,
                                rows=rows,
                                compute_snapshot=compute_snapshot,
                            )
                        self.assertFalse(output.exists())

    def test_compute_progress_must_match_checkpoint_metadata(self):
        rows = self._rows()
        metadata = self._metadata(rows)
        compute_snapshot = self._compute_snapshot()
        compute_snapshot["progress"]["environment_interactions"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            with self.assertRaisesRegex(EvaluationArtifactError, "checkpoint"):
                write_evaluation_artifact(
                    output,
                    metadata=metadata,
                    rows=rows,
                    compute_snapshot=compute_snapshot,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
