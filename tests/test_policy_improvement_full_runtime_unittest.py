#!/usr/bin/env fbpython
"""Unit tests for the fail-closed full policy-improvement runtime core."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.policy_improvement_full_runtime import (
    BackendPackage,
    BackendRequest,
    execute_registered_run,
    execution_contract,
    FullRunBackend,
    FullRunFailure,
    FullRuntimeError,
    load_registered_full_run,
    RegisteredFullRun,
)
from scripts.policy_improvement_schema import canonical_json_bytes


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


class _Backend(FullRunBackend):
    def __init__(self, *, symlink: bool = False, bad_compute: bool = False) -> None:
        self.request: BackendRequest | None = None
        self.symlink = symlink
        self.bad_compute = bad_compute

    def execute(self, request: BackendRequest) -> BackendPackage:
        self.request = request
        generation = request.staging_generation
        checkpoint = generation / "checkpoints/interaction.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"strict checkpoint")
        checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.symlink:
            (generation / "bad-link").symlink_to(checkpoint)
        for kind in ("interaction", "compute"):
            validation = generation / f"validations/{kind}/checkpoint_validation.json"
            _write_json(validation, {"strict_resume_validated": True})
        _write_json(generation / "RUN_MANIFEST.json", {"complete": True})
        _write_json(generation / "model_state_inventory.json", {"complete": True})
        run = request.run
        observed_compute = (
            request.compute_target_recurrent_map_applications * 2
            if self.bad_compute
            else request.compute_target_recurrent_map_applications
        )
        result: dict[str, Any] = {
            **{
                field: run.row[field]
                for field in (
                    "run_id",
                    "phase",
                    "tier",
                    "seed",
                    "evaluation_split",
                    "method_id",
                    "base_method_id",
                    "n",
                    "K",
                    "alpha",
                    "ablation_variant",
                )
            },
            "status": "complete",
            "protocol_sha256": run.protocol_sha256,
            "registry_row_sha256": run.registry_row_sha256,
            "amendment_history_sha256": run.amendment_history_sha256,
            "applied_config_override": run.row["config_override"],
            "evaluation_snapshots": [
                {
                    "snapshot_kind": "interaction_matched",
                    "status": "available",
                    "target": {
                        "unit": "environment_interactions",
                        "registered_quantity": _available(
                            run.final_environment_interactions
                        ),
                    },
                    "observed_environment_interactions": _available(
                        run.final_environment_interactions
                    ),
                },
                {
                    "snapshot_kind": "compute_matched",
                    "status": "available",
                    "target": {
                        "unit": "recurrent_map_applications",
                        "registered_quantity": _available(
                            request.compute_target_recurrent_map_applications
                        ),
                    },
                    "observed_recurrent_map_applications": _available(observed_compute),
                },
            ],
            "identities": {
                "checkpoint_sha256": _available(checkpoint_digest),
                "runtime_authorization_sha256": run.runtime_authorization_sha256,
                "test_open_sha256": {
                    "status": "unavailable",
                    "reason": "test_data_not_opened",
                },
            },
            "artifacts": {"checkpoint": _available(checkpoint_digest)},
        }
        _write_json(generation / "result.json", result)
        return BackendPackage(
            result=result,
            primary_checkpoint_relative_path="checkpoints/interaction.pt",
        )


class _FailureBackend(FullRunBackend):
    def execute(self, request: BackendRequest) -> BackendPackage:
        run = request.run
        result = {
            "status": "failed",
            "run_id": run.row["run_id"],
            "protocol_sha256": run.protocol_sha256,
            "registry_row_sha256": run.registry_row_sha256,
            "failure": {"phase": "training"},
            "identities": {
                "runtime_authorization_sha256": run.runtime_authorization_sha256
            },
        }
        raise FullRunFailure("training", result)


class FullRuntimeRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.project = base / "project"
        self.protocol_path = (
            self.project / "configs/policy_improvement_v1/protocol.json"
        )
        self.registry_path = (
            self.project / "configs/policy_improvement_v1/registry.json"
        )
        self.amendment_path = base / "compute.json"
        self.evidence = base / "evidence"
        self.evidence.mkdir(mode=0o700)
        _write_json(self.protocol_path, {"fixture": "protocol"})
        _write_json(self.registry_path, {"fixture": "registry"})
        _write_json(self.amendment_path, {"fixture": "compute"})
        self.protocol = {
            "full_execution_gate": {
                "environment_variable": "RUN_UPITRM_FULL_EXPERIMENTS",
                "required_value": "1",
            },
            "output_root": {"relative_path": "policy_improvement_v1"},
            "budgets": {
                "pilot": {
                    "checkpoint_environment_interactions": [10, 20, 40, 80],
                    "environment_interactions": 80,
                    "evaluation_records": 256,
                },
                "confirmatory": {
                    "checkpoint_environment_interactions": [10, 20, 40, 80],
                    "environment_interactions": 80,
                    "evaluation_records": 512,
                },
            },
        }
        self.history = [
            {
                "common_compute_targets": {
                    "unit": "recurrent_map_applications",
                    "pilot": 900,
                    "confirmatory": 1200,
                    "ablation": 1300,
                }
            }
        ]

    def _row(self, method: str) -> dict[str, object]:
        return {
            "row_kind": "concrete",
            "run_id": f"s1-{method}-n2-k1-s7",
            "phase": "stage1_screen",
            "tier": "pilot",
            "seed": 7,
            "evaluation_split": "validation",
            "method_id": method,
            "base_method_id": method,
            "n": 2,
            "K": 1,
            "alpha": 0.1,
            "ablation_variant": None,
            "config_override": {"inner_unroll_n": 2},
        }

    def _load(self, row: dict[str, object]) -> RegisteredFullRun:
        registry = {"rows": [row]}
        with (
            mock.patch(
                "scripts.policy_improvement_full_runtime.validate_protocol",
                return_value=self.protocol,
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.validate_amendment_history",
                return_value=self.history,
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.load_registered_base_configs",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.validate_registry_document",
                return_value=registry,
            ),
        ):
            return load_registered_full_run(
                project_root=self.project,
                protocol_path=self.protocol_path,
                registry_path=self.registry_path,
                amendment_paths=[self.amendment_path],
                evidence_root=self.evidence,
                row_id=str(row["run_id"]),
                runtime_authorization_sha256=_digest("runtime authorization"),
                environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
            )

    def test_all_four_primary_methods_get_exact_schedules(self) -> None:
        for method in (
            "fixed_base_exact_persistent",
            "fixed_base_exact_episodic",
            "legacy_parameter_interpolation",
            "matched_ppo",
        ):
            with self.subTest(method=method):
                run = self._load(self._row(method))
                self.assertEqual(run.interaction_checkpoints, (10, 20, 40, 80))
                self.assertEqual(run.final_environment_interactions, 80)
                self.assertEqual(
                    run.compute_target_recurrent_map_applications,
                    900,
                )
                self.assertEqual(run.evaluation_records, 256)
                self.assertIsNone(run.test_open_sha256)

    def test_full_flag_is_mandatory(self) -> None:
        with self.assertRaisesRegex(FullRuntimeError, "RUN_UPITRM_FULL_EXPERIMENTS"):
            load_registered_full_run(
                project_root=self.project,
                protocol_path=self.protocol_path,
                registry_path=self.registry_path,
                amendment_paths=[self.amendment_path],
                evidence_root=self.evidence,
                row_id="blocked",
                runtime_authorization_sha256=_digest("runtime authorization"),
                environment={},
            )

    def test_selection_template_and_validation_test_mismatch_fail_closed(self) -> None:
        template = self._row("matched_ppo")
        template["row_kind"] = "selection_template"
        with self.assertRaisesRegex(FullRuntimeError, "Selection-dependent"):
            self._load(template)
        mismatch = self._row("matched_ppo")
        mismatch["evaluation_split"] = "test"
        with self.assertRaisesRegex(FullRuntimeError, "Pilot execution"):
            self._load(mismatch)


class FullRuntimePublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        project = base / "project"
        project.mkdir()
        evidence = base / "evidence"
        evidence.mkdir(mode=0o700)
        row = {
            "run_id": "s1-fixed_base_exact_persistent-n2-k1-s7",
            "phase": "stage1_screen",
            "tier": "pilot",
            "seed": 7,
            "evaluation_split": "validation",
            "method_id": "fixed_base_exact_persistent",
            "base_method_id": "fixed_base_exact_persistent",
            "n": 2,
            "K": 1,
            "alpha": 0.1,
            "ablation_variant": None,
            "config_override": {"inner_unroll_n": 2},
        }
        self.run = RegisteredFullRun(
            project_root=project,
            protocol_path=project / "protocol.json",
            registry_path=project / "registry.json",
            evidence_root=evidence,
            protocol={"output_root": {"relative_path": "policy_improvement_v1"}},
            registry={"rows": [row]},
            amendment_history=({},),
            row=row,
            protocol_sha256=_digest("protocol"),
            registry_sha256=_digest("registry"),
            amendment_history_sha256=_digest("history"),
            registry_row_sha256=_digest("row"),
            runtime_authorization_sha256=_digest("runtime authorization"),
            interaction_checkpoints=(10, 20, 40, 80),
            final_environment_interactions=80,
            compute_target_recurrent_map_applications=1000,
            evaluation_records=256,
            test_open_sha256=None,
        )

    def test_complete_run_publishes_once_with_exact_backend_contract(self) -> None:
        backend = _Backend()
        final = execute_registered_run(
            self.run,
            backend=backend,
            result_validator=lambda value: value,
        )
        self.assertTrue((final / "segments/env_000000080/MANIFEST.json").is_file())
        self.assertIsNotNone(backend.request)
        assert backend.request is not None
        self.assertEqual(backend.request.interaction_checkpoints, (10, 20, 40, 80))
        self.assertEqual(
            backend.request.compute_target_recurrent_map_applications, 1000
        )
        self.assertTrue(backend.request.require_strict_checkpoint_resume)
        self.assertTrue(backend.request.require_interaction_and_compute_snapshots)
        with self.assertRaisesRegex(FullRuntimeError, "already exists"):
            execute_registered_run(
                self.run,
                backend=_Backend(),
                result_validator=lambda value: value,
            )
        runs = final.parent
        self.assertEqual(
            [path.name for path in runs.iterdir()],
            [str(self.run.row["run_id"])],
        )

    def test_missing_backend_bad_compute_and_symlink_fail_without_leak(self) -> None:
        with self.assertRaisesRegex(FullRuntimeError, "No sealed"):
            execute_registered_run(self.run, backend=None)
        for backend, message in (
            (_Backend(bad_compute=True), "Compute-matched"),
            (_Backend(symlink=True), "symlink"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(FullRuntimeError, message):
                    execute_registered_run(
                        self.run,
                        backend=backend,
                        result_validator=lambda value: value,
                    )
                runs = self.run.evidence_root / "policy_improvement_v1" / "runs"
                self.assertFalse(any(runs.iterdir()))

    def test_failed_backend_publishes_one_immutable_attempt(self) -> None:
        final = execute_registered_run(
            self.run,
            backend=_FailureBackend(),
            result_validator=lambda value: value,
        )
        attempts = list((final / "attempts/complete").iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            {path.name for path in attempts[0].iterdir()},
            {"MANIFEST.json", "result.json"},
        )
        self.assertFalse((final / "segments").exists())

    def test_contract_reports_non_executable_integration_hooks(self) -> None:
        contract = execution_contract(self.run)
        self.assertFalse(contract["execution_ready"])
        self.assertEqual(contract["interaction_checkpoints"], [10, 20, 40, 80])
        self.assertEqual(len(contract["blocked_by"]), 3)


if __name__ == "__main__":
    unittest.main()
