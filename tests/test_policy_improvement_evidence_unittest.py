#!/usr/bin/env fbpython
"""Filesystem-bound tests for policy-improvement evidence publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from scripts.policy_improvement_evidence import authenticate_complete_generation
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
)
from scripts.policy_improvement_test_open import (
    _authenticate_record,
    _publish_record,
)


HEX64 = "a" * 64


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _write_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class CompleteGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "evidence"
        self.root.mkdir(mode=0o700)
        self.project_root = Path(self.temporary.name) / "project"
        self.project_root.mkdir()
        self.dataset_root = Path(self.temporary.name) / "dataset"
        self.dataset_root.mkdir()
        self.run_id = "s0-fixed-base-s1"
        self.protocol_sha = "1" * 64
        self.registry_sha = "2" * 64
        self.row_sha = "3" * 64
        self.protocol = {
            "methods": [
                {
                    "id": "matched_ppo",
                    "config_sha256": "4" * 64,
                }
            ],
            "dataset": {"manifest_sha256": {"status": "available", "value": "8" * 64}},
        }
        self.row = {
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "base_method_id": "matched_ppo",
            "seed": 1,
            "expected_effective_config_sha256": "5" * 64,
        }
        self.training_execution_identity = {
            "role": "policy-improvement-training",
            "source_git_commit": "b" * 40,
            "runtime_sha256": "c" * 64,
            "runtime_profile_sha256": "d" * 64,
            "selected_source_manifest_sha256": "e" * 64,
            "runtime_authorization_sha256": "f" * 64,
            "launcher_sha256": "0" * 64,
        }
        self.generation = (
            self.root / "runs" / self.run_id / "segments" / "env_000000032"
        )
        (self.generation / "checkpoints").mkdir(parents=True)
        (self.generation / "evaluations").mkdir()
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint bytes")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        parent_generation = (
            self.root / "runs" / self.run_id / "segments" / "env_000000016"
        )
        (parent_generation / "checkpoints").mkdir(parents=True)
        parent_checkpoint = parent_generation / "checkpoints" / "checkpoint.pt"
        parent_checkpoint.write_bytes(b"parent checkpoint bytes")
        parent_checkpoint_sha = hashlib.sha256(
            parent_checkpoint.read_bytes()
        ).hexdigest()
        _write_json(
            parent_generation / "checkpoint_validation.json",
            {
                "schema_name": "policy_improvement_checkpoint_validation_v1",
                "schema_version": 1,
                "validator": "policy_improvement_smoke_runtime",
                "validator_execution_identity": dict(self.training_execution_identity),
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "snapshot_kind": "interaction_matched",
                "environment_interactions": 16,
                "checkpoint_sha256": parent_checkpoint_sha,
                "parent_checkpoint_sha256": None,
                "model_state_sha256": hashlib.sha256(
                    canonical_json_bytes({"model": "9" * 64})
                ).hexdigest(),
                "role_state_sha256s": {"model": "9" * 64},
                "strict_resume_validated": True,
            },
        )
        parent_files = {
            "checkpoints/checkpoint.pt": _identity(parent_checkpoint),
            "checkpoint_validation.json": _identity(
                parent_generation / "checkpoint_validation.json"
            ),
        }
        _write_json(
            parent_generation / "MANIFEST.json",
            {
                "schema_name": "policy_improvement_smoke_segment_v1",
                "schema_version": 1,
                "protocol_sha256": self.protocol_sha,
                "registry_sha256": self.registry_sha,
                "registry_row_sha256": self.row_sha,
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "segment": "prepare",
                "environment_interactions": 16,
                "parent_checkpoint_sha256": None,
                "result_status": None,
                "outputs": {
                    "checkpoint": {
                        "path": "checkpoints/checkpoint.pt",
                        **parent_files["checkpoints/checkpoint.pt"],
                    },
                    "files": parent_files,
                },
                "storage_bytes": sum(
                    int(item["bytes"]) for item in parent_files.values()
                ),
            },
        )
        model_roles = {"model": "7" * 64}
        model_sha = hashlib.sha256(canonical_json_bytes(model_roles)).hexdigest()
        initialization_sha = "6" * 64
        checkpoint_lineage_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_name": "policy_improvement_smoke_checkpoint_lineage_v1",
                    "run_id": self.run_id,
                    "initialization_sha256": initialization_sha,
                    "parent_checkpoint_sha256": parent_checkpoint_sha,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
        ).hexdigest()
        checkpoint_validation_digest = _write_json(
            self.generation / "checkpoint_validation.json",
            {
                "schema_name": "policy_improvement_checkpoint_validation_v1",
                "schema_version": 1,
                "validator": "policy_improvement_smoke_runtime",
                "validator_execution_identity": dict(self.training_execution_identity),
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "snapshot_kind": "interaction_matched",
                "environment_interactions": 32,
                "checkpoint_sha256": checkpoint_sha,
                "parent_checkpoint_sha256": parent_checkpoint_sha,
                "model_state_sha256": model_sha,
                "role_state_sha256s": model_roles,
                "strict_resume_validated": True,
            },
        )
        model_inventory_digest = _write_json(
            self.generation / "model_state_inventory.json",
            {
                "schema_name": "policy_improvement_model_state_inventory_v1",
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "model_state_sha256": model_sha,
                "role_state_sha256s": model_roles,
                "snapshot_state_bindings": [
                    {
                        "snapshot_kind": "interaction_matched",
                        "checkpoint_sha256": checkpoint_sha,
                        "model_state_sha256": model_sha,
                        "checkpoint_validation_sha256": (checkpoint_validation_digest),
                    }
                ],
            },
        )
        self.per_instance = {
            "schema_name": "fixture_per_instance",
            "records": [{"registered_index": 0, "solved": True}],
        }
        per_digest = hashlib.sha256(canonical_json_bytes(self.per_instance)).hexdigest()
        primary = {
            "solve_rate": _available(1.0),
            "solved_count": _available(1),
            "denominator": _available(1),
        }
        secondary = {
            "discounted_return_mean": _available(1.0),
            "edits_to_solve_mean": _available(1.0),
            "terminal_reason_counts": _available({"solved": 1}),
            "value_calibration": _available(0.0),
        }
        aggregate_semantic = {"primary": primary, "secondary": secondary}
        aggregate_digest = hashlib.sha256(
            canonical_json_bytes(aggregate_semantic)
        ).hexdigest()
        _write_json(
            self.generation / "evaluations" / "realized_policy.json",
            {**aggregate_semantic, "diagnostic_details": {}},
        )
        _write_json(
            self.generation / "evaluations" / "realized_policy.per_instance.json",
            self.per_instance,
        )
        run_manifest = {
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "environment_interactions": 32,
            "protocol_sha256": self.protocol_sha,
            "registry_row_sha256": self.row_sha,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": model_sha,
            "model_state_sha256s": model_roles,
            "model_state_inventory_sha256": model_inventory_digest,
        }
        run_manifest_digest = _write_json(
            self.generation / "RUN_MANIFEST.json", run_manifest
        )
        self.result = {
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "tier": "smoke",
            "seed": 1,
            "artifacts": {
                "checkpoint": _available(checkpoint_sha),
                "checkpoint_validation": _available(checkpoint_validation_digest),
                "model_state_inventory": _available(model_inventory_digest),
                "run_manifest": _available(run_manifest_digest),
            },
            "identities": {
                "checkpoint_sha256": _available(checkpoint_sha),
                "model_state_sha256": _available(model_sha),
                "initialization_sha256": initialization_sha,
                "training_source_git_commit": self.training_execution_identity[
                    "source_git_commit"
                ],
                "training_runtime_sha256": self.training_execution_identity[
                    "runtime_sha256"
                ],
                "training_runtime_profile_sha256": self.training_execution_identity[
                    "runtime_profile_sha256"
                ],
                "training_selected_source_manifest_sha256": (
                    self.training_execution_identity["selected_source_manifest_sha256"]
                ),
                "runtime_authorization_sha256": self.training_execution_identity[
                    "runtime_authorization_sha256"
                ],
                "launcher_sha256": self.training_execution_identity["launcher_sha256"],
                "method_config_sha256": "4" * 64,
                "effective_config_sha256": "5" * 64,
                "dataset_manifest_sha256": "8" * 64,
            },
            "evaluation_snapshots": [
                {
                    "snapshot_kind": "interaction_matched",
                    "status": "available",
                    "observed_environment_interactions": _available(32),
                    "checkpoint_sha256": _available(checkpoint_sha),
                    "model_state_sha256": _available(model_sha),
                    "checkpoint_lineage_sha256": _available(checkpoint_lineage_sha),
                    "policy_evaluations": [
                        {
                            "policy_variant": "realized_policy",
                            "evaluation_artifact_sha256": _available(aggregate_digest),
                            "per_instance_artifact_sha256": _available(per_digest),
                            "primary": primary,
                            "secondary": secondary,
                        }
                    ],
                }
            ],
        }
        _write_json(self.generation / "result.json", self.result)
        files: dict[str, dict[str, object]] = {}
        for path in sorted(self.generation.rglob("*")):
            if path.is_file():
                files[path.relative_to(self.generation).as_posix()] = _identity(path)
        manifest = {
            "schema_name": "policy_improvement_smoke_segment_v1",
            "schema_version": 1,
            "protocol_sha256": self.protocol_sha,
            "registry_sha256": self.registry_sha,
            "registry_row_sha256": self.row_sha,
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "segment": "resume",
            "environment_interactions": 32,
            "parent_checkpoint_sha256": parent_checkpoint_sha,
            "result_status": "complete",
            "outputs": {
                "checkpoint": {
                    "path": "checkpoints/checkpoint.pt",
                    **files["checkpoints/checkpoint.pt"],
                },
                "files": files,
            },
            "storage_bytes": sum(int(item["bytes"]) for item in files.values()),
        }
        _write_json(self.generation / "MANIFEST.json", manifest)
        self.documents = {per_digest: self.per_instance}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _authenticate(self) -> dict[str, object]:
        return authenticate_complete_generation(
            evidence_root=self.root,
            result_path=self.generation / "result.json",
            result=self.result,
            protocol_sha256=self.protocol_sha,
            registry_sha256=self.registry_sha,
            registry_row_sha256=self.row_sha,
            expected_environment_interactions=32,
            per_instance_documents=self.documents,
            checkpoint_validator=self._checkpoint_validator,
            protocol=self.protocol,
            registry_row=self.row,
            project_root=self.project_root,
            dataset_root=self.dataset_root,
            runtime_authorization={},
        )

    def _checkpoint_validator(self, request: dict[str, object]) -> dict[str, object]:
        checkpoint = Path(str(request["checkpoint_path"]))
        receipt = next(
            value
            for value in (
                json.loads(path.read_text(encoding="ascii"))
                for path in checkpoint.parents[1].rglob("checkpoint_validation.json")
            )
            if value["checkpoint_sha256"] == request["checkpoint_sha256"]
        )
        return {
            "schema_name": "policy_improvement_checkpoint_semantic_validation_v1",
            "schema_version": 1,
            "run_id": request["run_id"],
            "method_id": request["method_id"],
            "seed": request["seed"],
            "snapshot_kind": request["snapshot_kind"],
            "environment_interactions": request["environment_interactions"],
            "parent_checkpoint_sha256": request["parent_checkpoint_sha256"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "initialization_sha256": request["initialization_sha256"],
            "model_state_sha256": receipt["model_state_sha256"],
            "role_state_sha256s": receipt["role_state_sha256s"],
            "method_config_sha256": "4" * 64,
            "registered_effective_config_sha256": "5" * 64,
            "effective_config_sha256": "a" * 64,
            "dataset_manifest_sha256": "8" * 64,
            "dataset_provenance_sha256": "9" * 64,
            "run_identity_sha256": None,
            "training_call_delta": 0,
            "evaluation_call_delta": 0,
            "optimizer_step_delta": 0,
        }

    def test_complete_generation_reopens_every_artifact(self) -> None:
        observed = self._authenticate()
        self.assertEqual(
            observed["checkpoint_sha256"],
            self.result["artifacts"]["checkpoint"]["value"],
        )

    def test_detached_result_is_rejected(self) -> None:
        detached = self.root / "detached.json"
        _write_json(detached, self.result)
        with self.assertRaises(PolicyImprovementSchemaError):
            authenticate_complete_generation(
                evidence_root=self.root,
                result_path=detached,
                result=self.result,
                protocol_sha256=self.protocol_sha,
                registry_sha256=self.registry_sha,
                registry_row_sha256=self.row_sha,
                expected_environment_interactions=32,
                per_instance_documents=self.documents,
                checkpoint_validator=self._checkpoint_validator,
                protocol=self.protocol,
                registry_row=self.row,
                project_root=self.project_root,
                dataset_root=self.dataset_root,
                runtime_authorization={},
            )

    def test_tampered_checkpoint_is_rejected(self) -> None:
        (self.generation / "checkpoints" / "checkpoint.pt").write_bytes(b"tampered")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()

    def test_tampered_parent_checkpoint_is_rejected(self) -> None:
        parent = (
            self.root
            / "runs"
            / self.run_id
            / "segments"
            / "env_000000016"
            / "checkpoints"
            / "checkpoint.pt"
        )
        parent.write_bytes(b"tampered parent")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()

    def test_extra_file_is_rejected(self) -> None:
        (self.generation / "extra").write_bytes(b"extra")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()

    def test_symlink_and_hardlink_are_rejected(self) -> None:
        target = self.generation / "RUN_MANIFEST.json"
        alias = self.generation / "alias"
        os.link(target, alias)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()
        alias.unlink()
        alias.symlink_to(target)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()


class TestOpenPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "test-open"
        self.root.mkdir(mode=0o700)
        self.record = {
            "schema_name": "policy_improvement_test_open_v1",
            "schema_version": 3,
            "record_id": "confirmatory-test-open-v1",
            "state": "opened_immutable",
            "protocol_sha256": "1" * 64,
            "registry_sha256": "2" * 64,
            "amendment_history_sha256": "3" * 64,
            "runtime_authorization_sha256": "4" * 64,
            "test_manifest_sha256": "5" * 64,
            "prior_open_record_sha256": "6" * 64,
            "open_ordinal": 1,
            "authorized_phases": ["stage2_confirmatory", "stage3_ablation"],
            "final_selection_created_at_utc": "2026-08-14T12:00:00Z",
            "opened_at_utc": "2026-08-14T12:01:00Z",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_publish_once_and_authenticate(self) -> None:
        identity = _publish_record(owner_root=self.root, record=self.record)
        self.assertEqual(
            _authenticate_record(
                owner_root=self.root,
                expected_record=self.record,
                expected_sha256=identity["sha256"],
            ),
            identity,
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            _publish_record(owner_root=self.root, record=self.record)

    def test_concurrent_publish_has_one_winner(self) -> None:
        outcomes: list[str] = []

        def publish() -> None:
            try:
                _publish_record(owner_root=self.root, record=self.record)
                outcomes.append("published")
            except PolicyImprovementSchemaError:
                outcomes.append("rejected")

        threads = [threading.Thread(target=publish) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(outcomes), ["published", "rejected"])
        self.assertEqual(
            [path.name for path in self.root.iterdir()], ["TEST_OPEN.json"]
        )

    def test_tamper_and_symlink_are_rejected(self) -> None:
        identity = _publish_record(owner_root=self.root, record=self.record)
        path = self.root / "TEST_OPEN.json"
        path.write_bytes(b"tampered")
        with self.assertRaises(PolicyImprovementSchemaError):
            _authenticate_record(
                owner_root=self.root,
                expected_record=self.record,
                expected_sha256=identity["sha256"],
            )
        path.unlink()
        outside = Path(self.temporary.name) / "outside"
        outside.write_bytes(canonical_json_bytes(self.record) + b"\n")
        path.symlink_to(outside)
        with self.assertRaises(PolicyImprovementSchemaError):
            _authenticate_record(
                owner_root=self.root,
                expected_record=self.record,
                expected_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
