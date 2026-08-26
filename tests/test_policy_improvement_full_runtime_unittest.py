#!/usr/bin/env fbpython
"""Unit tests for the fail-closed full policy-improvement runtime core."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.policy_improvement_full_runtime import (
    _full_producer_runtime_role,
    AuthenticatedFullCheckpoint,
    BackendPackage,
    BackendRequest,
    execute_registered_run,
    execution_contract,
    FullRunBackend,
    FullRunFailure,
    FullRuntimeError,
    load_registered_full_run,
    RegisteredFullRun,
    PublishedFullRunFailure,
    resolve_authenticated_full_checkpoint,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    runtime_authorization_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


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
        for kind in ("interaction_matched", "compute_matched"):
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


class _ResolverBackend(FullRunBackend):
    def __init__(self, runtime_authorization: dict[str, object]) -> None:
        self.runtime_authorization = runtime_authorization

    def execute(self, request: BackendRequest) -> BackendPackage:
        generation = request.staging_generation
        run = request.run
        authorization = self.runtime_authorization
        roles = authorization["roles"]
        assert isinstance(roles, list)
        training = roles[0]
        assert isinstance(training, dict)
        execution_identity = {
            "role": "policy-improvement-training",
            "source_git_commit": training["source_git_commit"],
            "runtime_sha256": training["runtime_sha256"],
            "runtime_profile_sha256": training["runtime_profile_sha256"],
            "selected_source_manifest_sha256": training[
                "selected_source_manifest_sha256"
            ],
            "runtime_authorization_sha256": run.runtime_authorization_sha256,
            "launcher_sha256": authorization["launcher_sha256"],
        }
        nodes = [
            (
                "scheduled",
                10,
                "checkpoints/scheduled/env_000000010/rl_checkpoint_step_10.pt",
                "validations/scheduled/env_000000010/resume_validation.json",
            ),
            (
                "scheduled",
                20,
                "checkpoints/scheduled/env_000000020/rl_checkpoint_step_20.pt",
                "validations/scheduled/env_000000020/resume_validation.json",
            ),
            (
                "compute_matched",
                30,
                "checkpoints/compute_matched/rl_checkpoint_step_30.pt",
                "validations/compute_matched/checkpoint_validation.json",
            ),
            (
                "scheduled",
                40,
                "checkpoints/scheduled/env_000000040/rl_checkpoint_step_40.pt",
                "validations/scheduled/env_000000040/resume_validation.json",
            ),
            (
                "interaction_matched",
                80,
                "checkpoints/interaction_matched/rl_checkpoint_step_80.pt",
                "validations/interaction_matched/checkpoint_validation.json",
            ),
        ]
        captured: dict[tuple[str, int], dict[str, object]] = {}
        parent = None
        for kind, interactions, checkpoint_name, validation_name in nodes:
            checkpoint = generation / checkpoint_name
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{kind}:{interactions}".encode("ascii"))
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            role_hashes = {"model": _digest(f"{kind}:{interactions}:model")}
            model_sha256 = hashlib.sha256(canonical_json_bytes(role_hashes)).hexdigest()
            validation = {
                "schema_name": "policy_improvement_checkpoint_validation_v1",
                "schema_version": 1,
                "validator": "policy_improvement_full_runtime",
                "validator_execution_identity": execution_identity,
                "run_id": run.row["run_id"],
                "method_id": run.row["method_id"],
                "snapshot_kind": kind,
                "environment_interactions": interactions,
                "checkpoint_sha256": checkpoint_sha256,
                "parent_checkpoint_sha256": parent,
                "model_state_sha256": model_sha256,
                "role_state_sha256s": role_hashes,
                "strict_resume_validated": True,
            }
            validation_sha256 = _write_json(generation / validation_name, validation)
            captured[(kind, interactions)] = {
                "checkpoint_name": checkpoint_name,
                "checkpoint_sha256": checkpoint_sha256,
                "validation_sha256": validation_sha256,
                "model_state_sha256": model_sha256,
                "role_state_sha256s": role_hashes,
                "parent_checkpoint_sha256": parent,
            }
            parent = checkpoint_sha256

        interaction = captured[("interaction_matched", 80)]
        compute = captured[("compute_matched", 30)]
        model_inventory = {
            "schema_name": "policy_improvement_model_state_inventory_v1",
            "run_id": run.row["run_id"],
            "method_id": run.row["method_id"],
            "model_state_sha256": interaction["model_state_sha256"],
            "role_state_sha256s": interaction["role_state_sha256s"],
            "snapshot_state_bindings": [
                {
                    "snapshot_kind": kind,
                    "checkpoint_sha256": captured[(kind, interactions)][
                        "checkpoint_sha256"
                    ],
                    "model_state_sha256": captured[(kind, interactions)][
                        "model_state_sha256"
                    ],
                    "checkpoint_validation_sha256": captured[(kind, interactions)][
                        "validation_sha256"
                    ],
                }
                for kind, interactions in (
                    ("interaction_matched", 80),
                    ("compute_matched", 30),
                )
            ],
        }
        model_inventory_sha256 = _write_json(
            generation / "model_state_inventory.json", model_inventory
        )
        run_manifest = {
            "schema_name": "policy_improvement_full_run_manifest_v1",
            "schema_version": 1,
            "run_id": run.row["run_id"],
            "method_id": run.row["method_id"],
            "environment_interactions": 80,
            "protocol_sha256": run.protocol_sha256,
            "registry_row_sha256": run.registry_row_sha256,
            "amendment_history_sha256": run.amendment_history_sha256,
            "runtime_authorization_sha256": run.runtime_authorization_sha256,
            "runtime_sha256": training["runtime_sha256"],
            "source_git_commit": training["source_git_commit"],
            "source_manifest_sha256": training["selected_source_manifest_sha256"],
            "dataset_manifest_sha256": _digest("dataset"),
            "checkpoint_sha256": interaction["checkpoint_sha256"],
            "model_state_sha256": interaction["model_state_sha256"],
            "model_state_sha256s": interaction["role_state_sha256s"],
            "model_state_inventory_sha256": model_inventory_sha256,
            "checkpoint_schedule": [
                {
                    "environment_interactions": interactions,
                    "checkpoint_sha256": captured[("scheduled", interactions)][
                        "checkpoint_sha256"
                    ],
                    "resume_validation_sha256": captured[("scheduled", interactions)][
                        "validation_sha256"
                    ],
                }
                for interactions in (10, 20, 40)
            ],
            "snapshots": [
                {
                    "snapshot_kind": kind,
                    "checkpoint_path": captured[(kind, interactions)][
                        "checkpoint_name"
                    ],
                    "checkpoint_sha256": captured[(kind, interactions)][
                        "checkpoint_sha256"
                    ],
                    "environment_interactions": interactions,
                    "recurrent_map_applications": 1000,
                }
                for kind, interactions in (
                    ("interaction_matched", 80),
                    ("compute_matched", 30),
                )
            ],
        }
        run_manifest_sha256 = _write_json(
            generation / "RUN_MANIFEST.json", run_manifest
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
                        "registered_quantity": _available(80),
                    },
                    "observed_environment_interactions": _available(80),
                    "observed_recurrent_map_applications": _available(1000),
                    "checkpoint_sha256": _available(interaction["checkpoint_sha256"]),
                    "model_state_sha256": _available(interaction["model_state_sha256"]),
                    "checkpoint_lineage_sha256": _available(
                        hashlib.sha256(
                            canonical_json_bytes(
                                {
                                    "schema_name": (
                                        "policy_improvement_full_checkpoint_lineage_v1"
                                    ),
                                    "run_id": run.row["run_id"],
                                    "snapshot_kind": "interaction_matched",
                                    "parent_checkpoint_sha256": interaction[
                                        "parent_checkpoint_sha256"
                                    ],
                                    "checkpoint_sha256": interaction[
                                        "checkpoint_sha256"
                                    ],
                                    "environment_interactions": 80,
                                    "recurrent_map_applications": 1000,
                                }
                            )
                        ).hexdigest()
                    ),
                },
                {
                    "snapshot_kind": "compute_matched",
                    "status": "available",
                    "target": {
                        "unit": "recurrent_map_applications",
                        "registered_quantity": _available(1000),
                    },
                    "observed_environment_interactions": _available(30),
                    "observed_recurrent_map_applications": _available(1000),
                    "checkpoint_sha256": _available(compute["checkpoint_sha256"]),
                    "model_state_sha256": _available(compute["model_state_sha256"]),
                    "checkpoint_lineage_sha256": _available(
                        hashlib.sha256(
                            canonical_json_bytes(
                                {
                                    "schema_name": (
                                        "policy_improvement_full_checkpoint_lineage_v1"
                                    ),
                                    "run_id": run.row["run_id"],
                                    "snapshot_kind": "compute_matched",
                                    "parent_checkpoint_sha256": compute[
                                        "parent_checkpoint_sha256"
                                    ],
                                    "checkpoint_sha256": compute["checkpoint_sha256"],
                                    "environment_interactions": 30,
                                    "recurrent_map_applications": 1000,
                                }
                            )
                        ).hexdigest()
                    ),
                },
            ],
            "identities": {
                "producer_git_commit": authorization["producer_git_commit"],
                "producer_manifest_sha256": authorization[
                    "producer_source_manifest_sha256"
                ],
                "runtime_authorization_sha256": run.runtime_authorization_sha256,
                "training_source_git_commit": training["source_git_commit"],
                "training_runtime_sha256": training["runtime_sha256"],
                "training_runtime_profile_sha256": training["runtime_profile_sha256"],
                "training_selected_source_manifest_sha256": training[
                    "selected_source_manifest_sha256"
                ],
                "launcher_sha256": authorization["launcher_sha256"],
                "dataset_manifest_sha256": _digest("dataset"),
                "checkpoint_sha256": _available(interaction["checkpoint_sha256"]),
                "model_state_sha256": _available(interaction["model_state_sha256"]),
                "test_open_sha256": {
                    "status": "unavailable",
                    "reason": "test_data_not_opened",
                },
            },
            "artifacts": {
                "checkpoint": _available(interaction["checkpoint_sha256"]),
                "checkpoint_validation": _available(interaction["validation_sha256"]),
                "model_state_inventory": _available(model_inventory_sha256),
                "run_manifest": _available(run_manifest_sha256),
            },
        }
        _write_json(generation / "result.json", result)
        return BackendPackage(
            result=result,
            primary_checkpoint_relative_path=str(interaction["checkpoint_name"]),
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
            {"schema_name": "policy_improvement_theory_bridge_amendment_v1"},
            {
                "common_compute_targets": {
                    "unit": "recurrent_map_applications",
                    "pilot": 900,
                    "confirmatory": 1200,
                    "ablation": 1300,
                }
            },
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
            mock.patch(
                "scripts.policy_improvement_full_runtime.generate_registry",
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

    def test_v2_full_run_uses_static_registry_and_registered_population(self) -> None:
        protocol_path = self.project / "configs/policy_improvement_v2/protocol.json"
        registry_path = self.project / "configs/policy_improvement_v2/registry.json"
        theory_path = self.project / "configs/policy_improvement_v2/theory.json"
        base_policy_path = self.project / "configs/policy_improvement_v2/base.json"
        compute_path = self.project / "configs/policy_improvement_v2/compute.json"
        selection_path = self.project / "configs/policy_improvement_v2/selection.json"
        for path in (
            protocol_path,
            registry_path,
            theory_path,
            base_policy_path,
            compute_path,
            selection_path,
        ):
            _write_json(path, {"fixture": path.stem})
        authorization_sha256 = _digest("runtime authorization")
        registered_dataset = self.project / "data/registered-v2"
        registered_dataset.mkdir(parents=True)
        population_document = {
            "populations": {
                "validation_select": {
                    "population_id": "validation_select",
                    "split": "validation",
                    "count": 128,
                    "binding_sha256": _digest("validation select"),
                },
                "validation_bridge": {
                    "population_id": "validation_bridge",
                    "split": "validation",
                    "count": 128,
                    "binding_sha256": _digest("validation bridge"),
                },
            }
        }
        protocol = {
            "schema_name": "policy_improvement_protocol_v2",
            "schema_version": 2,
            "protocol_id": "policy-improvement-v2-20260818",
            "full_execution_gate": {
                "environment_variable": "RUN_UPITRM_FULL_EXPERIMENTS",
                "required_value": "1",
                "stage0_exempt": True,
                "stage1_to_stage3_blocked_by_base_policy": True,
            },
            "base_policy_artifact": {
                "schema_name": "policy_improvement_base_policy_artifact_v2",
                "schema_version": 1,
                "status": "available",
                "stage1_execution_allowed": True,
            },
            "dataset": {"root": "data/registered-v2"},
            "output_root": {"relative_path": "policy_improvement_v2"},
            "population_registry": {"sha256": _digest("populations")},
            "budgets": {
                "pilot": {
                    "checkpoint_environment_interactions": [
                        10000,
                        20000,
                        40000,
                        80000,
                    ],
                    "maximum_environment_interactions": 80000,
                    "selection_records": 128,
                    "bridge_records": 128,
                }
            },
        }
        row = {
            **self._row("fixed_base_exact_persistent"),
            "schema_name": "policy_improvement_registry_row_v2",
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "evaluation_population": "validation_select",
            "checkpoint_environment_interactions": [
                10000,
                20000,
                40000,
                80000,
            ],
        }
        registry = {
            "schema_name": "policy_improvement_registry_v2",
            "registry_schema_version": 1,
            "rows": [row],
        }
        protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        registry_sha256 = hashlib.sha256(canonical_json_bytes(registry)).hexdigest()
        theory = {
            "schema_name": "policy_improvement_theory_bridge_amendment_v2",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "population_registry_sha256": protocol["population_registry"]["sha256"],
            "source_registry_sha256": registry_sha256,
            "prior_amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
        }
        base_policy = {
            "schema_name": "policy_improvement_base_policy_amendment_v2",
            "runtime_authorization_sha256": authorization_sha256,
        }
        compute = {
            "schema_name": "policy_improvement_compute_freeze_v2",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "source_registry_sha256": registry_sha256,
            "prior_amendment_history_sha256": hashlib.sha256(
                canonical_json_bytes([theory, base_policy])
            ).hexdigest(),
            "runtime_authorization_sha256": authorization_sha256,
            "common_compute_targets": {
                "unit": "recurrent_map_applications",
                "pilot": 900,
                "confirmatory": 1200,
                "ablation": 1300,
            },
        }
        selection = {
            "schema_name": "policy_improvement_stage1_configuration_selection_v2",
            "protocol_sha256": protocol_sha256,
            "population_registry_sha256": protocol["population_registry"]["sha256"],
            "source_registry_sha256": registry_sha256,
            "prior_amendment_history_sha256": hashlib.sha256(
                canonical_json_bytes([theory, base_policy, compute])
            ).hexdigest(),
            "compute_freeze_sha256": hashlib.sha256(
                canonical_json_bytes(compute)
            ).hexdigest(),
            "base_policy_artifact_sha256": _digest("base artifact"),
            "runtime_authorization_sha256": authorization_sha256,
            "selection_population_binding_sha256": population_document["populations"][
                "validation_select"
            ]["binding_sha256"],
            "bridge_population_binding_sha256": population_document["populations"][
                "validation_bridge"
            ]["binding_sha256"],
            "selected_configurations": [
                {
                    "method_id": "fixed_base_exact_persistent",
                    "n": 2,
                    "K": 1,
                },
                {
                    "method_id": "fixed_base_exact_episodic",
                    "n": 4,
                    "K": 5,
                },
            ],
        }
        authenticated_base_policy = object()
        base_policy_artifact = "/registered/base_policy.pt"
        with (
            mock.patch(
                "scripts.policy_improvement_full_runtime.validate_protocol",
                return_value=protocol,
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.load_registered_base_configs",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.validate_registry_document",
                return_value=registry,
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.generate_registry",
                return_value=registry,
            ) as generate,
            mock.patch(
                "scripts.policy_improvement_populations.load_registered_populations",
                return_value=population_document,
            ),
            mock.patch(
                "scripts.policy_improvement_full_runtime.validate_v2_amendment_history",
                side_effect=lambda history, **_: [
                    theory,
                    base_policy,
                    compute,
                    *([selection] if len(history) == 4 else []),
                ],
            ) as validate_history,
            mock.patch(
                "scripts.policy_improvement_full_runtime."
                "authenticate_base_policy_artifact",
                return_value=authenticated_base_policy,
            ) as authenticate,
        ):
            # The registered base artifact is mandatory for learned execution.
            with self.assertRaisesRegex(
                FullRuntimeError,
                "requires the registered train-only base-policy artifact",
            ):
                load_registered_full_run(
                    project_root=self.project,
                    protocol_path=protocol_path,
                    registry_path=registry_path,
                    amendment_paths=[theory_path, base_policy_path, compute_path],
                    evidence_root=self.evidence,
                    dataset_root=registered_dataset,
                    row_id=str(row["run_id"]),
                    runtime_authorization_sha256=authorization_sha256,
                    environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
                )
            # The Stage 1 screen: three amendments, no V_select yet. It stops
            # at the first registered checkpoint, so it needs no cross-
            # generation continuation.
            screen_run = load_registered_full_run(
                project_root=self.project,
                protocol_path=protocol_path,
                registry_path=registry_path,
                amendment_paths=[theory_path, base_policy_path, compute_path],
                evidence_root=self.evidence,
                dataset_root=registered_dataset,
                row_id=str(row["run_id"]),
                runtime_authorization_sha256=authorization_sha256,
                environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
                base_policy_artifact=base_policy_artifact,
            )
            self.assertEqual(screen_run.interaction_checkpoints, (10000,))
            self.assertEqual(screen_run.final_environment_interactions, 10000)
            self.assertIs(screen_run.base_policy, authenticated_base_policy)
            with self.assertRaisesRegex(
                FullRuntimeError,
                "requires an authenticated V_select configuration selection",
            ):
                load_registered_full_run(
                    project_root=self.project,
                    protocol_path=protocol_path,
                    registry_path=registry_path,
                    amendment_paths=[theory_path, base_policy_path, compute_path],
                    evidence_root=self.evidence,
                    dataset_root=registered_dataset,
                    row_id=str(row["run_id"]),
                    runtime_authorization_sha256=authorization_sha256,
                    environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
                    require_stage1_selection=True,
                    base_policy_artifact=base_policy_artifact,
                )
            selected_run = load_registered_full_run(
                project_root=self.project,
                protocol_path=protocol_path,
                registry_path=registry_path,
                amendment_paths=[
                    theory_path,
                    base_policy_path,
                    compute_path,
                    selection_path,
                ],
                evidence_root=self.evidence,
                dataset_root=registered_dataset,
                row_id=str(row["run_id"]),
                runtime_authorization_sha256=authorization_sha256,
                environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
                require_stage1_selection=True,
                base_policy_artifact=base_policy_artifact,
            )
            self.assertEqual(len(selected_run.amendment_history), 4)
            self.assertIs(selected_run.base_policy, authenticated_base_policy)
            self.assertEqual(
                authenticate.call_args.args, (base_policy_artifact,)
            )
            self.assertEqual(
                authenticate.call_args.kwargs, {"amendment": base_policy}
            )
            rejected_selection = copy.deepcopy(selection)
            rejected_selection["selected_configurations"][0]["n"] = 4
            validate_history.side_effect = lambda history, **_: [
                theory,
                base_policy,
                compute,
                rejected_selection,
            ]
            with self.assertRaisesRegex(FullRuntimeError, "was not selected"):
                load_registered_full_run(
                    project_root=self.project,
                    protocol_path=protocol_path,
                    registry_path=registry_path,
                    amendment_paths=[
                        theory_path,
                        base_policy_path,
                        compute_path,
                        selection_path,
                    ],
                    evidence_root=self.evidence,
                    dataset_root=registered_dataset,
                    row_id=str(row["run_id"]),
                    runtime_authorization_sha256=authorization_sha256,
                    environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
                    require_stage1_selection=True,
                    base_policy_artifact=base_policy_artifact,
                )
            validate_history.side_effect = lambda history, **_: [
                theory,
                base_policy,
                compute,
                selection,
            ]
            alternate_dataset = self.project / "data/alternate-v2"
            alternate_dataset.mkdir()
            with self.assertRaisesRegex(
                FullRuntimeError,
                "differs from its registration",
            ):
                load_registered_full_run(
                    project_root=self.project,
                    protocol_path=protocol_path,
                    registry_path=registry_path,
                    amendment_paths=[
                        theory_path,
                        base_policy_path,
                        compute_path,
                        selection_path,
                    ],
                    evidence_root=self.evidence,
                    dataset_root=alternate_dataset,
                    row_id=str(row["run_id"]),
                    runtime_authorization_sha256=authorization_sha256,
                    environment={"RUN_UPITRM_FULL_EXPERIMENTS": "1"},
                    require_stage1_selection=True,
                    base_policy_artifact=base_policy_artifact,
                )
        self.assertEqual(
            selected_run.interaction_checkpoints,
            (10000, 20000, 40000, 80000),
        )
        self.assertEqual(selected_run.final_environment_interactions, 80000)
        self.assertEqual(selected_run.evaluation_records, 128)
        self.assertEqual(selected_run.compute_target_recurrent_map_applications, 900)
        self.assertEqual(selected_run.population_document, population_document)
        self.assertEqual(generate.call_args_list[0].args[1], [])
        self.assertEqual(len(generate.call_args.args[1]), 4)

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
        self.runtime_authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v2",
            "schema_version": 2,
            "authorization_id": "full-runtime-resolver-test-v1",
            "created_at_utc": "2026-08-16T12:00:00Z",
            "protocol_sha256": _digest("protocol"),
            "producer_git_commit": "b" * 40,
            "producer_source_manifest_sha256": "e" * 64,
            "launcher_sha256": "0" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "b" * 40,
                    "runtime_sha256": "c" * 64,
                    "runtime_profile_sha256": "d" * 64,
                    "selected_source_manifest_sha256": "d" * 64,
                }
                for role in (
                    "policy-improvement-training",
                    "policy-improvement-evaluation",
                    "policy-improvement-audit",
                    "policy-improvement-analysis",
                    "policy-improvement-full",
                    "policy-improvement-theory-bridge",
                )
            ],
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
            runtime_authorization_sha256=runtime_authorization_sha256(
                self.runtime_authorization
            ),
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
        with self.assertRaisesRegex(FullRuntimeError, "already complete"):
            execute_registered_run(
                self.run,
                backend=_Backend(),
                result_validator=lambda value: value,
            )
        runs = final.parent
        self.assertEqual(
            sorted(path.name for path in runs.iterdir()),
            [".locks", str(self.run.row["run_id"])],
        )

    def test_v3_full_resolver_uses_the_full_role_not_training(self) -> None:
        legacy_name, legacy_role = _full_producer_runtime_role(
            self.runtime_authorization
        )
        self.assertEqual(legacy_name, "policy-improvement-training")
        self.assertEqual(legacy_role["runtime_sha256"], "c" * 64)

        authorization = dict(self.runtime_authorization)
        authorization["schema_name"] = "policy_improvement_runtime_authorization_v3"
        authorization["schema_version"] = 3
        authorization["roles"] = [
            {
                **role,
                "runtime_sha256": f"{index + 1:x}" * 64,
            }
            for index, role in enumerate(self.runtime_authorization["roles"])
        ]
        role_name, role = _full_producer_runtime_role(authorization)
        self.assertEqual(role_name, "policy-improvement-full")
        self.assertEqual(role["runtime_sha256"], "5" * 64)

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
                self.assertEqual(
                    [path.name for path in runs.iterdir() if path.name != ".locks"],
                    [],
                )

    def test_failed_backend_attempt_survives_successful_retry(self) -> None:
        raised = None
        for _ in range(2):
            with self.assertRaises(PublishedFullRunFailure) as current:
                execute_registered_run(
                    self.run,
                    backend=_FailureBackend(),
                    result_validator=lambda value: value,
                )
            raised = current
        assert raised is not None
        final = raised.exception.published_run
        attempts = list((final / "attempts/complete").iterdir())
        self.assertEqual(len(attempts), 2)
        for attempt in attempts:
            self.assertEqual(
                {path.name for path in attempt.iterdir()},
                {"MANIFEST.json", "result.json"},
            )
        self.assertFalse((final / "segments").exists())
        self.assertEqual(
            execute_registered_run(
                self.run,
                backend=_Backend(),
                result_validator=lambda value: value,
            ),
            final,
        )
        manifest = json.loads(
            (final / "segments/env_000000080/MANIFEST.json").read_text()
        )
        self.assertEqual(len(manifest["prior_failed_attempts"]), 2)
        self.assertTrue((attempts[0] / "MANIFEST.json").is_file())

    def test_theory_checkpoint_resolves_only_from_complete_publication(self) -> None:
        final = execute_registered_run(
            self.run,
            backend=_ResolverBackend(self.runtime_authorization),
            result_validator=lambda value: value,
        )
        scheduled = resolve_authenticated_full_checkpoint(
            self.run,
            checkpoint_environment_interactions=10,
            runtime_authorization=self.runtime_authorization,
            result_validator=lambda value: value,
        )
        self.assertIsInstance(scheduled, AuthenticatedFullCheckpoint)
        self.addCleanup(os.close, scheduled.sealed_descriptor)
        self.assertEqual(scheduled.snapshot_kind, "scheduled")
        self.assertEqual(scheduled.environment_interactions, 10)
        with self.assertRaises(OSError):
            os.pwrite(scheduled.sealed_descriptor, b"tamper", 0)
        self.assertEqual(
            scheduled.path,
            final / "segments/env_000000080/checkpoints/scheduled/"
            "env_000000010/rl_checkpoint_step_10.pt",
        )
        interaction = resolve_authenticated_full_checkpoint(
            self.run,
            checkpoint_environment_interactions=80,
            runtime_authorization=self.runtime_authorization,
            result_validator=lambda value: value,
        )
        self.addCleanup(os.close, interaction.sealed_descriptor)
        self.assertEqual(interaction.snapshot_kind, "interaction_matched")
        self.assertIn("/interaction_matched/", interaction.path.as_posix())
        self.assertNotIn("/compute_matched/", interaction.path.as_posix())
        with self.assertRaisesRegex(FullRuntimeError, "registered full-run schedule"):
            resolve_authenticated_full_checkpoint(
                self.run,
                checkpoint_environment_interactions=30,
                runtime_authorization=self.runtime_authorization,
                result_validator=lambda value: value,
            )

    def test_scheduled_and_snapshot_checkpoints_cannot_be_substituted(self) -> None:
        """Adversarial test 3: kinds are not interchangeable, even byte-for-byte."""

        final = execute_registered_run(
            self.run,
            backend=_ResolverBackend(self.runtime_authorization),
            result_validator=lambda value: value,
        )
        generation = final / "segments/env_000000080"
        scheduled = (
            generation / "checkpoints/scheduled/env_000000010/rl_checkpoint_step_10.pt"
        )
        interaction = next(
            (generation / "checkpoints/interaction_matched").rglob("*.pt")
        )
        compute = next((generation / "checkpoints/compute_matched").rglob("*.pt"))
        self.assertNotEqual(scheduled.read_bytes(), interaction.read_bytes())
        self.assertNotEqual(compute.read_bytes(), interaction.read_bytes())

        for source, destination in (
            (interaction, scheduled),
            (compute, interaction),
            (scheduled, compute),
        ):
            with self.subTest(
                source=source.parent.name, target=destination.parent.name
            ):
                original = destination.read_bytes()
                destination.write_bytes(source.read_bytes())
                try:
                    for interactions in (10, 80):
                        with self.assertRaises(FullRuntimeError):
                            resolve_authenticated_full_checkpoint(
                                self.run,
                                checkpoint_environment_interactions=interactions,
                                runtime_authorization=self.runtime_authorization,
                                result_validator=lambda value: value,
                            )
                finally:
                    destination.write_bytes(original)

        # Restoring the exact published bytes makes resolution succeed again,
        # so the rejections above are not an artifact of the fixture.
        resolved = resolve_authenticated_full_checkpoint(
            self.run,
            checkpoint_environment_interactions=10,
            runtime_authorization=self.runtime_authorization,
            result_validator=lambda value: value,
        )
        self.addCleanup(os.close, resolved.sealed_descriptor)
        self.assertEqual(resolved.snapshot_kind, "scheduled")

    def test_theory_checkpoint_tamper_is_rejected_before_restore(self) -> None:
        final = execute_registered_run(
            self.run,
            backend=_ResolverBackend(self.runtime_authorization),
            result_validator=lambda value: value,
        )
        checkpoint = (
            final / "segments/env_000000080/checkpoints/scheduled/"
            "env_000000010/rl_checkpoint_step_10.pt"
        )
        checkpoint.write_bytes(b"attacker-controlled pickle bytes")
        with self.assertRaisesRegex(FullRuntimeError, "differs from its manifest"):
            resolve_authenticated_full_checkpoint(
                self.run,
                checkpoint_environment_interactions=10,
                runtime_authorization=self.runtime_authorization,
                result_validator=lambda value: value,
            )

    def test_theory_checkpoint_rejects_reinventoried_result_mixing(self) -> None:
        final = execute_registered_run(
            self.run,
            backend=_ResolverBackend(self.runtime_authorization),
            result_validator=lambda value: value,
        )
        generation = final / "segments/env_000000080"
        result_path = generation / "result.json"
        result = json.loads(result_path.read_text())
        result["artifacts"]["checkpoint_validation"] = _available(
            _digest("forged checkpoint validation")
        )
        result_sha256 = _write_json(result_path, result)
        manifest_path = generation / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["files"]["result.json"] = {
            "bytes": result_path.stat().st_size,
            "sha256": result_sha256,
        }
        manifest["storage_bytes"] = sum(
            int(identity["bytes"]) for identity in manifest["outputs"]["files"].values()
        )
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(FullRuntimeError, "primary validation"):
            resolve_authenticated_full_checkpoint(
                self.run,
                checkpoint_environment_interactions=80,
                runtime_authorization=self.runtime_authorization,
                result_validator=lambda value: value,
            )

    def test_theory_checkpoint_rejects_reinventoried_compute_work(self) -> None:
        final = execute_registered_run(
            self.run,
            backend=_ResolverBackend(self.runtime_authorization),
            result_validator=lambda value: value,
        )
        generation = final / "segments/env_000000080"
        run_manifest_path = generation / "RUN_MANIFEST.json"
        run_manifest = json.loads(run_manifest_path.read_text())
        compute = next(
            snapshot
            for snapshot in run_manifest["snapshots"]
            if snapshot["snapshot_kind"] == "compute_matched"
        )
        compute["recurrent_map_applications"] = 999
        run_manifest_sha256 = _write_json(run_manifest_path, run_manifest)

        result_path = generation / "result.json"
        result = json.loads(result_path.read_text())
        result["artifacts"]["run_manifest"] = _available(run_manifest_sha256)
        result_sha256 = _write_json(result_path, result)

        manifest_path = generation / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        for name, path, digest in (
            ("RUN_MANIFEST.json", run_manifest_path, run_manifest_sha256),
            ("result.json", result_path, result_sha256),
        ):
            manifest["outputs"]["files"][name] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        manifest["storage_bytes"] = sum(
            int(identity["bytes"]) for identity in manifest["outputs"]["files"].values()
        )
        _write_json(manifest_path, manifest)

        with self.assertRaisesRegex(FullRuntimeError, "snapshot differs"):
            resolve_authenticated_full_checkpoint(
                self.run,
                checkpoint_environment_interactions=80,
                runtime_authorization=self.runtime_authorization,
                result_validator=lambda value: value,
            )

    def test_contract_reports_non_executable_integration_hooks(self) -> None:
        contract = execution_contract(self.run)
        self.assertFalse(contract["execution_ready"])
        self.assertEqual(contract["interaction_checkpoints"], [10, 20, 40, 80])
        self.assertEqual(len(contract["blocked_by"]), 3)


if __name__ == "__main__":
    unittest.main()
