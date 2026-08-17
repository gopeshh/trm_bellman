#!/usr/bin/env fbpython
"""Synthetic and training-only tests for the read-only theory bridge."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from scripts.policy_improvement_registry import (
    generate_registry,
    registry_sha256,
    validate_registry_document,
)
from scripts.policy_improvement_full_runtime import AuthenticatedFullCheckpoint
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    runtime_authorization_sha256,
    validate_amendment_history,
    validate_protocol,
)
from scripts.policy_improvement_theory_bridge import (
    ReadOnlySnapshot,
    TheoryBackendInputs,
    TheoryBridgeError,
    TheoryOutcome,
    TheoryRollout,
    TheoryState,
    _state_identity,
    evaluate_theory_bridge,
)
from scripts.policy_improvement_theory_backend import (
    _registered_checkpoint_snapshot_kind,
    create_theory_bridge_backend,
)
from scripts.policy_improvement_theory_schema import (
    THEORY_REQUEST_SCHEMA_NAME,
    THEORY_SCHEMA_VERSION,
    theory_document_sha256,
    validate_request_against_amendment,
    validate_theory_amendment,
    validate_theory_result,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AMENDMENT_PATH = (
    _PROJECT_ROOT / "configs/policy_improvement_v1/amendments/theory_bridge_v1.json"
)
_PROTOCOL_PATH = _PROJECT_ROOT / "configs/policy_improvement_v1/protocol.json"
_REGISTRY_PATH = _PROJECT_ROOT / "configs/policy_improvement_v1/registry.json"


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _record_pairs(count: int) -> list[dict[str, object]]:
    return [
        {
            "record_index": index,
            "dataset_record_sha256": _digest(f"training-record-{index}"),
        }
        for index in range(count)
    ]


def _identity(
    checkpoint_payload: bytes,
    amendment: dict[str, Any],
    *,
    checkpoint_environment_interactions: int = 10000,
    record_count: int = 64,
    snapshot_kind: str | None = None,
) -> dict[str, Any]:
    pairs = _record_pairs(record_count)
    producer_commit = "1" * 40
    return {
        "protocol_sha256": amendment["protocol_sha256"],
        "theory_amendment_sha256": theory_document_sha256(amendment),
        "registry_row_sha256": _digest("registry-row"),
        "checkpoint": {
            "sha256": _digest(checkpoint_payload),
            "size_bytes": len(checkpoint_payload),
            "snapshot_kind": snapshot_kind
            or (
                "interaction_matched"
                if checkpoint_environment_interactions == 80000
                else "scheduled"
            ),
            "environment_interactions": checkpoint_environment_interactions,
        },
        "model": {
            "model_sha256": _digest("model"),
            "model_config_sha256": _digest("model-config"),
            "current_policy_sha256": _digest("current-policy"),
            "candidate_policy_sha256": _digest("candidate-policy"),
            "deployed_policy_sha256": _digest("deployed-policy"),
            "recurrent_transition_sha256": _digest("fixed-F-n"),
        },
        "config": {
            "file_sha256": _digest("config-file"),
            "base_canonical_sha256": _digest("config-canonical"),
            "effective_config_sha256": _digest("effective-config"),
        },
        "producer_source": {
            "git_commit": producer_commit,
            "source_manifest_sha256": _digest("producer-source-manifest"),
        },
        "training_runtime": {
            "role": "policy-improvement-full",
            "source_git_commit": producer_commit,
            "source_manifest_sha256": _digest("training-runtime-profile"),
            "runtime_sha256": _digest("training-runtime"),
            "runtime_profile_sha256": _digest("training-runtime-profile"),
            "selected_source_manifest_sha256": _digest("training-runtime-profile"),
            "runtime_authorization_sha256": _digest("runtime-authorization"),
            "launcher_sha256": _digest("launcher"),
        },
        "dataset_records": {
            "split": "validation",
            "split_manifest_sha256": _digest("validation-manifest"),
            "ordered_record_sha256": _digest("validation-order"),
            "selected_record_indices_sha256": _digest(
                canonical_json_bytes(list(range(record_count)))
            ),
            "selected_records_sha256": _digest(canonical_json_bytes(pairs)),
            "record_count": record_count,
        },
        "evaluator_source": {
            "git_commit": "2" * 40,
            "source_manifest_sha256": _digest("evaluator-source-manifest"),
        },
        "evaluator_runtime": {
            "runtime_sha256": _digest("evaluator-runtime"),
            "runtime_profile_sha256": _digest("evaluator-runtime-profile"),
            "runtime_authorization_sha256": _digest("runtime-authorization"),
            "launcher_sha256": _digest("launcher"),
        },
    }


def _runtime_authorization(identity: dict[str, Any]) -> dict[str, Any]:
    training = identity["training_runtime"]
    evaluator = identity["evaluator_runtime"]
    evaluator_source = identity["evaluator_source"]
    authorization = {
        "schema_name": "policy_improvement_runtime_authorization_v2",
        "schema_version": 2,
        "authorization_id": "theory-adapter-test-v1",
        "created_at_utc": "2026-08-16T12:00:00Z",
        "protocol_sha256": identity["protocol_sha256"],
        "producer_git_commit": identity["producer_source"]["git_commit"],
        "producer_source_manifest_sha256": identity["producer_source"][
            "source_manifest_sha256"
        ],
        "launcher_sha256": training["launcher_sha256"],
        "roles": [
            {
                "role": role,
                "source_git_commit": source_commit,
                "runtime_sha256": runtime_sha256,
                "runtime_profile_sha256": runtime_profile_sha256,
                "selected_source_manifest_sha256": runtime_profile_sha256,
            }
            for role, source_commit, runtime_sha256, runtime_profile_sha256 in (
                (
                    "policy-improvement-training",
                    training["source_git_commit"],
                    training["runtime_sha256"],
                    training["runtime_profile_sha256"],
                ),
                (
                    "policy-improvement-evaluation",
                    evaluator_source["git_commit"],
                    evaluator["runtime_sha256"],
                    evaluator["runtime_profile_sha256"],
                ),
                (
                    "policy-improvement-audit",
                    evaluator_source["git_commit"],
                    evaluator["runtime_sha256"],
                    evaluator["runtime_profile_sha256"],
                ),
                (
                    "policy-improvement-analysis",
                    evaluator_source["git_commit"],
                    evaluator["runtime_sha256"],
                    evaluator["runtime_profile_sha256"],
                ),
                (
                    "policy-improvement-full",
                    training["source_git_commit"],
                    training["runtime_sha256"],
                    training["runtime_profile_sha256"],
                ),
                (
                    "policy-improvement-theory-bridge",
                    evaluator_source["git_commit"],
                    evaluator["runtime_sha256"],
                    evaluator["runtime_profile_sha256"],
                ),
            )
        ],
    }
    digest = runtime_authorization_sha256(authorization)
    training["runtime_authorization_sha256"] = digest
    evaluator["runtime_authorization_sha256"] = digest
    return authorization


def _request(
    identity: dict[str, Any],
    *,
    horizon: int,
    reference_depth: int = 8,
) -> dict[str, Any]:
    estimator: dict[str, object]
    if horizon == 1:
        estimator = {
            "kind": "exact_masked_action_sum_v1",
            "rollout_count": 0,
            "base_seed": None,
            "seed_derivation": "not_applicable",
        }
    else:
        estimator = {
            "kind": "common_random_number_monte_carlo_v1",
            "rollout_count": 64,
            "base_seed": 26081601,
            "seed_derivation": "sha256_base_record_state_repeat_v1",
        }
    return {
        "schema_name": THEORY_REQUEST_SCHEMA_NAME,
        "schema_version": THEORY_SCHEMA_VERSION,
        "evaluation_id": f"synthetic-k{horizon}-m{reference_depth}",
        "run_id": "synthetic-fixed-base-exact-persistent",
        "method_id": "fixed_base_exact_persistent",
        "latent_mode": "persistent",
        "checkpoint_environment_interactions": identity["checkpoint"][
            "environment_interactions"
        ],
        "n": 2,
        "reference_depth_m": reference_depth,
        "reference_role": "primary" if reference_depth == 8 else "exploratory",
        "bellman_horizon": horizon,
        "gamma": 0.9,
        "alpha": 0.1,
        "bellman_estimator": estimator,
        "identity": identity,
        "test_data_opened": False,
    }


class _SyntheticBackend:
    def __init__(
        self,
        identity: dict[str, Any],
        checkpoint_path: Path,
        *,
        mutate_training: bool = False,
        mutate_checkpoint: bool = False,
    ) -> None:
        self.identity = copy.deepcopy(identity)
        self.checkpoint_path = checkpoint_path
        self.training_sha256 = _digest("training-state")
        self.snapshot_kind = str(identity["checkpoint"]["snapshot_kind"])
        self.environment_interactions = int(
            identity["checkpoint"]["environment_interactions"]
        )
        self.mutate_training = mutate_training
        self.mutate_checkpoint = mutate_checkpoint
        self.mutated = False
        self.rollout_calls: list[tuple[str, int]] = []
        self.endpoint_calls: list[tuple[str, int]] = []
        self.states: list[TheoryState] = []
        for pair in _record_pairs(64):
            index = int(pair["record_index"])
            state = TheoryState(
                state_id=f"state-{index:03d}",
                record_index=index,
                dataset_record_sha256=str(pair["dataset_record_sha256"]),
                registered_state_sha256="0" * 64,
                action_mask=(True, True, False),
                current_probabilities=(0.75, 0.25, 0.0),
                candidate_probabilities=(0.25, 0.75, 0.0),
                deployed_probabilities=(0.70, 0.30, 0.0),
            )
            self.states.append(
                replace(state, registered_state_sha256=_state_identity(state))
            )

    def identity_bundle(self) -> dict[str, Any]:
        return copy.deepcopy(self.identity)

    def read_only_snapshot(self) -> ReadOnlySnapshot:
        return ReadOnlySnapshot(
            checkpoint_sha256=str(self.identity["checkpoint"]["sha256"]),
            model_state_sha256=str(self.identity["model"]["model_sha256"]),
            training_state_sha256=self.training_sha256,
            recurrent_transition_sha256=str(
                self.identity["model"]["recurrent_transition_sha256"]
            ),
            snapshot_kind=self.snapshot_kind,
            environment_interactions=self.environment_interactions,
        )

    def registered_states(self) -> list[TheoryState]:
        return list(self.states)

    def _maybe_mutate(self) -> None:
        if self.mutated:
            return
        self.mutated = True
        if self.mutate_training:
            self.training_sha256 = _digest("mutated-training-state")
        if self.mutate_checkpoint:
            with self.checkpoint_path.open("ab") as handle:
                handle.write(b"mutation")

    def endpoint_value(self, state_id: str, depth: int) -> float:
        self.endpoint_calls.append((state_id, depth))
        if self.mutate_training or self.mutate_checkpoint:
            self._maybe_mutate()
        state_component = (
            int.from_bytes(hashlib.sha256(state_id.encode("ascii")).digest()[:2], "big")
            % 11
        )
        return float(state_component) / 10.0 + float(depth) / 20.0

    def exact_action_outcomes(
        self,
        state_id: str,
        action_index: int,
    ) -> list[TheoryOutcome]:
        return [
            TheoryOutcome(
                probability=1.0,
                reward=float(action_index),
                terminal=False,
                next_state_id=f"next-{state_id}-{action_index}",
            )
        ]

    def sample_rollout(
        self,
        state_id: str,
        horizon: int,
        seed: int,
    ) -> TheoryRollout:
        self.rollout_calls.append((state_id, seed))
        rewards = tuple(float((seed >> offset) & 1) for offset in range(horizon))
        return TheoryRollout(
            rewards=rewards,
            terminal=False,
            bootstrap_state_id=f"rollout-next-{state_id}-{seed}",
            trajectory_sha256=_digest(f"{state_id}:{horizon}:{seed}"),
        )


class TheoryAmendmentTest(unittest.TestCase):
    def test_amendment_binds_current_protocol_and_registry_before_outcomes(
        self,
    ) -> None:
        protocol = validate_protocol(_load_json(_PROTOCOL_PATH))
        source_registry = generate_registry(protocol)
        amendment = validate_theory_amendment(
            _load_json(_AMENDMENT_PATH),
            protocol=protocol,
            registry_sha256=registry_sha256(source_registry),
        )
        history = validate_amendment_history([amendment], protocol=protocol)
        regenerated = generate_registry(protocol, history)
        committed = _load_json(_REGISTRY_PATH)
        base = generate_registry(protocol)
        active_from_base = validate_registry_document(
            committed,
            protocol,
            history,
        )

        self.assertFalse(amendment["test_data_opened"])
        self.assertFalse(amendment["outcome_evidence_inspected"])
        self.assertEqual(amendment["reference_depths"]["primary_m"], 8)
        self.assertEqual(
            amendment["reference_depths"]["optional_exploratory_m"],
            [16],
        )
        self.assertFalse(amendment["analysis"]["scientific_selection"])
        self.assertEqual(
            _REGISTRY_PATH.read_bytes(),
            canonical_json_bytes(base) + b"\n",
        )
        self.assertEqual(active_from_base, regenerated)
        self.assertEqual(
            registry_sha256(validate_registry_document(committed, protocol, [])),
            registry_sha256(base),
        )
        self.assertEqual(
            [row["run_id"] for row in base["rows"] if row["phase"] == "stage0_smoke"],
            [
                row["run_id"]
                for row in active_from_base["rows"]
                if row["phase"] == "stage0_smoke"
            ],
        )


class TheoryBridgeEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint_payload = b"synthetic read-only checkpoint"
        self.checkpoint.write_bytes(self.checkpoint_payload)
        self.amendment = _load_json(_AMENDMENT_PATH)
        self.identity = _identity(self.checkpoint_payload, self.amendment)

    def _evaluate(
        self,
        *,
        horizon: int,
        backend: _SyntheticBackend | None = None,
        request: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], _SyntheticBackend]:
        active_request = request or _request(self.identity, horizon=horizon)
        active_backend = backend or _SyntheticBackend(
            active_request["identity"], self.checkpoint
        )
        result = evaluate_theory_bridge(
            request_document=active_request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=active_backend,
        )
        return validate_theory_result(result, request=active_request), active_backend

    def test_k1_uses_exact_masked_sum_and_preserves_all_state(self) -> None:
        before_bytes = self.checkpoint.read_bytes()
        result, backend = self._evaluate(horizon=1)

        self.assertEqual(result["state_count"], 64)
        self.assertEqual(backend.rollout_calls, [])
        self.assertEqual(
            result["monte_carlo_uncertainty"]["operator_standard_error"]["maximum"],
            0.0,
        )
        self.assertLessEqual(result["metrics"]["centering_defect"]["maximum"], 1e-12)
        self.assertLessEqual(result["metrics"]["deployment_tv"]["maximum"], 1e-12)
        self.assertTrue(result["persistent_semantics"]["endpoint_depth_only"])
        self.assertTrue(result["persistent_semantics"]["F_n_unchanged"])
        self.assertEqual(self.checkpoint.read_bytes(), before_bytes)
        self.assertEqual(backend.training_sha256, _digest("training-state"))

    def test_k5_uses_one_preregistered_crn_path_per_state_repeat(self) -> None:
        result, backend = self._evaluate(horizon=5)

        self.assertEqual(len(backend.rollout_calls), 64 * 64)
        self.assertEqual(len(set(backend.rollout_calls)), 64 * 64)
        rollout_endpoint_depths = {
            depth
            for state_id, depth in backend.endpoint_calls
            if state_id.startswith("rollout-next-")
        }
        self.assertEqual(rollout_endpoint_depths, {8})
        self.assertGreater(
            result["monte_carlo_uncertainty"]["operator_standard_error"]["maximum"],
            0.0,
        )
        self.assertGreater(
            result["monte_carlo_uncertainty"]["B_nm_standard_error"]["maximum"],
            0.0,
        )

    def test_missing_identity_fails_closed(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        del request["identity"]["model"]["candidate_policy_sha256"]
        backend = _SyntheticBackend(self.identity, self.checkpoint)

        with self.assertRaisesRegex(TheoryBridgeError, "identity.model fields differ"):
            self._evaluate(horizon=1, backend=backend, request=request)

    def test_mixed_backend_identity_fails_closed(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        mixed_identity = copy.deepcopy(self.identity)
        mixed_identity["dataset_records"]["split_manifest_sha256"] = _digest(
            "other-validation-manifest"
        )
        backend = _SyntheticBackend(mixed_identity, self.checkpoint)

        with self.assertRaisesRegex(TheoryBridgeError, "missing or mixed"):
            self._evaluate(horizon=1, backend=backend, request=request)

    def test_checkpoint_progress_and_snapshot_kind_fail_closed(self) -> None:
        scheduled_request = _request(copy.deepcopy(self.identity), horizon=1)
        checked_scheduled, _ = validate_request_against_amendment(
            scheduled_request,
            self.amendment,
        )
        self.assertEqual(
            checked_scheduled["identity"]["checkpoint"]["snapshot_kind"],
            "scheduled",
        )

        final_identity = _identity(
            self.checkpoint_payload,
            self.amendment,
            checkpoint_environment_interactions=80000,
            record_count=256,
        )
        final_request = _request(final_identity, horizon=1)
        checked_final, _ = validate_request_against_amendment(
            final_request,
            self.amendment,
        )
        self.assertEqual(
            checked_final["identity"]["checkpoint"]["snapshot_kind"],
            "interaction_matched",
        )

        wrong_final = copy.deepcopy(final_request)
        wrong_final["identity"]["checkpoint"]["snapshot_kind"] = "scheduled"
        with self.assertRaisesRegex(ValueError, "checkpoint progress"):
            validate_request_against_amendment(wrong_final, self.amendment)

        wrong_scheduled = copy.deepcopy(scheduled_request)
        wrong_scheduled["identity"]["checkpoint"][
            "snapshot_kind"
        ] = "interaction_matched"
        with self.assertRaisesRegex(ValueError, "checkpoint progress"):
            validate_request_against_amendment(wrong_scheduled, self.amendment)

        request = _request(copy.deepcopy(self.identity), horizon=1)
        request["checkpoint_environment_interactions"] = 20000
        backend = _SyntheticBackend(request["identity"], self.checkpoint)
        with self.assertRaisesRegex(TheoryBridgeError, "checkpoint progress"):
            self._evaluate(horizon=1, backend=backend, request=request)

        request = _request(copy.deepcopy(self.identity), horizon=1)
        backend = _SyntheticBackend(request["identity"], self.checkpoint)
        backend.snapshot_kind = "compute_matched"
        with self.assertRaisesRegex(TheoryBridgeError, "snapshot_kind"):
            self._evaluate(horizon=1, backend=backend, request=request)

        backend = _SyntheticBackend(request["identity"], self.checkpoint)
        backend.environment_interactions = 20000
        with self.assertRaisesRegex(TheoryBridgeError, "snapshot differs"):
            self._evaluate(horizon=1, backend=backend, request=request)

    def test_result_validation_binds_exact_request_and_summaries(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        result, _ = self._evaluate(horizon=1, request=request)

        mixed = copy.deepcopy(result)
        mixed["run_id"] = "other-run"
        with self.assertRaisesRegex(
            ValueError,
            "differs from its exact request",
        ):
            validate_theory_result(mixed, request=request)

        mixed = copy.deepcopy(result)
        mixed["request_sha256"] = _digest("other-request")
        with self.assertRaisesRegex(
            ValueError,
            "differs from its exact request",
        ):
            validate_theory_result(mixed, request=request)

        mixed = copy.deepcopy(result)
        mixed["identity"]["checkpoint"]["environment_interactions"] = 20000
        with self.assertRaises(ValueError):
            validate_theory_result(mixed, request=request)

        mixed = copy.deepcopy(result)
        mixed["metrics"]["D_nm"]["minimum"] = max(
            0.0,
            mixed["metrics"]["D_nm"]["minimum"] - 0.01,
        )
        with self.assertRaisesRegex(ValueError, "differs from its state rows"):
            validate_theory_result(mixed, request=request)

        request_k5 = _request(copy.deepcopy(self.identity), horizon=5)
        result_k5, _ = self._evaluate(horizon=5, request=request_k5)
        mixed = copy.deepcopy(result_k5)
        mixed["monte_carlo_uncertainty"]["operator_standard_error"]["maximum"] += 0.01
        with self.assertRaisesRegex(ValueError, "differs from its state rows"):
            validate_theory_result(mixed, request=request_k5)

    def test_test_records_are_rejected_without_a_test_open_path(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        request["identity"]["dataset_records"]["split"] = "test"
        request["test_data_opened"] = True
        backend = _SyntheticBackend(request["identity"], self.checkpoint)

        with self.assertRaises(TheoryBridgeError):
            self._evaluate(horizon=1, backend=backend, request=request)

    def test_training_state_mutation_is_detected(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        backend = _SyntheticBackend(
            request["identity"],
            self.checkpoint,
            mutate_training=True,
        )

        with self.assertRaisesRegex(
            TheoryBridgeError, "mutated checkpoint or training"
        ):
            self._evaluate(horizon=1, backend=backend, request=request)

    def test_checkpoint_mutation_is_detected(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        backend = _SyntheticBackend(
            request["identity"],
            self.checkpoint,
            mutate_checkpoint=True,
        )

        with self.assertRaisesRegex(TheoryBridgeError, "metadata changed"):
            self._evaluate(horizon=1, backend=backend, request=request)

    def test_concrete_adapter_uses_only_the_sealed_full_backend_api(self) -> None:
        request = _request(copy.deepcopy(self.identity), horizon=1)
        published_roles = {"model": _digest("published-role")}
        published_model = hashlib.sha256(
            canonical_json_bytes(published_roles)
        ).hexdigest()
        request["identity"]["model"]["model_sha256"] = published_model
        runtime_authorization = _runtime_authorization(request["identity"])
        session = _SyntheticBackend(request["identity"], self.checkpoint)
        registered_run = SimpleNamespace(
            row={
                "run_id": request["run_id"],
                "method_id": request["method_id"],
                "n": request["n"],
                "K": request["bellman_horizon"],
                "alpha": request["alpha"],
                "base_method_id": request["method_id"],
                "config_override": {
                    "inner_unroll_n": request["n"],
                    "K": request["bellman_horizon"],
                    "mixture_alpha": request["alpha"],
                },
                "base_config_canonical_sha256": request["identity"]["config"][
                    "base_canonical_sha256"
                ],
                "expected_effective_config_sha256": request["identity"]["config"][
                    "effective_config_sha256"
                ],
                "evaluation_split": "validation",
            },
            amendment_history=(self.amendment,),
            protocol={
                "methods": [
                    {
                        "id": request["method_id"],
                        "config_sha256": request["identity"]["config"]["file_sha256"],
                    }
                ]
            },
            protocol_sha256=request["identity"]["protocol_sha256"],
            registry_row_sha256=request["identity"]["registry_row_sha256"],
            interaction_checkpoints=(10000, 80000),
            final_environment_interactions=80000,
        )
        calls: dict[str, object] = {}

        def load_registered_run(**kwargs: object) -> object:
            calls["load"] = kwargs
            return registered_run

        def open_session(
            session_request: dict[str, object],
            checkpoint_path: Path,
            **kwargs: object,
        ) -> object:
            calls["open_request"] = session_request
            calls["open_kwargs"] = kwargs
            self.assertEqual(checkpoint_path, self.checkpoint)
            self.assertGreaterEqual(kwargs["sealed_checkpoint_descriptor"], 0)
            self.assertEqual(
                kwargs["authenticated_model_state_sha256"],
                published_model,
            )
            self.assertEqual(
                kwargs["authenticated_role_state_sha256s"],
                published_roles,
            )
            self.assertEqual(
                kwargs["authenticated_validation_sha256"],
                _digest("validation"),
            )
            return session

        full_module = SimpleNamespace(open_theory_bridge_session=open_session)
        checkpoint_loader = mock.Mock()
        training_module = SimpleNamespace(_load_checkpoint_payload=checkpoint_loader)

        def resolved_checkpoint(*args: object, **kwargs: object) -> object:
            return AuthenticatedFullCheckpoint(
                path=self.checkpoint,
                sha256=request["identity"]["checkpoint"]["sha256"],
                size_bytes=request["identity"]["checkpoint"]["size_bytes"],
                snapshot_kind="scheduled",
                environment_interactions=10000,
                model_state_sha256=published_model,
                role_state_sha256s=published_roles,
                parent_checkpoint_sha256=None,
                generation_manifest_sha256=_digest("generation-manifest"),
                run_manifest_sha256=_digest("run-manifest"),
                validation_sha256=_digest("validation"),
                sealed_descriptor=os.open(self.checkpoint, os.O_RDONLY),
            )

        def import_module(name: str) -> object:
            if name == "policy_improvement_full_backend":
                return full_module
            if name == "upi_trm_train":
                return training_module
            raise AssertionError(f"unexpected import {name}")

        inputs = TheoryBackendInputs(
            project_root=self.root,
            protocol_path=_PROTOCOL_PATH,
            registry_path=_REGISTRY_PATH,
            amendment_paths=(_AMENDMENT_PATH,),
            evidence_root=self.root,
            dataset_root=self.root,
            row_id=str(request["run_id"]),
            runtime_authorization=runtime_authorization,
        )
        with mock.patch(
            "scripts.policy_improvement_theory_backend.load_registered_full_run",
            side_effect=load_registered_run,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.load_registered_base_configs",
            return_value={request["method_id"]: {"gamma": request["gamma"]}},
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.resolve_authenticated_full_checkpoint",
            side_effect=resolved_checkpoint,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.importlib.import_module",
            side_effect=import_module,
        ):
            backend = create_theory_bridge_backend(request, self.checkpoint, inputs)

        self.assertEqual(len(backend.registered_states()), 64)
        self.assertEqual(backend.read_only_snapshot(), session.read_only_snapshot())
        self.assertIs(calls["open_kwargs"]["training_module"], training_module)
        self.assertEqual(
            calls["open_request"]["registered_state_indices"],
            tuple(range(64)),
        )
        self.assertEqual(
            _registered_checkpoint_snapshot_kind(registered_run, 10000),
            "scheduled",
        )
        self.assertEqual(
            _registered_checkpoint_snapshot_kind(registered_run, 80000),
            "interaction_matched",
        )
        with self.assertRaisesRegex(TheoryBridgeError, "registered run schedule"):
            _registered_checkpoint_snapshot_kind(registered_run, 20000)

        mixed_gamma = copy.deepcopy(request)
        mixed_gamma["gamma"] = 0.8
        with mock.patch(
            "scripts.policy_improvement_theory_backend.load_registered_full_run",
            side_effect=load_registered_run,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.load_registered_base_configs",
            return_value={request["method_id"]: {"gamma": request["gamma"]}},
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.resolve_authenticated_full_checkpoint",
            side_effect=resolved_checkpoint,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.importlib.import_module",
            side_effect=import_module,
        ), self.assertRaisesRegex(
            TheoryBridgeError, "authenticated effective config"
        ):
            create_theory_bridge_backend(mixed_gamma, self.checkpoint, inputs)

        mixed_kind = copy.deepcopy(request)
        mixed_kind["identity"]["checkpoint"]["snapshot_kind"] = "interaction_matched"
        with mock.patch(
            "scripts.policy_improvement_theory_backend.load_registered_full_run",
            side_effect=load_registered_run,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.importlib.import_module",
            side_effect=import_module,
        ), self.assertRaisesRegex(
            TheoryBridgeError, "registered run schedule"
        ):
            create_theory_bridge_backend(mixed_kind, self.checkpoint, inputs)

        external = self.root / "caller-controlled.pt"
        external.write_bytes(b"caller-controlled pickle")
        external_request = copy.deepcopy(request)
        external_request["identity"]["checkpoint"].update(
            {
                "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
                "size_bytes": external.stat().st_size,
            }
        )
        calls.pop("open_request", None)
        imported_full_backend = False
        imported_training = False

        def import_without_training(name: str) -> object:
            nonlocal imported_full_backend, imported_training
            if name == "policy_improvement_full_backend":
                imported_full_backend = True
                return full_module
            if name == "upi_trm_train":
                imported_training = True
                return training_module
            raise AssertionError(f"unexpected import {name}")

        with mock.patch(
            "scripts.policy_improvement_theory_backend.load_registered_full_run",
            side_effect=load_registered_run,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.resolve_authenticated_full_checkpoint",
            side_effect=resolved_checkpoint,
        ), mock.patch(
            "scripts.policy_improvement_theory_backend.importlib.import_module",
            side_effect=import_without_training,
        ), self.assertRaisesRegex(
            TheoryBridgeError, "published evidence"
        ):
            create_theory_bridge_backend(external_request, external, inputs)
        self.assertFalse(imported_full_backend)
        self.assertFalse(imported_training)
        self.assertNotIn("open_request", calls)
        checkpoint_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
