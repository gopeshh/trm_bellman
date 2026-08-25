#!/usr/bin/env fbpython

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

from policy_improvement_sealed_evidence import seal_generation_checkpoint
from scripts import policy_improvement_audit
from scripts.policy_improvement_schema import canonical_json_bytes
from scripts.policy_improvement_theory_backend_v2 import (
    _AuthenticatedStage0Checkpoint,
    _independently_audit_stage0,
    _resolve_stage0_checkpoint,
    create_theory_bridge_backend_v2,
    SealedTheoryBackendV2,
)
from scripts.policy_improvement_theory_bridge_v2 import (
    _seed,
    BellmanRolloutV2,
    CurrentPolicyReturnV2,
    evaluate_theory_bridge_v2,
    main,
    PairedPolicyReturnV2,
    PersistentEndpointWitnessV2,
    ReadOnlySnapshotV2,
    state_identity,
    TheoryBackendInputsV2,
    TheoryBridgeV2Error,
    TheoryOutcomeV2,
    TheoryStateV2,
    TrainingAdvantageEstimatorV2,
)
from scripts.policy_improvement_theory_schema_v2 import (
    build_stage0_theory_request,
    build_validation_theory_request,
    PROTOCOL_ID,
    THEORY_AMENDMENT_SCHEMA_NAME,
    theory_document_sha256,
    THEORY_REQUEST_SCHEMA_NAME,
    THEORY_RESULT_SCHEMA_NAME,
    THEORY_SCHEMA_VERSION,
    validate_theory_result,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_INDICES = [17, 3, 91, 22, 5, 73, 40, 11]
_VALIDATION_INDICES = [(index * 37) % 256 for index in range(128)]


def _records(indices: list[int]) -> list[dict[str, object]]:
    return [
        {"record_index": index, "dataset_record_sha256": _digest(f"record-{index}")}
        for index in indices
    ]


def _amendment() -> dict[str, Any]:
    return {
        "schema_name": THEORY_AMENDMENT_SCHEMA_NAME,
        "schema_version": THEORY_SCHEMA_VERSION,
        "amendment_id": "theory-bridge-v2",
        "created_at_utc": "2026-08-18T00:00:00Z",
        "protocol_id": PROTOCOL_ID,
        "protocol_schema_name": "policy_improvement_protocol_v2",
        "protocol_schema_version": 2,
        "protocol_sha256": _digest("protocol-v2"),
        "population_registry_sha256": _digest("populations-v2"),
        "source_registry_schema_name": "policy_improvement_registry_v2",
        "source_registry_schema_version": 1,
        "prior_amendment_history_sha256": _digest("prior-amendments"),
        "source_registry_sha256": _digest("registry-v2"),
        "test_data_opened": False,
        "outcome_evidence_inspected": False,
        "result_schema": {
            "schema_name": THEORY_RESULT_SCHEMA_NAME,
            "schema_version": THEORY_SCHEMA_VERSION,
        },
        "evaluator_contract": {
            "checkpoint_access": "authenticated_write_sealed_read_only_descriptor",
            "launcher_purpose": "policy-improvement-theory-bridge",
            "persistent_reference_semantics": (
                "m_changes_endpoint_evaluator_only_deployed_F_n_policy_transition_"
                "and_carried_successor_latent_fixed"
            ),
            "population_id": "validation_bridge",
            "runtime_role": "policy-improvement-theory-bridge",
            "selection_use": "none",
            "source_tree_fallback": False,
            "training_state_mutation": "forbidden",
        },
        "metrics": {
            "D_nm": "registered",
            "bellman_residual_proxy": "registered",
            "B_nm": "registered",
            "policy_overlap_tau": "registered",
            "constructed_centering_roundoff": "registered",
            "training_estimator_centering_defect": "registered",
            "training_estimator_parity_max_abs_error": "registered",
            "exact_mixture_deployment_identity_tv": "registered",
            "deployment_discrepancy_delta_dep": "registered",
            "E_n": "registered",
        },
        "reference_depths": {
            "primary_m": 8,
            "optional_exploratory_m": [16],
            "m16_checkpoint_environment_interactions": [80000],
            "selection_use": "none",
        },
        "bellman_estimators": {
            "K1": {
                "horizon": 1,
                "kind": "exact_masked_action_sum_v2",
                "rollout_count": 0,
                "uncertainty": "exact_zero_monte_carlo_standard_error",
            },
            "K5": {
                "horizon": 5,
                "kind": "common_random_number_monte_carlo_v2",
                "rollout_count": 8,
                "base_seed": 104729,
                "seed_derivation": "sha256_protocol_checkpoint_record_state_repeat_v2",
                "uncertainty": "sample_standard_error_of_operator_mean",
            },
        },
        "predictive_return_estimator": {
            "rollout_count": 4,
            "base_seed": 130363,
            "seed_derivation": "sha256_protocol_checkpoint_record_state_repeat_v2",
            "maximum_environment_steps": 16,
            "terminal_requirement": "must_terminate_without_bootstrap",
            "common_random_numbers": True,
            "independent_from_bellman_estimator": True,
            "policy": "current_policy",
            "reported_uncertainty": "sample_standard_error",
        },
        "centering_contract": {
            "constructed_centering_roundoff_absolute_tolerance": 1e-12,
            "miscentered_estimator_action": "reject",
            "trainer_reconstruction": (
                "exact_frozen_checkpoint_advantage_tensor_with_registered_action_mask_"
                "clipping_and_recentering"
            ),
            "training_estimator_parity_absolute_tolerance": 1e-12,
        },
        "deployment_contract": {
            "exact_method_role": "identity_check_expected_zero",
            "exact_mixture_identity_tv_absolute_tolerance": 1e-12,
            "nonexact_policy_role": "deployment_discrepancy_nonselection_diagnostic",
        },
        "checkpoint_schedule": {
            "environment_interactions": [10000, 20000, 40000, 80000],
            "optional_exploratory_reference_depths_at_final": [16],
            "primary_reference_depths": [8, 8, 8, 8],
        },
        "multi_fidelity_exact_method_screen": {
            "all_rows_required_through_environment_interactions": 10000,
            "continuation_checkpoints": [20000, 40000, 80000],
            "final_exact_selection_checkpoint": 80000,
            "final_tie_break": [
                "persistent_before_episodic",
                "n_ascending",
                "K_ascending",
            ],
            "methods": [
                "fixed_base_exact_persistent",
                "fixed_base_exact_episodic",
            ],
            "population_id": "validation_select",
            "theory_bridge_selection_use": "none",
            "within_latent_mode_selection_checkpoint": 10000,
            "within_latent_mode_tie_break": ["n_ascending", "K_ascending"],
        },
        "analysis": {
            "B_nm_calibration_bins": {
                "count": 4,
                "kind": "fixed_rank_quantile_bins_v1",
                "ties": "stable_record_index_order",
            },
            "alpha_values": [0.05, 0.1, 0.2],
            "claims": "no_uniform_certificate_and_no_claim_from_smoke_values",
            "deployment_discrepancy_policies": [
                "legacy_parameter_interpolation",
                "registered_distilled_realization_when_available",
            ],
            "finite_population_only": True,
            "flexible_model_fitting": False,
            "population_id": "validation_bridge",
            "post_outcome_threshold_tuning": False,
            "registered_targets": [
                "spearman_B_nm_vs_E_n",
                "spearman_D_nm_vs_E_n",
                "spearman_residual_penalty_vs_E_n",
                "fixed_B_nm_calibration_bins_with_paired_mean_E_n",
                "cluster_uncertainty_by_training_seed",
                (
                    "alpha_penalty_proxy_vs_negative_paired_current_to_mixture_"
                    "return_probability"
                ),
            ],
            "scientific_selection": False,
        },
        "smoke_policy": {
            "allowed_population_ids": ["stage0_smoke"],
            "allowed_splits": ["train"],
            "paper_evidence_eligible": False,
            "scientific_selection": False,
            "test_records": False,
            "validation_records": False,
        },
    }


def _identity(
    amendment: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    records = _records(_INDICES)
    source_commit = "1" * 40
    profile = _digest("source-profile")
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_schema_name": "policy_improvement_protocol_v2",
        "protocol_schema_version": 2,
        "protocol_sha256": amendment["protocol_sha256"],
        "population_registry_schema_name": "policy_improvement_populations_v2",
        "population_registry_schema_version": 1,
        "population_registry_sha256": amendment["population_registry_sha256"],
        "registry_schema_name": "policy_improvement_registry_v2",
        "registry_schema_version": 1,
        "registry_sha256": amendment["source_registry_sha256"],
        "registry_row_schema_name": "policy_improvement_registry_row_v2",
        "registry_row_schema_version": 1,
        "theory_amendment_sha256": theory_document_sha256(amendment),
        "registry_row_sha256": _digest("row"),
        "checkpoint": {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "size_bytes": checkpoint.stat().st_size,
            "snapshot_kind": "smoke_resume",
            "environment_interactions": 32,
        },
        "model": {
            "model_sha256": _digest("model"),
            "model_config_sha256": _digest("model-config"),
            "current_policy_sha256": _digest("current"),
            "candidate_policy_sha256": _digest("candidate"),
            "deployed_policy_sha256": _digest("deployed"),
            "recurrent_transition_sha256": _digest("F-n"),
        },
        "config": {
            "file_sha256": _digest("config-file"),
            "base_canonical_sha256": _digest("base-config"),
            "effective_config_sha256": _digest("effective-config"),
        },
        "producer_source": {
            "git_commit": source_commit,
            "source_manifest_sha256": _digest("producer-manifest"),
        },
        "training_runtime": {
            "role": "policy-improvement-smoke",
            "source_git_commit": source_commit,
            "source_manifest_sha256": profile,
            "runtime_sha256": _digest("training-runtime"),
            "runtime_profile_sha256": profile,
            "selected_source_manifest_sha256": profile,
            "runtime_authorization_sha256": _digest("authorization"),
            "launcher_sha256": _digest("launcher"),
        },
        "dataset_records": {
            "split": "train",
            "population_id": "stage0_smoke",
            "population_binding_sha256": _digest("stage0-population-binding"),
            "split_manifest_sha256": _digest("train-manifest"),
            "ordered_record_sha256": _digest("train-order"),
            "ordered_input_sha256": _digest("train-input-order"),
            "selected_record_indices": _INDICES,
            "selected_record_indices_sha256": theory_document_sha256(_INDICES),
            "selected_records": records,
            "selected_records_sha256": theory_document_sha256(records),
            "selected_input_sha256s": [_digest(f"input-{index}") for index in _INDICES],
            "selected_input_sha256s_sha256": theory_document_sha256(
                [_digest(f"input-{index}") for index in _INDICES]
            ),
            "record_count": 8,
        },
        "evaluator_source": {
            "git_commit": source_commit,
            "source_manifest_sha256": _digest("evaluator-source"),
        },
        "evaluator_runtime": {
            "runtime_sha256": _digest("evaluator-runtime"),
            "runtime_profile_sha256": _digest("evaluator-profile"),
            "runtime_authorization_sha256": _digest("authorization"),
            "launcher_sha256": _digest("launcher"),
        },
    }


def _request(
    amendment: dict[str, Any],
    identity: dict[str, Any],
    *,
    method_id: str = "fixed_base_exact_persistent",
) -> dict[str, Any]:
    return {
        "schema_name": THEORY_REQUEST_SCHEMA_NAME,
        "schema_version": THEORY_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evaluation_id": "stage0-theory-smoke",
        "evaluation_population": "stage0_smoke",
        "run_id": "stage0-run",
        "method_id": method_id,
        "latent_mode": (
            "episodic" if method_id == "fixed_base_exact_episodic" else "persistent"
        ),
        "checkpoint_environment_interactions": 32,
        "n": 2,
        "reference_depth_m": 8,
        "reference_role": "primary",
        "bellman_horizon": 1,
        "gamma": 0.9,
        "alpha": 0.1,
        "bellman_estimator": copy.deepcopy(amendment["bellman_estimators"]["K1"]),
        "return_estimator": copy.deepcopy(amendment["predictive_return_estimator"]),
        "advantage_clipping": {"kind": "none", "clip_value": None},
        "constructed_centering_tolerance": amendment["centering_contract"][
            "constructed_centering_roundoff_absolute_tolerance"
        ],
        "centering_parity_tolerance": amendment["centering_contract"][
            "training_estimator_parity_absolute_tolerance"
        ],
        "deployment_identity_tolerance": amendment["deployment_contract"][
            "exact_mixture_identity_tv_absolute_tolerance"
        ],
        "scientific_selection": False,
        "paper_evidence_eligible": False,
        "identity": identity,
        "test_data_opened": False,
    }


def _validation_identity(
    amendment: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    identity = _identity(amendment, checkpoint)
    indices = list(_VALIDATION_INDICES)
    records = _records(indices)
    inputs = [_digest(f"input-{index}") for index in indices]
    identity["checkpoint"] = {
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "size_bytes": checkpoint.stat().st_size,
        "snapshot_kind": "scheduled",
        "environment_interactions": 10000,
    }
    identity["training_runtime"]["role"] = "policy-improvement-full"
    identity["dataset_records"] = {
        "split": "validation",
        "population_id": "validation_bridge",
        "population_binding_sha256": _digest("validation-bridge-binding"),
        "split_manifest_sha256": _digest("validation-manifest"),
        "ordered_record_sha256": _digest("validation-bridge-record-order"),
        "ordered_input_sha256": _digest("validation-bridge-input-order"),
        "selected_record_indices": indices,
        "selected_record_indices_sha256": theory_document_sha256(indices),
        "selected_records": records,
        "selected_records_sha256": theory_document_sha256(records),
        "selected_input_sha256s": inputs,
        "selected_input_sha256s_sha256": theory_document_sha256(inputs),
        "record_count": 128,
    }
    return identity


def _validation_request(
    amendment: dict[str, Any],
    identity: dict[str, Any],
    *,
    method_id: str = "fixed_base_exact_persistent",
) -> dict[str, Any]:
    row = {
        "row_kind": "concrete",
        "phase": "stage1_screen",
        "run_id": "validation-run",
        "method_id": method_id,
        "evaluation_split": "validation",
        "evaluation_population": "validation_select",
        "n": 2,
        "K": 1,
        "alpha": 0.1,
        "checkpoint_environment_interactions": [10000, 20000, 40000, 80000],
    }
    return build_validation_theory_request(
        amendment_value=amendment,
        row=row,
        identity=identity,
        effective_config={"gamma": 0.9, "advantage_clip": None},
        checkpoint_environment_interactions=10000,
    )


_MUTABLE_NAMES = (
    "optimizer_states",
    "scheduler_states",
    "rng_states",
    "replay_state",
    "collector_state",
    "environment_state",
    "persistent_latent_state",
    "exact_centering_state",
)


class _Backend:
    def __init__(
        self,
        identity: dict[str, Any],
        *,
        method_id: str = "fixed_base_exact_persistent",
        miscentered: bool = False,
        carry_depends_on_m: bool = False,
        reorder: bool = False,
        non_mixture_deployment: bool = False,
        unnormalized_states: bool = False,
        masked_leakage: bool = False,
    ) -> None:
        self.identity = copy.deepcopy(identity)
        self.method_id = method_id
        self.miscentered = miscentered
        self.carry_depends_on_m = carry_depends_on_m
        self.reorder = reorder
        self.non_mixture_deployment = non_mixture_deployment
        self.unnormalized_states = unnormalized_states
        self.masked_leakage = masked_leakage
        self.closed = False
        self.transaction_active = False
        self.return_calls: list[tuple[str, int]] = []
        self.paired_return_calls: list[tuple[str, int, float]] = []
        self.mutable = {name: _digest(name) for name in _MUTABLE_NAMES}
        self.normalization: dict[str, object] = {
            "kind": "float32_categorical_to_binary64_renormalization",
            "float32_mass_envelope": 1e-4,
            "normalized_state_count": 2,
            "maximum_valid_mass_error": 3.2e-8,
            "maximum_normalization_correction": 1.1e-9,
            "masked_entries_zeroed_before_validation": False,
            "deployed_reconstructed_from_mixture": False,
        }
        self.states = self._states()

    def begin_read_only_evaluation(self) -> None:
        if self.transaction_active:
            raise RuntimeError("duplicate transaction")
        self.transaction_active = True

    def end_read_only_evaluation(self) -> None:
        if not self.transaction_active:
            raise RuntimeError("missing transaction")
        self.transaction_active = False

    def _states(self) -> tuple[TheoryStateV2, ...]:
        result: list[TheoryStateV2] = []
        for index in self.identity["dataset_records"]["selected_record_indices"]:
            current = (0.6, 0.4, 0.0)
            candidate = (0.2, 0.8, 0.0)
            deployed = (
                (0.56, 0.44, 0.0)
                if self.method_id.startswith("fixed_base_exact_")
                else (0.5, 0.5, 0.0)
            )
            if self.non_mixture_deployment:
                # An exact method whose deployed law is not the alpha mixture.
                deployed = (0.50, 0.50, 0.0)
            if self.unnormalized_states:
                # Float32-scale rounding the backend must have removed.
                current = (0.6, 0.4 + 3e-8, 0.0)
            if self.masked_leakage:
                current = (0.6, 0.4, 1e-6)
            state = TheoryStateV2(
                state_id=f"state-{index}",
                record_index=index,
                dataset_record_sha256=_digest(f"record-{index}"),
                registered_state_sha256="",
                action_mask=(True, True, False),
                current_probabilities=current,
                candidate_probabilities=candidate,
                deployed_probabilities=deployed,
            )
            result.append(replace(state, registered_state_sha256=state_identity(state)))
        if self.reorder:
            result.reverse()
        return tuple(result)

    def identity_bundle(self) -> dict[str, Any]:
        return copy.deepcopy(self.identity)

    def read_only_snapshot(self) -> ReadOnlySnapshotV2:
        checkpoint = self.identity["checkpoint"]
        model = self.identity["model"]
        return ReadOnlySnapshotV2(
            checkpoint_sha256=checkpoint["sha256"],
            model_state_sha256=model["model_sha256"],
            mutable_state_sha256s=copy.deepcopy(self.mutable),
            recurrent_transition_sha256=model["recurrent_transition_sha256"],
            snapshot_kind=checkpoint["snapshot_kind"],
            environment_interactions=checkpoint["environment_interactions"],
        )

    def registered_states(self) -> tuple[TheoryStateV2, ...]:
        return self.states

    def normalization_diagnostic(self) -> dict[str, object]:
        return dict(self.normalization)

    def endpoint_value(self, state_id: str, depth: int) -> float:
        index = int(state_id.rsplit("-", 1)[-1]) if state_id.startswith("state-") else 0
        return 0.01 * index + (1.0 if depth == 2 else 1.2)

    def exact_action_outcomes(
        self, state_id: str, action_index: int
    ) -> tuple[TheoryOutcomeV2, ...]:
        return (
            TheoryOutcomeV2(
                probability=1.0,
                reward=float(action_index + 1),
                terminal=True,
                next_state_id=None,
            ),
        )

    def sample_bellman_rollout(
        self, state_id: str, horizon: int, seed: int
    ) -> BellmanRolloutV2:
        return BellmanRolloutV2(
            rewards=(1.0,) * horizon,
            terminal=True,
            bootstrap_state_id=None,
            trajectory_sha256=_digest(f"bellman-{state_id}-{seed}"),
        )

    def training_advantage_estimator(
        self, state_id: str
    ) -> TrainingAdvantageEstimatorV2:
        offset = 0.25 if self.miscentered else 0.0
        return TrainingAdvantageEstimatorV2(
            action_mask=(True, True, False),
            current_probabilities=(0.6, 0.4, 0.0),
            advantages=(-0.4 + offset, 0.6 + offset, 0.0),
            clipping_kind="none",
            clip_value=None,
        )

    def persistent_endpoint_witness(
        self, state_id: str, endpoint_depth: int
    ) -> PersistentEndpointWitnessV2:
        carry = f"carry-{endpoint_depth}" if self.carry_depends_on_m else "carry-F2"
        return PersistentEndpointWitnessV2(
            requested_endpoint_depth=endpoint_depth,
            deployed_transition_depth=(
                endpoint_depth if self.carry_depends_on_m else 2
            ),
            carried_successor_latent_sha256=_digest(carry),
            trajectory_sha256=_digest(f"trajectory-{state_id}"),
            action_probabilities_sha256=_digest(f"probabilities-{state_id}"),
            recurrent_transition_sha256=self.identity["model"][
                "recurrent_transition_sha256"
            ],
        )

    def sample_current_policy_return(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
    ) -> CurrentPolicyReturnV2:
        self.return_calls.append((state_id, seed))
        return CurrentPolicyReturnV2(
            discounted_return=1.5 + 0.1 * (seed % 2),
            environment_steps=2,
            terminal=True,
            trajectory_sha256=_digest(f"return-{state_id}-{seed}"),
        )

    def sample_paired_policy_returns(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
        alpha: float,
    ) -> PairedPolicyReturnV2:
        self.paired_return_calls.append((state_id, seed, alpha))
        current_return = 1.5 + 0.1 * (seed % 2)
        paired_delta = -alpha if seed % 2 == 0 else 2.0 * alpha
        current = CurrentPolicyReturnV2(
            discounted_return=current_return,
            environment_steps=2,
            terminal=True,
            trajectory_sha256=_digest(f"paired-current-{state_id}-{seed}"),
        )
        mixture = CurrentPolicyReturnV2(
            discounted_return=current_return + paired_delta,
            environment_steps=2,
            terminal=True,
            trajectory_sha256=(_digest(f"paired-mixture-{state_id}-{seed}-{alpha}")),
        )
        return PairedPolicyReturnV2(
            current_policy=current,
            exact_mixture=mixture,
            common_random_numbers_sha256=_digest(f"crn-{state_id}-{seed}"),
        )

    def close(self) -> None:
        self.closed = True


class TheoryBridgeV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "checkpoint.pt"
        self.checkpoint.write_bytes(b"synthetic protocol-v2 checkpoint")
        self.amendment = _amendment()
        self.identity = _identity(self.amendment, self.checkpoint)
        self.request = _request(self.amendment, self.identity)

    def test_stage0_request_builder_uses_only_registered_identity(self) -> None:
        row = {
            "run_id": "stage0-run",
            "phase": "stage0_smoke",
            "method_id": "fixed_base_exact_persistent",
            "evaluation_split": "train",
            "evaluation_population": "stage0_smoke",
            "n": 2,
            "K": 1,
            "alpha": 0.1,
        }
        built = build_stage0_theory_request(
            amendment_value=self.amendment,
            row=row,
            identity=self.identity,
            effective_config={"gamma": 0.9, "advantage_clip": None},
        )
        self.assertEqual(built["identity"], self.identity)
        self.assertEqual(built["evaluation_population"], "stage0_smoke")
        self.assertEqual(
            built["advantage_clipping"], {"kind": "none", "clip_value": None}
        )
        clipped = build_stage0_theory_request(
            amendment_value=self.amendment,
            row=row,
            identity=self.identity,
            effective_config={"gamma": 0.9, "advantage_clip": 10.0},
        )
        self.assertEqual(
            clipped["advantage_clipping"],
            {"kind": "clip_then_exact_recenter", "clip_value": 10.0},
        )

    def test_exact_bridge_separates_roundoff_from_trainer_parity(self) -> None:
        backend = _Backend(self.identity)
        result = evaluate_theory_bridge_v2(
            request_document=self.request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=backend,
        )
        self.assertTrue(backend.closed)
        self.assertEqual([row["record_index"] for row in result["states"]], _INDICES)
        self.assertLessEqual(
            result["metrics"]["constructed_centering_roundoff"]["maximum"],
            1e-12,
        )
        self.assertLessEqual(
            result["metrics"]["training_estimator_parity_max_abs_error"]["maximum"],
            1e-12,
        )
        self.assertEqual(
            result["deployment_diagnostic"]["metric"],
            "exact_mixture_deployment_identity_tv",
        )
        self.assertTrue(result["persistent_semantics"]["endpoint_depth_only"])
        self.assertEqual(len(backend.return_calls), 8 * 4)
        self.assertTrue(all(row["E_n"] >= 0.0 for row in result["states"]))

    def test_result_cannot_relax_request_bound_tolerances(self) -> None:
        result = evaluate_theory_bridge_v2(
            request_document=self.request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=_Backend(self.identity),
        )
        cases = (
            (
                "constructed_centering_tolerance",
                self.request["constructed_centering_tolerance"] * 10.0,
                "request-bound centering tolerance",
            ),
            (
                "training_estimator_parity_tolerance",
                self.request["centering_parity_tolerance"] * 10.0,
                "request-bound centering tolerance",
            ),
            (
                "training_estimator_centering_defect",
                self.request["centering_parity_tolerance"] * 2.0,
                "centering defect exceeds request tolerance",
            ),
            (
                "exact_mixture_deployment_identity_tv",
                self.request["deployment_identity_tolerance"] * 2.0,
                "deployment identity exceeds request tolerance",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                corrupted = copy.deepcopy(result)
                corrupted["states"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_theory_result(corrupted, request=self.request)

    def test_crn_seed_ignores_alpha_dependent_deployment_identity(self) -> None:
        state = self._state_for_seed()
        changed = replace(
            state,
            registered_state_sha256=_digest("alpha-dependent-state-identity"),
            deployed_probabilities=(0.52, 0.48, 0.0),
        )
        arguments = {
            "namespace": "upi-trm-policy-improvement-v2-current-return-crn",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": self.identity["protocol_sha256"],
            "checkpoint_sha256": self.identity["checkpoint"]["sha256"],
            "base_seed": 130363,
            "repeat_index": 3,
        }
        self.assertEqual(
            _seed(state=state, **arguments),
            _seed(state=changed, **arguments),
        )

    def _state_for_seed(self) -> TheoryStateV2:
        return _Backend(self.identity).states[0]

    def test_deliberately_miscentered_trainer_estimator_is_rejected(self) -> None:
        backend = _Backend(self.identity, miscentered=True)
        with self.assertRaisesRegex(
            TheoryBridgeV2Error,
            "exact-advantage tensors differ",
        ):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )
        self.assertTrue(backend.closed)

    def test_persistent_F_m_carry_is_rejected(self) -> None:
        backend = _Backend(self.identity, carry_depends_on_m=True)
        with self.assertRaisesRegex(TheoryBridgeV2Error, "changes deployed F_n"):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )
        self.assertTrue(backend.closed)

    def test_explicit_noncontiguous_population_reordering_is_rejected(self) -> None:
        backend = _Backend(self.identity, reorder=True)
        with self.assertRaisesRegex(TheoryBridgeV2Error, "reordered record"):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )
        self.assertTrue(backend.closed)

    def test_realized_policy_reports_discrepancy_not_exact_identity(self) -> None:
        request = _request(
            self.amendment,
            self.identity,
            method_id="legacy_parameter_interpolation",
        )
        backend = _Backend(
            self.identity,
            method_id="legacy_parameter_interpolation",
        )
        result = evaluate_theory_bridge_v2(
            request_document=request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=backend,
        )
        self.assertEqual(
            result["deployment_diagnostic"]["metric"],
            "deployment_discrepancy_delta_dep",
        )
        self.assertTrue(
            all(
                row["exact_mixture_deployment_identity_tv"] is None
                and row["deployment_discrepancy_delta_dep"] is not None
                for row in result["states"]
            )
        )
        self.assertTrue(
            all(
                row["paired_current_to_exact_mixture_return"]
                == {
                    "status": "unavailable",
                    "reason": "stage0_smoke_not_scientific_calibration",
                }
                for row in result["states"]
            )
        )

    def test_validation_bridge_exact_reports_registered_paired_crn_alphas(
        self,
    ) -> None:
        identity = _validation_identity(self.amendment, self.checkpoint)
        request = _validation_request(self.amendment, identity)
        backend = _Backend(identity)
        result = evaluate_theory_bridge_v2(
            request_document=request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=backend,
        )
        self.assertEqual(len(backend.return_calls), 0)
        self.assertEqual(len(backend.paired_return_calls), 128 * 4 * 3)
        first_state_calls = [
            call
            for call in backend.paired_return_calls
            if call[0] == f"state-{_VALIDATION_INDICES[0]}"
        ]
        self.assertEqual(
            {
                alpha: [
                    seed
                    for _, seed, observed_alpha in first_state_calls
                    if observed_alpha == alpha
                ]
                for alpha in (0.05, 0.1, 0.2)
            },
            {
                alpha: [
                    seed
                    for _, seed, observed_alpha in first_state_calls
                    if observed_alpha == 0.05
                ]
                for alpha in (0.05, 0.1, 0.2)
            },
        )
        for row in result["states"]:
            diagnostic = row["paired_current_to_exact_mixture_return"]
            self.assertEqual(diagnostic["status"], "available")
            self.assertEqual(
                [item["alpha"] for item in diagnostic["alpha_results"]],
                [0.05, 0.1, 0.2],
            )
            self.assertEqual(
                len(
                    {
                        item["current_policy_return_mean"]
                        for item in diagnostic["alpha_results"]
                    }
                ),
                1,
            )
            self.assertEqual(
                len(
                    {
                        item["common_random_numbers_sha256"]
                        for item in diagnostic["alpha_results"]
                    }
                ),
                1,
            )
        corrupted = copy.deepcopy(result)
        corrupted_alpha = corrupted["states"][0][
            "paired_current_to_exact_mixture_return"
        ]["alpha_results"][1]
        corrupted_alpha["current_policy_return_mean"] += 0.25
        corrupted_alpha["exact_mixture_return_mean"] += 0.25
        with self.assertRaisesRegex(
            ValueError,
            "current-policy or CRN identity changed by alpha",
        ):
            validate_theory_result(corrupted, request=request)

    def test_validation_bridge_legacy_keeps_paired_return_unavailable(self) -> None:
        identity = _validation_identity(self.amendment, self.checkpoint)
        request = _validation_request(
            self.amendment,
            identity,
            method_id="legacy_parameter_interpolation",
        )
        backend = _Backend(identity, method_id="legacy_parameter_interpolation")
        result = evaluate_theory_bridge_v2(
            request_document=request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=backend,
        )
        self.assertEqual(len(backend.return_calls), 128 * 4)
        self.assertEqual(backend.paired_return_calls, [])
        self.assertTrue(
            all(
                row["paired_current_to_exact_mixture_return"]
                == {
                    "status": "unavailable",
                    "reason": "not_an_exact_probability_mixture_method",
                }
                for row in result["states"]
            )
        )

    def test_v1_request_cannot_enter_v2_evaluator(self) -> None:
        request = copy.deepcopy(self.request)
        request["schema_name"] = "policy_improvement_theory_bridge_request_v1"
        backend = _Backend(self.identity)
        with self.assertRaisesRegex(TheoryBridgeV2Error, "unsupported"):
            evaluate_theory_bridge_v2(
                request_document=request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )
        self.assertTrue(backend.closed)

    def test_stage0_smoke_rejects_validation_and_test_before_backend_access(
        self,
    ) -> None:
        for split in ("validation", "test"):
            with self.subTest(split=split):
                request = copy.deepcopy(self.request)
                request["identity"]["dataset_records"]["split"] = split
                backend = _Backend(self.identity)
                with self.assertRaises(TheoryBridgeV2Error):
                    evaluate_theory_bridge_v2(
                        request_document=request,
                        amendment_document=self.amendment,
                        checkpoint_path=self.checkpoint,
                        backend=backend,
                    )
                self.assertTrue(backend.closed)
                self.assertEqual(backend.return_calls, [])

    def test_sealed_adapter_closes_full_session_once(self) -> None:
        source = _Backend(self.identity)

        class Session:
            def __init__(self) -> None:
                self.close_count = 0

            def identity_bundle(self) -> dict[str, Any]:
                return source.identity_bundle()

            def begin_read_only_evaluation(self) -> None:
                source.begin_read_only_evaluation()

            def end_read_only_evaluation(self) -> None:
                source.end_read_only_evaluation()

            def read_only_snapshot_v2(self) -> SimpleNamespace:
                snapshot = source.read_only_snapshot()
                return SimpleNamespace(**snapshot.__dict__)

            def registered_states(self) -> tuple[TheoryStateV2, ...]:
                return source.registered_states()

            def normalization_diagnostic(self) -> dict[str, Any]:
                return source.normalization_diagnostic()

            def endpoint_value(self, state_id: str, depth: int) -> float:
                return source.endpoint_value(state_id, depth)

            def exact_action_outcomes(
                self, state_id: str, action_index: int
            ) -> tuple[TheoryOutcomeV2, ...]:
                return source.exact_action_outcomes(state_id, action_index)

            def sample_rollout(
                self, state_id: str, horizon: int, seed: int
            ) -> BellmanRolloutV2:
                return source.sample_bellman_rollout(state_id, horizon, seed)

            def training_advantage_estimator(
                self, state_id: str
            ) -> TrainingAdvantageEstimatorV2:
                return source.training_advantage_estimator(state_id)

            def persistent_endpoint_witness(
                self, state_id: str, endpoint_depth: int
            ) -> PersistentEndpointWitnessV2:
                return source.persistent_endpoint_witness(state_id, endpoint_depth)

            def sample_current_policy_return(
                self,
                state_id: str,
                seed: int,
                maximum_environment_steps: int,
                gamma: float,
            ) -> CurrentPolicyReturnV2:
                return source.sample_current_policy_return(
                    state_id,
                    seed,
                    maximum_environment_steps,
                    gamma,
                )

            def close(self) -> None:
                self.close_count += 1

        session = Session()
        backend = SealedTheoryBackendV2(session)
        result = evaluate_theory_bridge_v2(
            request_document=self.request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=backend,
        )
        backend.close()
        self.assertEqual(result["state_count"], 8)
        self.assertEqual(session.close_count, 1)

    def test_v2_main_emits_one_canonical_document_through_injected_backend(
        self,
    ) -> None:
        root = Path(self.temporary.name)
        request_path = root / "request.json"
        amendment_path = root / "theory.json"
        protocol_path = root / "protocol.json"
        registry_path = root / "registry.json"
        request_path.write_bytes(canonical_json_bytes(self.request))
        amendment_path.write_bytes(canonical_json_bytes(self.amendment))
        protocol_path.write_text("{}", encoding="ascii")
        registry_path.write_text("{}", encoding="ascii")
        evidence_root = root / "evidence"
        dataset_root = root / "dataset"
        evidence_root.mkdir()
        dataset_root.mkdir()
        backend = _Backend(self.identity)
        calls: list[Any] = []

        def factory(request: Any, checkpoint: Path, inputs: Any) -> _Backend:
            calls.append((request, checkpoint, inputs))
            return backend

        source = self.identity["evaluator_source"]
        runtime = self.identity["evaluator_runtime"]
        attestation = {
            "source_git_commit": source["git_commit"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "runtime_sha256": runtime["runtime_sha256"],
            "runtime_profile_sha256": runtime["runtime_profile_sha256"],
            "runtime_authorization_sha256": runtime["runtime_authorization_sha256"],
            "launcher_sha256": runtime["launcher_sha256"],
        }
        output = tempfile.TemporaryFile()
        saved_stdout = os.dup(1)
        try:
            os.dup2(output.fileno(), 1)
            status = main(
                [
                    "--policy-improvement-theory-bridge-entrypoint",
                    "--request",
                    str(request_path),
                    "--theory-amendment",
                    str(amendment_path),
                    "--amendment",
                    str(amendment_path),
                    "--checkpoint",
                    str(self.checkpoint),
                    "--project-root",
                    str(root),
                    "--protocol",
                    str(protocol_path),
                    "--registry",
                    str(registry_path),
                    "--evidence-root",
                    str(evidence_root),
                    "--dataset-root",
                    str(dataset_root),
                    "--row-id",
                    self.request["run_id"],
                ],
                backend_factory=factory,
                evaluator_attestation=attestation,
                runtime_authorization={},
            )
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        output.seek(0)
        payload = output.read()
        output.close()
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(payload.endswith(b"\n"))
        document = json.loads(payload)
        self.assertEqual(payload, canonical_json_bytes(document) + b"\n")
        self.assertEqual(document["schema_name"], THEORY_RESULT_SCHEMA_NAME)

    def test_stage0_complete_generation_resolver_transfers_or_closes_descriptor(
        self,
    ) -> None:
        parent_sha256 = _digest("prepare-checkpoint")
        model_sha256 = _digest("model-state")
        roles = {"model": _digest("model-role")}
        theory_model_identity = {
            "model_sha256": model_sha256,
            "model_config_sha256": _digest("model-config"),
            "current_policy_sha256": _digest("current-policy"),
            "candidate_policy_sha256": _digest("candidate-policy"),
            "deployed_policy_sha256": _digest("deployed-policy"),
            "recurrent_transition_sha256": _digest("recurrent-transition"),
        }

        def fixture() -> tuple[
            tempfile.TemporaryDirectory[str],
            Path,
            Any,
            dict[str, Any],
            TheoryBackendInputsV2,
            dict[str, Any],
        ]:
            temporary = tempfile.TemporaryDirectory()
            root = Path(temporary.name)
            generation = root / "publication/runs/run/segments/env_000000032"
            checkpoint = generation / "checkpoints/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"sealed-stage0-checkpoint")
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            sealed = seal_generation_checkpoint(
                generation,
                "checkpoints/checkpoint.pt",
                expected_sha256=checkpoint_sha256,
                expected_size_bytes=checkpoint.stat().st_size,
            )
            context = SimpleNamespace(
                run_root=root / "publication/runs/run",
                row={
                    "run_id": "run",
                    "phase": "stage0_smoke",
                    "tier": "smoke",
                    "evaluation_split": "train",
                    "method_id": "fixed_base_exact_persistent",
                    "base_method_id": "fixed_base_exact_persistent",
                    "n": 2,
                    "K": 1,
                    "alpha": 0.1,
                },
                protocol={},
                registry={},
                protocol_sha256=_digest("protocol"),
                registry_sha256=_digest("registry"),
                registry_row_sha256=_digest("row"),
                source_root=root,
                dataset_root=root,
            )
            request = {
                "identity": {
                    "checkpoint": {
                        "sha256": checkpoint_sha256,
                        "size_bytes": checkpoint.stat().st_size,
                        "snapshot_kind": "smoke_resume",
                        "environment_interactions": 32,
                    },
                    "model": copy.deepcopy(theory_model_identity),
                }
            }
            inputs = TheoryBackendInputsV2(
                project_root=root,
                protocol_path=root / "protocol.json",
                registry_path=root / "registry.json",
                amendment_paths=(root / "theory.json",),
                evidence_root=root,
                dataset_root=root,
                row_id="run",
                runtime_authorization={},
            )
            authenticated = {
                "generation_path": str(generation),
                "sealed_checkpoint": sealed,
                "parent_generation_manifest_sha256": _digest("parent-manifest"),
                "semantic_validations": [
                    {
                        "checkpoint_sha256": parent_sha256,
                        "environment_interactions": 16,
                        "parent_checkpoint_sha256": None,
                    },
                    {
                        "checkpoint_sha256": checkpoint_sha256,
                        "environment_interactions": 32,
                        "parent_checkpoint_sha256": parent_sha256,
                        "model_state_sha256": model_sha256,
                        "role_state_sha256s": roles,
                        "theory_model_identity": copy.deepcopy(theory_model_identity),
                    },
                ],
            }
            return (
                temporary,
                checkpoint,
                context,
                request,
                inputs,
                authenticated,
            )

        temporary, checkpoint, context, request, inputs, authenticated = fixture()
        try:
            descriptor = authenticated["sealed_checkpoint"].descriptor
            result_document = {
                "schema_name": "policy_improvement_result_v2",
                "payload": {"document_marker": "current-envelope"},
            }
            runtime_result = {
                **context.row,
                "status": "complete",
                "amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
            }
            with (
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2._load_stable_json",
                    return_value={"ignored": True},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.validated_result_payload",
                    return_value=(result_document, runtime_result),
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.load_registered_populations",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.bind_v2_result_to_registration"
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2._per_instance_documents",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.authenticate_complete_generation",
                    return_value=authenticated,
                ) as authenticate,
            ):
                resolved = _resolve_stage0_checkpoint(
                    request=request,
                    checkpoint_path=checkpoint,
                    inputs=inputs,
                    authorization={},
                    context=context,
                )
            self.assertIs(authenticate.call_args.kwargs["result"], result_document)
            os.fstat(descriptor)
            resolved.sealed.close()
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        finally:
            temporary.cleanup()

        temporary, checkpoint, context, request, inputs, authenticated = fixture()
        try:
            descriptor = authenticated["sealed_checkpoint"].descriptor
            result_document = {
                "schema_name": "policy_improvement_result_v2",
                "payload": {"document_marker": "current-envelope"},
            }
            with (
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2._load_stable_json",
                    return_value={"ignored": True},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.validated_result_payload",
                    return_value=(
                        result_document,
                        {
                            **context.row,
                            "status": "complete",
                            "amendment_history_sha256": hashlib.sha256(
                                b"[]"
                            ).hexdigest(),
                        },
                    ),
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.load_registered_populations",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.bind_v2_result_to_registration"
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2._per_instance_documents",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.authenticate_complete_generation",
                    return_value=authenticated,
                ),
                self.assertRaisesRegex(TheoryBridgeV2Error, "path or identity"),
            ):
                _resolve_stage0_checkpoint(
                    request=request,
                    checkpoint_path=checkpoint.with_name("wrong.pt"),
                    inputs=inputs,
                    authorization={},
                    context=context,
                )
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        finally:
            temporary.cleanup()

        temporary, checkpoint, context, request, inputs, authenticated = fixture()
        try:
            descriptor = authenticated["sealed_checkpoint"].descriptor
            request["identity"]["model"]["deployed_policy_sha256"] = _digest(
                "mixed-model"
            )
            result_document = {
                "schema_name": "policy_improvement_result_v2",
                "payload": {"document_marker": "current-envelope"},
            }
            with (
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2._load_stable_json",
                    return_value={"ignored": True},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.validated_result_payload",
                    return_value=(
                        result_document,
                        {
                            **context.row,
                            "status": "complete",
                            "amendment_history_sha256": hashlib.sha256(
                                b"[]"
                            ).hexdigest(),
                        },
                    ),
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.load_registered_populations",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.bind_v2_result_to_registration"
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2._per_instance_documents",
                    return_value={},
                ),
                mock.patch(
                    "scripts.policy_improvement_theory_backend_v2.authenticate_complete_generation",
                    return_value=authenticated,
                ),
                self.assertRaisesRegex(
                    TheoryBridgeV2Error,
                    "differs from sealed semantic validation",
                ),
            ):
                _resolve_stage0_checkpoint(
                    request=request,
                    checkpoint_path=checkpoint,
                    inputs=inputs,
                    authorization={},
                    context=context,
                )
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        finally:
            temporary.cleanup()

    def test_stage0_theory_requires_independent_four_row_reaudit(self) -> None:
        root = Path(self.temporary.name)
        publication = root / "publication"
        runs = publication / "runs"
        run_ids = [f"stage0-{index}" for index in range(4)]
        for run_id in run_ids:
            generation = runs / run_id / "segments/env_000000032"
            generation.mkdir(parents=True)
            (generation / "result.json").write_text("{}", encoding="ascii")
        checkpoint = runs / run_ids[0] / "segments/env_000000032/checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        request = {"canonical": True}
        context = SimpleNamespace(
            registry={
                "rows": [
                    {"run_id": run_id, "phase": "stage0_smoke"} for run_id in run_ids
                ]
            },
            protocol={},
            source_root=root,
            dataset_root=root,
            run_root=runs / run_ids[0],
            row={"run_id": run_ids[0]},
        )
        authorization = {
            "launcher_sha256": _digest("launcher"),
            "producer_git_commit": "1" * 40,
            "producer_source_manifest_sha256": _digest("producer"),
        }
        theory_role = {
            "runtime_sha256": _digest("theory runtime"),
            "runtime_profile_sha256": _digest("theory profile"),
            "source_git_commit": "2" * 40,
        }
        report = {
            "expected_rows": 4,
            "complete_rows": 4,
            "failed_rows": 0,
            "historical_failed_attempt_count": 0,
            "per_instance_artifact_count": 10,
            "semantic_checkpoint_validation_count": 8,
            "compute_accounting_artifact_count": 4,
            "validation_data_opened": False,
            "test_data_opened": False,
            "test_open_verified": False,
            "stage0_theory_requests": [
                {
                    "run_id": run_ids[0],
                    "checkpoint_path": str(checkpoint),
                    "request": request,
                },
                {
                    "run_id": run_ids[1],
                    "checkpoint_path": str(checkpoint),
                    "request": {"other": True},
                },
            ],
        }
        document_sets = [
            {
                _digest(f"per-instance-{index}-{offset}"): {"index": offset}
                for offset in range(count)
            }
            for index, count in enumerate((3, 3, 3, 1))
        ]
        with (
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._load_stable_json",
                side_effect=[{"run_id": run_id} for run_id in sorted(run_ids)],
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._per_instance_documents",
                side_effect=document_sets,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._runtime_role",
                return_value=theory_role,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.load_registered_base_configs",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.load_sealed_checkpoint_validator",
                return_value=object(),
            ),
            mock.patch.object(
                policy_improvement_audit,
                "audit_result_set",
                return_value=report,
            ) as audit,
        ):
            _independently_audit_stage0(
                request=request,
                checkpoint_path=checkpoint,
                inputs=TheoryBackendInputsV2(
                    project_root=root,
                    protocol_path=root / "protocol.json",
                    registry_path=root / "registry.json",
                    amendment_paths=(),
                    evidence_root=root,
                    dataset_root=root,
                    row_id=run_ids[0],
                    runtime_authorization=authorization,
                ),
                authorization=authorization,
                context=context,
            )
        self.assertEqual(len(audit.call_args.args[2]), 4)
        self.assertEqual(len(audit.call_args.args[3]), 10)
        self.assertEqual(
            audit.call_args.kwargs["execution_role_name"],
            "policy-improvement-theory-bridge",
        )

    def test_stage0_factory_closes_resolved_descriptor_when_session_open_fails(
        self,
    ) -> None:
        root = Path(self.temporary.name)
        generation = root / "generation"
        checkpoint = generation / "checkpoints/checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"factory-cleanup")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        sealed = seal_generation_checkpoint(
            generation,
            "checkpoints/checkpoint.pt",
            expected_sha256=checkpoint_sha256,
            expected_size_bytes=checkpoint.stat().st_size,
        )
        descriptor = sealed.descriptor
        resolved = _AuthenticatedStage0Checkpoint(
            path=checkpoint,
            sealed=sealed,
            parent_checkpoint_sha256=_digest("parent"),
            model_state_sha256=_digest("model"),
            role_state_sha256s={"model": _digest("model-role")},
        )
        request = {
            "evaluation_population": "stage0_smoke",
            "identity": {},
        }
        inputs = TheoryBackendInputsV2(
            project_root=root,
            protocol_path=root / "protocol.json",
            registry_path=root / "registry.json",
            amendment_paths=(root / "theory.json",),
            evidence_root=root,
            dataset_root=root,
            row_id="run",
            runtime_authorization={},
        )
        with (
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.validate_theory_request",
                return_value=request,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.validate_runtime_authorization",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._stage0_context",
                return_value=(
                    SimpleNamespace(
                        protocol={},
                        source_root=root,
                        row={"base_method_id": "base", "config_override": {}},
                    ),
                    {},
                    {},
                ),
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._expected_stage0_identity",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._load_stable_json",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.load_registered_base_configs",
                return_value={"base": {}},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.build_stage0_theory_request",
                return_value=request,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._independently_audit_stage0"
            ) as reaudit,
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._resolve_stage0_checkpoint",
                return_value=resolved,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.importlib.import_module",
                side_effect=RuntimeError("import rejected"),
            ),
            self.assertRaisesRegex(RuntimeError, "import rejected"),
        ):
            create_theory_bridge_backend_v2(request, checkpoint, inputs)
        reaudit.assert_called_once()
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_stage0_factory_rejects_noncanonical_gamma_before_checkpoint_resolution(
        self,
    ) -> None:
        root = Path(self.temporary.name)
        request = {
            "evaluation_population": "stage0_smoke",
            "gamma": 0.5,
            "identity": {},
        }
        canonical_request = {**request, "gamma": 0.99}
        inputs = TheoryBackendInputsV2(
            project_root=root,
            protocol_path=root / "protocol.json",
            registry_path=root / "registry.json",
            amendment_paths=(root / "theory.json",),
            evidence_root=root,
            dataset_root=root,
            row_id="run",
            runtime_authorization={},
        )
        context = SimpleNamespace(
            protocol={},
            source_root=root,
            row={"base_method_id": "base", "config_override": {}},
        )
        with (
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.validate_theory_request",
                return_value=request,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.validate_runtime_authorization",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._stage0_context",
                return_value=(context, {}, {}),
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._expected_stage0_identity",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._load_stable_json",
                return_value={},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.load_registered_base_configs",
                return_value={"base": {}},
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.build_stage0_theory_request",
                return_value=canonical_request,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2._resolve_stage0_checkpoint"
            ) as resolve,
            self.assertRaisesRegex(
                TheoryBridgeV2Error,
                "differs from the audited canonical request",
            ),
        ):
            create_theory_bridge_backend_v2(request, root / "checkpoint.pt", inputs)
        resolve.assert_not_called()

    def test_validation_factory_remains_fail_closed(self) -> None:
        root = Path(self.temporary.name)
        protocol_path = root / "configs/policy_improvement_v2/protocol.json"
        registry_path = root / "configs/policy_improvement_v2/registry.json"
        theory_path = (
            root / "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
        )
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"authenticated full checkpoint")
        theory = {"schema_name": THEORY_AMENDMENT_SCHEMA_NAME}
        theory_sha256 = theory_document_sha256(theory)
        authorization_sha256 = _digest("authorization")
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v3",
            "producer_git_commit": "1" * 40,
            "producer_source_manifest_sha256": _digest("producer source"),
            "launcher_sha256": _digest("launcher"),
            "protocol_sha256": _digest("protocol"),
            "registry": {"sha256": _digest("registry")},
            "amendments": [
                {
                    "schema_name": THEORY_AMENDMENT_SCHEMA_NAME,
                    "sha256": theory_sha256,
                }
            ],
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "1" * 40,
                    "runtime_sha256": _digest(f"{role} runtime"),
                    "runtime_profile_sha256": _digest(f"{role} profile"),
                    "selected_source_manifest_sha256": _digest(f"{role} profile"),
                }
                for role in (
                    "policy-improvement-full",
                    "policy-improvement-theory-bridge",
                )
            ],
        }
        row = {
            "schema_name": "policy_improvement_registry_row_v2",
            "schema_version": 1,
            "row_kind": "concrete",
            "phase": "stage1_screen",
            "run_id": "validation-run",
            "method_id": "fixed_base_exact_persistent",
            "base_method_id": "fixed_base_exact_persistent",
            "evaluation_split": "validation",
            "evaluation_population": "validation_select",
            "n": 2,
            "K": 1,
            "alpha": 0.1,
            "config_override": {},
            "base_config_canonical_sha256": _digest("base config"),
            "expected_effective_config_sha256": _digest("effective config"),
        }
        population = {
            "population_id": "validation_bridge",
            "split": "validation",
            "count": 128,
        }
        registered_run = SimpleNamespace(
            protocol_sha256=authorization["protocol_sha256"],
            registry_sha256=authorization["registry"]["sha256"],
            amendment_history=(theory, {"schema_name": "compute"}),
            row=row,
            population_document={"populations": {"validation_bridge": population}},
            protocol={"methods": [{"id": row["base_method_id"]}]},
            project_root=root,
        )
        request = {
            "evaluation_population": "validation_bridge",
            "run_id": row["run_id"],
            "method_id": row["method_id"],
            "n": row["n"],
            "bellman_horizon": row["K"],
            "alpha": row["alpha"],
            "reference_depth_m": 8,
            "identity": {
                "checkpoint": {"environment_interactions": 10000},
                "training_runtime": {},
            },
        }
        inputs = TheoryBackendInputsV2(
            project_root=root,
            protocol_path=protocol_path,
            registry_path=registry_path,
            amendment_paths=(theory_path, root / "compute.json"),
            evidence_root=root,
            dataset_root=root,
            row_id=row["run_id"],
            runtime_authorization={},
        )
        with (
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.validate_theory_request",
                return_value=request,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.validate_runtime_authorization",
                return_value=authorization,
            ),
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.load_registered_full_run"
            ) as load_run,
            mock.patch(
                "scripts.policy_improvement_theory_backend_v2.resolve_authenticated_full_checkpoint"
            ) as resolve,
        ):
            with self.assertRaisesRegex(
                TheoryBridgeV2Error,
                "canonical 30-request schedule",
            ):
                create_theory_bridge_backend_v2(request, checkpoint, inputs)
        load_run.assert_not_called()
        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TheoryBridgeV2NormalizationTest(unittest.TestCase):
    """The evaluator keeps its 1e-10 precondition and its mixture check."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "checkpoint.pt"
        self.checkpoint.write_bytes(b"synthetic protocol-v2 checkpoint")
        self.amendment = _amendment()
        self.identity = _identity(self.amendment, self.checkpoint)
        self.request = _request(self.amendment, self.identity)

    def test_float32_rounding_is_rejected_at_the_bridge(self) -> None:
        """The bridge never absorbs rounding; the backend must normalize."""

        backend = _Backend(self.identity, unnormalized_states=True)
        with self.assertRaisesRegex(
            TheoryBridgeV2Error, "does not sum to one on valid actions"
        ):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )

    def test_masked_leakage_is_rejected_at_the_bridge(self) -> None:
        backend = _Backend(self.identity, masked_leakage=True)
        with self.assertRaisesRegex(TheoryBridgeV2Error, "masked action"):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )

    def test_non_mixture_deployment_is_rejected_for_exact_methods(self) -> None:
        """Normalization must not be able to force mixture equality."""

        backend = _Backend(self.identity, non_mixture_deployment=True)
        with self.assertRaisesRegex(
            TheoryBridgeV2Error, "Exact pointwise mixture identity failed"
        ):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )

    def test_normalization_diagnostic_is_reported(self) -> None:
        backend = _Backend(self.identity)
        result = evaluate_theory_bridge_v2(
            request_document=self.request,
            amendment_document=self.amendment,
            checkpoint_path=self.checkpoint,
            backend=backend,
        )
        diagnostic = result["normalization_diagnostic"]
        self.assertEqual(
            diagnostic["kind"],
            "float32_categorical_to_binary64_renormalization",
        )
        self.assertEqual(diagnostic["maximum_valid_mass_error"], 3.2e-8)
        self.assertEqual(diagnostic["maximum_normalization_correction"], 1.1e-9)
        self.assertIs(diagnostic["masked_entries_zeroed_before_validation"], False)
        self.assertIs(diagnostic["deployed_reconstructed_from_mixture"], False)

    def test_dishonest_normalization_diagnostics_fail_closed(self) -> None:
        for override, pattern in (
            ({"masked_entries_zeroed_before_validation": True}, "removed evidence"),
            ({"deployed_reconstructed_from_mixture": True}, "removed evidence"),
            ({"maximum_valid_mass_error": 1.0}, "outside the float32 envelope"),
            ({"maximum_valid_mass_error": -1e-9}, "invalid"),
            ({"normalized_state_count": -1}, "invalid"),
        ):
            with self.subTest(override=override):
                backend = _Backend(self.identity)
                backend.normalization.update(override)
                with self.assertRaisesRegex(TheoryBridgeV2Error, pattern):
                    evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )

        backend = _Backend(self.identity)
        backend.normalization.pop("kind")
        with self.assertRaisesRegex(TheoryBridgeV2Error, "inventory differs"):
            evaluate_theory_bridge_v2(
                request_document=self.request,
                amendment_document=self.amendment,
                checkpoint_path=self.checkpoint,
                backend=backend,
            )
