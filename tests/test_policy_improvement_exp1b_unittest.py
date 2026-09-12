#!/usr/bin/env fbpython
"""Focused tests for the Experiment 1B authenticated vertical slice.

Every fixture is synthetic. No dataset record, checkpoint payload, validation or
test datum, or prior scientific result is read. The census fixtures below use
the *registration metadata* in the committed population document (ordered record
indices and digests), never record contents.

Route fixtures write their documents into a temporary directory and open the
route through the production factory, so the trust anchors under test are the
ones the factory computes from those bytes — not values handed back to it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import unittest
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from scripts.policy_improvement_exp1_diagnostics import (
    Exp1DiagnosticsError,
    PairedEndpointObservation,
    paired_population_summary,
    paired_state_diagnostic,
    PopulationMember,
)
from scripts.policy_improvement_exp1b_aggregate import (
    aggregate_exp1b_result,
    validated_exp1b_document,
    Exp1bAggregateError,
    RESULT_SCHEMA_NAME,
)
from scripts.policy_improvement_exp1b_bootstrap import (
    _interval_for_test,
    BOOTSTRAP_RNG_NAMESPACE,
    BOOTSTRAP_SEED,
    _exact_mean,
    decode_replicate_vector,
    encode_replicate_vector,
    Exp1bBootstrapError,
    paired_seed_percentile_interval,
    replicate_digest,
    REPLICATES,
    resample_plan,
    revalidate_replicate_vector,
    type7_quantile,
    UNITS,
)
from scripts.policy_improvement_exp1b_bridge import (
    census_from_population,
    Exp1bBridgeError,
    Exp1bBridgeRoute,
    Exp1bCensus,
    Exp1bRuntimeAttestation,
    Exp1bSealedOctet,
    Exp1bSeedPayload,
    open_authenticated_exp1b_route,
    stable_file_digest,
)
import phase4_runtime_profile as profile_module
import scripts.policy_improvement_exp1b_runtime as exp1b_runtime
from scripts.policy_improvement_exp1b_theory_backend import (
    create_exp1b_theory_backend,
    Exp1bTheoryBackend,
    Exp1bTheoryBackendError,
)
from scripts.policy_improvement_exp1b_schema import (
    ADMISSIBLE_SLOTS,
    apply_exp1b_admission,
    BRIDGE_ROUTE_ID,
    EXECUTION_ENVIRONMENT_VARIABLE,
    EXECUTION_PURPOSE,
    CENTERING_PARITY_KIND,
    CENTERING_PARITY_TOLERANCE,
    DEPLOYMENT_MISMATCH_KIND,
    TRAINER_RECONSTRUCTION_CONTRACT,
    exp1b_state_id_for,
    FULL_ROLE,
    GAMMA,
    MIXTURE_IDENTITY_KIND,
    MIXTURE_IDENTITY_TOLERANCE,
    PERSISTENT_STATE_KIND,
    SECONDARY_DIAGNOSTIC_NAMES,
    TARGET_LAG_KIND,
    validate_secondary_diagnostics,
    THEORY_BRIDGE_ROLE,
    validate_exp1b_admission,
    DEPLOYED_DEPTH_N,
    MIXTURE_ALPHA,
    Exp1bRequestSchedule,
    Exp1bSchemaError,
    Exp1bScheduleStateError,
    exp1b_document_sha256,
    PROTOCOL_ID,
    REFERENCE_DEPTH_M,
    REGISTERED_SEEDS,
    registry_rows_by_run_id,
    TERMINAL_ENVIRONMENT_INTERACTIONS,
    TRAINING_RECORD_COUNT,
    validate_exp1b_amendment,
    validate_exp1b_protocol,
    validate_exp1b_registry,
    validate_substitute_provenance,
)
from scripts.policy_improvement_exp1b_evidence import (
    authenticate_base_artifact,
    authenticated_checkpoint_bytes,
    authenticate_sealed_octet_checkpoints,
    Exp1bEvidenceError,
    finalize_exp1b_evidence,
    open_evidence_generation,
    stable_bytes,
)
from scripts.policy_improvement_exp1b_session import (
    build_exp1b_training_session,
    Exp1bSessionError,
    Exp1bTrainingRegistration,
    LoadedTrainSplit,
    register_exp1b_training,
    TrainOnlyDatasetGuard,
)
from scripts.policy_improvement_exp1b_auditor import (
    audit_exp1b_result_document,
    Exp1bAuditError,
)
from scripts.policy_improvement_schema import canonical_json_bytes

_ROOT = Path(__file__).resolve().parent.parent
_EXP1B = _ROOT / "configs" / "policy_improvement_exp1b"
_V2 = _ROOT / "configs" / "policy_improvement_v2"

#: The frozen production namespace. Tests use it directly so the golden values
#: below are the ones the study will actually produce.
_NAMESPACE = BOOTSTRAP_RNG_NAMESPACE
#: A deliberately different namespace, used only to prove plan sensitivity.
_OTHER_NAMESPACE = "upi-trm-exp1b-not-the-approved-namespace"

#: Synthetic opaque artifact bytes. Nothing deserializes them; the evidence
#: layer authenticates size and SHA-256 only, which is exactly what a test
#: should be able to exercise without a real checkpoint.
_CHECKPOINT_BYTES = tuple(
    f"exp1b-synthetic-checkpoint-{offset}".encode("ascii") * (16 + offset)
    for offset in range(UNITS)
)
_BASE_ARTIFACT_BYTES = b"exp1b-synthetic-base-policy-artifact" * 32

_COMMIT = "a" * 40
_RUNTIME = "b" * 64
_LAUNCHER = "c" * 64
_AUTHORIZATION = "d" * 64


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attestation(**overrides: str) -> Exp1bRuntimeAttestation:
    """The **producer** (full-PAR) attestation, unless overridden."""

    values = {
        "source_git_commit": _COMMIT,
        "runtime_sha256": _RUNTIME,
        "launcher_sha256": _LAUNCHER,
        "runtime_authorization_sha256": _AUTHORIZATION,
    }
    values.update(overrides)
    return Exp1bRuntimeAttestation(**values)


def _theory_attestation(**overrides: str) -> Exp1bRuntimeAttestation:
    """The **evaluator** (theory-bridge PAR) attestation.

    A genuinely different runtime, launcher, and authorization. The admission
    schema requires the two roles' runtime digests to differ, so a fixture that
    used one identity for both steps could not exercise the production split --
    which is exactly how the cross-role defect survived the previous review.
    """

    values = {
        "source_git_commit": _COMMIT,
        "runtime_sha256": _THEORY_RUNTIME,
        "launcher_sha256": _THEORY_LAUNCHER,
        "runtime_authorization_sha256": _THEORY_AUTHORIZATION,
    }
    values.update(overrides)
    return Exp1bRuntimeAttestation(**values)


def _authorizations() -> tuple[Any, Any]:
    from scripts.policy_improvement_exp1b_schema import Exp1bRoleAuthorization

    return (
        Exp1bRoleAuthorization(
            role=FULL_ROLE,
            runtime_authorization_sha256=_AUTHORIZATION,
            launcher_sha256=_LAUNCHER,
            runtime_sha256=_RUNTIME,
            source_git_commit=_COMMIT,
        ),
        Exp1bRoleAuthorization(
            role=THEORY_BRIDGE_ROLE,
            runtime_authorization_sha256=_THEORY_AUTHORIZATION,
            launcher_sha256=_THEORY_LAUNCHER,
            runtime_sha256=_THEORY_RUNTIME,
            source_git_commit=_COMMIT,
        ),
    )


def _state_id_for(index: int, digest: str) -> str:
    """The registered derivation, not a shorthand.

    The fixtures used ``f"bridge-{index}"``, which made the census ordering
    digest a fixture-local value. The independent auditor rederives that digest
    from the registered population with ``exp1b_state_id_for``, so a fixture that
    opens its route under a different derivation is not exercising the
    production census at all.
    """

    return exp1b_state_id_for(index, digest)


#: The synthetic owner-signed admission. Two distinct runtime digests, one per
#: role: Stage A runs the full PAR and Stage B the theory-bridge PAR, and the
#: admission has to authorize each separately.
_THEORY_RUNTIME = "e" * 64
_THEORY_LAUNCHER = "f" * 64
_THEORY_AUTHORIZATION = "9" * 64
#: A synthetic stand-in for the signed Experiment 0 base-policy amendment. Its
#: canonical digest is what the admission binds, so the two must agree.
_BASE_AMENDMENT_DOCUMENT = {
    "schema_name": "policy_improvement_base_policy_amendment_v2",
    "schema_version": 1,
    "amendment_id": "exp0-base-policy-admission",
    # The adopt-only retry path reads the producer identity from here rather
    # than from a Torch-constructed AuthenticatedBasePolicy.
    "base_policy_artifact": {
        "architecture_sha256": _digest("base-architecture"),
        "producer_git_commit": "c" * 40,
        "producer_source_manifest_sha256": _digest("base-manifest"),
        "training_procedure_sha256": _digest("base-procedure"),
        "training_split_ordered_record_sha256": _digest("base-train-data"),
    },
}
_BASE_AMENDMENT_SHA = exp1b_document_sha256(_BASE_AMENDMENT_DOCUMENT)
_BASE_MODEL_STATE_SHA = _digest("exp1b-base-policy-model-state")


def _admission_document(
    *,
    protocol_sha256: str,
    registry_sha256: str,
    prior_amendment_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_name": "policy_improvement_exp1b_admission_v1",
        "schema_version": 1,
        "amendment_id": "exp1b-execution-admission",
        "created_at_utc": "2026-09-09T00:00:00Z",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "registry_sha256": registry_sha256,
        "prior_amendment_sha256": prior_amendment_sha256,
        "purpose": EXECUTION_PURPOSE,
        "execution_environment": {
            "environment_variable": EXECUTION_ENVIRONMENT_VARIABLE,
            "required_value": "1",
        },
        "resolved_slots": list(ADMISSIBLE_SLOTS),
        "base_policy_artifact": {
            "status": "available",
            "value": {
                "amendment_sha256": _BASE_AMENDMENT_SHA,
                "checkpoint_sha256": hashlib.sha256(
                    _BASE_ARTIFACT_BYTES
                ).hexdigest(),
                "model_state_sha256": _BASE_MODEL_STATE_SHA,
            },
        },
        "runtime_authorization": {
            "status": "available",
            "value": {
                FULL_ROLE: {
                    "authorization_sha256": _AUTHORIZATION,
                    "launcher_sha256": _LAUNCHER,
                    "runtime_sha256": _RUNTIME,
                    "source_git_commit": _COMMIT,
                },
                THEORY_BRIDGE_ROLE: {
                    "authorization_sha256": _THEORY_AUTHORIZATION,
                    "launcher_sha256": _THEORY_LAUNCHER,
                    "runtime_sha256": _THEORY_RUNTIME,
                    "source_git_commit": _COMMIT,
                },
            },
        },
        "execution_allowed": True,
    }


def _base_descriptor() -> dict[str, Any]:
    """The authenticated base-policy descriptor a real Stage A would hand over."""

    return {
        "initialization_kind": "train_only_pretrained_base_policy",
        "amendment_sha256": _BASE_AMENDMENT_SHA,
        "model_state_sha256": _BASE_MODEL_STATE_SHA,
        "architecture_sha256": _digest("base-architecture"),
        "producer_git_commit": "c" * 40,
        "producer_source_manifest_sha256": _digest("base-manifest"),
        "training_data_sha256": _digest("base-train-data"),
        "training_procedure_sha256": _digest("base-procedure"),
        "shared_across_seeds": True,
        "not_selected_by_validation_or_test": True,
    }


class _RouteFixture:
    """Materialise the evidence a real Stage A + finalization pass would leave.

    The provenance and generation-local base artifact are produced by the
    production finalizer, not authored here: a fixture that hand-wrote them
    would test a document shape nothing in the repository creates.
    """

    def __init__(self, directory: Path, **provenance_overrides: Any) -> None:
        self.directory = directory
        self.protocol = _load(_EXP1B / "protocol.json")
        self.registry = _load(_EXP1B / "registry.json")
        self.amendment = _load(_EXP1B / "amendments" / "reduced_study_exp1b.json")
        self.protocol_sha = exp1b_document_sha256(self.protocol)
        self.registry_sha = exp1b_document_sha256(self.registry)
        self.amendment_sha = exp1b_document_sha256(self.amendment)
        self.admission_document = _admission_document(
            protocol_sha256=self.protocol_sha,
            registry_sha256=self.registry_sha,
            prior_amendment_sha256=self.amendment_sha,
        )
        self.admission_sha = exp1b_document_sha256(self.admission_document)

        population = _load(_V2 / "populations.json")["populations"]["validation_bridge"]
        self.census = census_from_population(population, state_id_for=_state_id_for)

        registered_config = self.protocol["effective_config"]
        registered_training = self.protocol["training_population"]
        self.effective_config_sha256 = registered_config["effective_config_sha256"]
        self.train_ordered_record_sha256 = registered_training[
            "ordered_record_sha256"
        ]
        self.rows = self.registry["rows"]
        self._write(provenance_overrides)

    # -- layout --------------------------------------------------------------

    def _write_json(self, name: str, document: Any) -> Path:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(document) + b"\n")
        return path

    def _run_manifest(self, offset: int, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = _CHECKPOINT_BYTES[offset]
        return {
            "schema_name": "policy_improvement_exp1b_run_manifest_v1",
            "schema_version": 1,
            "run_id": row["run_id"],
            "seed": row["seed"],
            "seed_position": row["seed_position"],
            "applied_seed": row["seed"],
            "environment_interactions": 10000,
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_size_bytes": len(payload),
            "model_state_sha256": _digest(f"exp1b-model-{row['seed']}"),
            "effective_config_sha256": self.effective_config_sha256,
            "train_split_ordered_record_sha256": self.train_ordered_record_sha256,
            "train_record_count": TRAINING_RECORD_COUNT,
            "resolved_evaluation_data": False,
            "initialization_kind": "train_only_pretrained_base_policy",
            "initialization_artifact_sha256": hashlib.sha256(
                _BASE_ARTIFACT_BYTES
            ).hexdigest(),
            "restored_base_model_state_sha256": _BASE_MODEL_STATE_SHA,
            "admission_sha256": self.admission_sha,
            "producer_attestation": {
                "role": FULL_ROLE,
                "launcher_sha256": _LAUNCHER,
                "runtime_authorization_sha256": _AUTHORIZATION,
                "runtime_sha256": _RUNTIME,
                "source_git_commit": _COMMIT,
            },
            "counters": {
                "environment_interactions": 10000,
                "train_step_count": 100 + offset,
                "value_optimizer_step_count": 100 + offset,
                "policy_optimizer_step_count": 50 + offset,
                "distill_optimizer_step_count": 0,
                "puzzle_optimizer_step_count": 0,
                "exact_centering_batch_count": 10 + offset,
            },
        }

    def _write(self, provenance_overrides: Mapping[str, Any]) -> None:
        self.protocol_path = self._write_json("exp1b_protocol.json", self.protocol)
        self.registry_path = self._write_json("exp1b_registry.json", self.registry)
        self.amendment_path = self._write_json("exp1b_amendment.json", self.amendment)
        self.parent_protocol_path = self._write_json(
            "v2_protocol.json", _load(_V2 / "protocol.json")
        )
        self.parent_registry_path = self._write_json(
            "v2_registry.json", _load(_V2 / "registry.json")
        )
        self.parent_population_path = self._write_json(
            "v2_populations.json", _load(_V2 / "populations.json")
        )
        self.parent_theory_path = self._write_json(
            "v2_theory.json", _load(_V2 / "amendments" / "theory_bridge_v2.json")
        )
        # The signed admission lives outside any source checkout, like the real
        # one has to.
        self.owner_directory = self.directory / "owner"
        self.owner_directory.mkdir(parents=True, exist_ok=True)
        self.admission_path = self.owner_directory / "admission.json"
        self.admission_path.write_bytes(
            canonical_json_bytes(self.admission_document) + b"\n"
        )
        self.admitted_base_path = self.owner_directory / "base_policy.pt"
        self.admitted_base_path.write_bytes(_BASE_ARTIFACT_BYTES)

        self.evidence_root = self.directory / "evidence"
        self.generation_id = "gen-0001"
        self.generation_directory = self.evidence_root / self.generation_id
        (self.generation_directory / "payloads").mkdir(parents=True, exist_ok=True)
        (self.generation_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        generation = open_evidence_generation(
            evidence_root=self.evidence_root, generation_id=self.generation_id
        )
        # Stage A publication, through the production writer.
        self.run_manifests = []
        for offset, row in enumerate(self.rows):
            manifest = self._run_manifest(offset, row)
            self.run_manifests.append(manifest)
            generation.publish_checkpoint(
                seed_position=offset,
                checkpoint_bytes=_CHECKPOINT_BYTES[offset],
                run_manifest=manifest,
            )
        # Evidence finalization, through the production finalizer.
        producer_authorization, _evaluator_authorization = _authorizations()
        finalize_exp1b_evidence(
            generation=generation,
            producer_authorization=producer_authorization,
            admitted_base_artifact=self.admitted_base_path,
            admission_sha256=self.admission_sha,
            base_descriptor=_base_descriptor(),
            registry_rows=registry_rows_by_run_id(self.registry),
            exp1b_protocol_sha256=self.protocol_sha,
            exp1b_registry_sha256=self.registry_sha,
            exp1b_amendment_sha256=self.amendment_sha,
            parent=dict(self.protocol["parent"]),
            ordered_population_sha256=self.census.ordered_record_sha256,
            population_binding_sha256=self.census.binding_sha256,
            attestation={
                "source_git_commit": _COMMIT,
                "runtime_sha256": _RUNTIME,
                "launcher_sha256": _LAUNCHER,
                "runtime_authorization_sha256": _AUTHORIZATION,
            },
        )
        self.provenance_path = generation.provenance_path
        self.provenance = _load(self.provenance_path)
        self.sealed = list(self.provenance["sealed_checkpoints"])
        if provenance_overrides:
            # Adversarial cases only: re-publish the finalized provenance with
            # named fields replaced, to exercise the route's own checks.
            self.provenance.update(provenance_overrides)
            self.provenance_path.unlink()
            self.provenance_path.write_bytes(
                canonical_json_bytes(self.provenance) + b"\n"
            )
        self.schedule_state_path = generation.schedule_path

    def rewrite_provenance(self, **overrides: Any) -> None:
        """Adversarial cases only: republish the finalized provenance edited."""

        self.provenance.update(overrides)
        self.provenance_path.unlink()
        self.provenance_path.write_bytes(
            canonical_json_bytes(self.provenance) + b"\n"
        )

    def generation(self) -> Any:
        return open_evidence_generation(
            evidence_root=self.evidence_root, generation_id=self.generation_id
        )

    def open(self, *, attestation: Exp1bRuntimeAttestation | None = None) -> Any:
        producer_authorization, evaluator_authorization = _authorizations()
        return open_authenticated_exp1b_route(
            producer_authorization=producer_authorization,
            evaluator_authorization=evaluator_authorization,
            exp1b_protocol_path=self.protocol_path,
            exp1b_registry_path=self.registry_path,
            exp1b_amendment_path=self.amendment_path,
            parent_protocol_path=self.parent_protocol_path,
            parent_registry_path=self.parent_registry_path,
            parent_population_path=self.parent_population_path,
            parent_theory_amendment_path=self.parent_theory_path,
            evidence_root=self.evidence_root,
            generation_id=self.generation_id,
            attestation=(
                attestation if attestation is not None else _theory_attestation()
            ),
        )


def _evaluator(step: int, *, sign: float = 1.0, flat: float | None = None) -> dict[str, Any]:
    """A deterministic synthetic evaluator seam for one seed.

    Constructed so the three population maxima land at *different* states and
    are all nonzero, which is what makes the separate-maxima assertions bite.
    """

    def parts(state_id: str) -> tuple[float, float, float]:
        # Production identifiers are `exp1b-state-<sha256>`, not `bridge-<n>`,
        # so the synthetic index comes from the identifier's own digest. Still
        # deterministic, and still spread across the residues the separate-maxima
        # assertions rely on.
        index = int(hashlib.sha256(state_id.encode("ascii")).hexdigest()[:8], 16)
        if flat is not None:
            return flat, 0.0, 0.0
        discrepancy = (index % 7) * 0.5 + step
        residual_n = ((index % 5) * 0.25 + step * 0.1) * sign
        residual_m = (index % 3) * 0.125 + step * 0.05
        return discrepancy, residual_n, residual_m

    def endpoint_values(state_id: str, depth: int) -> float:
        discrepancy, _, _ = parts(state_id)
        return 0.0 if depth == DEPLOYED_DEPTH_N else discrepancy

    def action_values(state_id: str, depth: int) -> tuple[float, float]:
        discrepancy, residual_n, residual_m = parts(state_id)
        if depth == DEPLOYED_DEPTH_N:
            operator = 0.0 - residual_n
        else:
            operator = discrepancy - residual_m
        return (operator - 1.0, operator + 1.0)

    return {
        "endpoint_values": endpoint_values,
        "action_values": action_values,
        "action_mask": lambda state_id: (True, True),
        "base_probabilities": lambda state_id: (0.5, 0.5),
        "secondary_diagnostics": _secondary_diagnostics(),
    }


def _reissue(octet: Any, **changes: Any) -> Any:
    """Rebuild an octet with substituted fields, for adversarial tests only.

    Production code cannot do this: ``Exp1bSealedOctet`` requires a module
    private issuance token, so only a completed authenticated route mints one.
    The tests reach for the token deliberately, to prove that the *content*
    checks downstream of issuance still bite.
    """

    from scripts.policy_improvement_exp1b_bridge import (
        _ISSUED_OCTETS,
        _OCTET_ISSUANCE,
    )

    copy = replace(octet, _issued_by=_OCTET_ISSUANCE, **changes)
    # Register the copy so the *content* checks downstream of issuance are the
    # ones under test. The issuance gate itself is covered separately by
    # Exp1bPublicationBindingTest.test_a_replaced_octet_cannot_aggregate, which
    # uses a plain replace and asserts the gate fires first.
    _ISSUED_OCTETS[id(copy)] = copy
    return copy


@lru_cache(maxsize=1)
def _parent_populations() -> Any:
    """The protocol-pinned parent population registry the auditor rederives from."""

    return _load(_V2 / "populations.json")


@lru_cache(maxsize=1)
def _canonical_census() -> Any:
    """The registered validation_bridge census, built exactly as the route does."""

    return census_from_population(
        _load(_V2 / "populations.json")["populations"]["validation_bridge"],
        state_id_for=exp1b_state_id_for,
    )


def _member(index: int) -> str:
    """A real census member identifier, by position.

    The fixtures used ``bridge-<n>`` witness identifiers, which resolve to no
    census member at all. Nothing caught it: the schema checks the six witness
    fields as nonempty strings, the route did not compare them with its census,
    and the auditor checked four of the six for shape only. A real publication
    completed with all six seed-0 witnesses pointing at nothing.
    """

    return _canonical_census().members[index].state_id


def _secondary_diagnostics(**overrides: Any) -> dict[str, Any]:
    """A conforming set of the five secondary records, for Torch-free fixtures.

    The real ones come from the Torch adapter; these have the same shape and the
    same tolerances so the payload, aggregate, and audit layers are exercised
    exactly as they will be in production.
    """

    document = {
        "schema_name": "policy_improvement_exp1b_secondary_diagnostics_v1",
        "schema_version": 1,
        "target_lag": {
            "kind": TARGET_LAG_KIND,
            "maximum_absolute_target_lag": 0.125,
            "target_lag_witness_state_id": _member(3),
            "deployed_depth": DEPLOYED_DEPTH_N,
            "reference_depth": REFERENCE_DEPTH_M,
            "target_ema_tau": 0.99,
            "state_count": 128,
            "folded_into_signed_gap": False,
        },
        "centering_parity": {
            "kind": CENTERING_PARITY_KIND,
            "constructed_centering_roundoff": 1.1e-9,
            "constructed_centering_tolerance": CENTERING_PARITY_TOLERANCE,
            "centering_witness_state_id": _member(7),
            "centering_scheme": "exact_statewise",
            "state_count": 128,
            "folded_into_signed_gap": False,
            "training_estimator_centering_defect": 8.0e-10,
            "training_estimator_parity_max_abs_error": 3.0e-9,
            "training_estimator_parity_tolerance": CENTERING_PARITY_TOLERANCE,
            "parity_witness_state_id": _member(9),
            "centering_defect_witness_state_id": _member(4),
            "clipping_kind": "clip_then_exact_recenter",
            "clip_value": 10.0,
            "trainer_reconstruction": TRAINER_RECONSTRUCTION_CONTRACT,
        },
        "mixture_identity": {
            "kind": MIXTURE_IDENTITY_KIND,
            "maximum_identity_total_variation": 4.0e-9,
            "identity_tolerance": MIXTURE_IDENTITY_TOLERANCE,
            "identity_witness_state_id": _member(11),
            "mixture_alpha": 0.1,
            "policy_epsilon": 0.0,
            "identity_holds": True,
            "state_count": 128,
            "deployed_reconstructed_from_mixture": False,
            "operator_used_base_probabilities": True,
        },
        "deployment_mismatch": {
            "kind": DEPLOYMENT_MISMATCH_KIND,
            "maximum_candidate_base_total_variation": 0.25,
            "mismatch_witness_state_id": _member(2),
            "maximum_normalization_mass_error": 2.0e-8,
            "state_count": 128,
            "folded_into_signed_gap": False,
        },
        "persistent_state": {
            "kind": PERSISTENT_STATE_KIND,
            "latent_mode": "persistent",
            "deployed_transition_depth": DEPLOYED_DEPTH_N,
            "endpoint_depths": [DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M],
            "carried_successor_latent_sha256": _digest("carried-latents"),
            "action_probabilities_sha256": _digest("action-laws"),
            "states_with_carried_latent": 128,
            "state_count": 128,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(document.get(key), Mapping):
            document[key] = {**document[key], **value}
        else:
            document[key] = value
    return document


def _gap_evaluator(target_gap: float) -> dict[str, Any]:
    """An evaluator whose population produces exactly ``target_gap``.

    ``G = eps_n/(1-gamma) - (D_hat + eps_m/(1-gamma))``. A positive target uses
    the residual term with a zero discrepancy; a negative target uses the
    discrepancy term with zero residuals. Both go through the real arithmetic,
    so this is an end-to-end aggregate rather than a substituted record.
    """

    denominator = 1.0 - GAMMA
    if target_gap >= 0.0:
        # G = eps_n/(1-gamma) with D_hat = eps_m = 0.
        residual_n = target_gap * denominator
        discrepancy = 0.0
    else:
        # G = -D_hat with eps_n = eps_m = 0.
        residual_n = 0.0
        discrepancy = -target_gap

    def endpoint_values(state_id: str, depth: int) -> float:
        return 0.0 if depth == DEPLOYED_DEPTH_N else discrepancy

    def action_values(state_id: str, depth: int) -> tuple[float, float]:
        operator = -residual_n if depth == DEPLOYED_DEPTH_N else discrepancy
        return (operator, operator)

    return {
        "endpoint_values": endpoint_values,
        "action_values": action_values,
        "action_mask": lambda state_id: (True, True),
        "base_probabilities": lambda state_id: (0.5, 0.5),
        "secondary_diagnostics": _secondary_diagnostics(),
    }


def _serve_all(route: Any, fixture: _RouteFixture, **evaluator_kwargs: Any) -> None:
    for offset, item in enumerate(fixture.sealed):
        route.serve(
            seed_position=item["seed_position"],
            seed=item["seed"],
            run_id=item["run_id"],
            checkpoint_sha256=item["checkpoint_sha256"],
            evaluation_population="validation_bridge",
            **_evaluator(offset, **evaluator_kwargs),
        )


class Exp1bNamespaceTest(unittest.TestCase):
    """The committed namespace validates and binds the frozen v2 documents."""

    def test_committed_documents_validate_and_cross_bind(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        registry = _load(_EXP1B / "registry.json")
        amendment = _load(_EXP1B / "amendments" / "reduced_study_exp1b.json")
        protocol_sha = exp1b_document_sha256(protocol)
        registry_sha = exp1b_document_sha256(registry)

        validate_exp1b_protocol(protocol)
        validate_exp1b_registry(registry, protocol_sha256=protocol_sha)
        validate_exp1b_amendment(
            amendment, protocol_sha256=protocol_sha, registry_sha256=registry_sha
        )

    def test_parent_digests_match_the_committed_v2_documents(self) -> None:
        """The 1B namespace cites v2 by digest; a v2 edit must break the cite."""

        protocol = _load(_EXP1B / "protocol.json")
        parent = protocol["parent"]
        for field, path in (
            ("protocol_sha256", _V2 / "protocol.json"),
            ("registry_sha256", _V2 / "registry.json"),
            ("population_registry_sha256", _V2 / "populations.json"),
            (
                "theory_amendment_sha256",
                _V2 / "amendments" / "theory_bridge_v2.json",
            ),
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    parent[field], exp1b_document_sha256(_load(path))
                )

    def test_seeds_are_the_frozen_v2_confirmatory_values(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        v2 = _load(_V2 / "protocol.json")
        self.assertEqual(protocol["seeds"]["registered"], v2["seeds"]["confirmatory"])
        self.assertEqual(tuple(v2["seeds"]["confirmatory"]), REGISTERED_SEEDS)
        self.assertEqual(protocol["seeds"]["derivation"], v2["seeds"]["derivation"])
        self.assertEqual(protocol["seeds"]["namespace"], v2["seeds"]["namespace"])

    def test_population_binding_matches_the_registered_census(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        registered = _load(_V2 / "populations.json")["populations"]["validation_bridge"]
        block = protocol["evaluation_population"]
        self.assertEqual(block["ordered_record_sha256"], registered["ordered_record_sha256"])
        self.assertEqual(block["binding_sha256"], registered["binding_sha256"])
        self.assertEqual(block["count"], 128)
        self.assertEqual(block["selection_use"], "none")

    def test_namespace_ships_fail_closed_on_the_execution_gates(self) -> None:
        """The analysis gate is open; the two execution gates are not."""

        protocol = _load(_EXP1B / "protocol.json")
        self.assertEqual(protocol["base_policy_artifact"]["status"], "unavailable")
        self.assertEqual(protocol["runtime_authorization"]["status"], "unavailable")
        self.assertIs(protocol["execution_gate"]["execution_allowed"], False)
        self.assertIs(protocol["bridge_route"]["v2_route_unchanged"], True)
        self.assertEqual(protocol["amendments"], [])

    def test_protocol_registers_the_approved_rng_namespace(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        slot = protocol["analysis"]["rng_namespace"]
        self.assertEqual(slot, {"status": "available", "value": BOOTSTRAP_RNG_NAMESPACE})
        self.assertEqual(slot["value"], "upi-trm-exp1b-seed-bootstrap-v1")
        self.assertIs(protocol["analysis"]["interval_emission_allowed"], True)

    def test_a_wrong_or_regressed_namespace_slot_is_rejected(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        for label, slot in (
            ("wrong value", {"status": "available", "value": "something-else"}),
            ("regressed to reason", {"status": "unavailable", "reason": "x"}),
            ("bare string", "upi-trm-exp1b-seed-bootstrap-v1"),
            ("empty value", {"status": "available", "value": ""}),
            (
                "case changed",
                {"status": "available", "value": "UPI-TRM-EXP1B-SEED-BOOTSTRAP-V1"},
            ),
        ):
            with self.subTest(label=label):
                broken = json.loads(json.dumps(protocol))
                broken["analysis"]["rng_namespace"] = slot
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_protocol(broken)
        with self.subTest("emission disabled"):
            broken = json.loads(json.dumps(protocol))
            broken["analysis"]["interval_emission_allowed"] = False
            with self.assertRaisesRegex(Exp1bSchemaError, "must be exactly true"):
                validate_exp1b_protocol(broken)

    def test_retired_unavailable_reason_is_rejected(self) -> None:
        """The namespace slot's old reason must not reappear anywhere."""

        protocol = _load(_EXP1B / "protocol.json")
        broken = json.loads(json.dumps(protocol))
        broken["base_policy_artifact"] = {
            "status": "unavailable",
            "reason": "bootstrap_rng_namespace_literal_not_supplied",
        }
        with self.assertRaises(Exp1bSchemaError):
            validate_exp1b_protocol(broken)

    def test_registry_is_eight_rows_in_registered_seed_order(self) -> None:
        registry = _load(_EXP1B / "registry.json")
        self.assertEqual(registry["row_count"], 8)
        self.assertEqual(
            tuple(row["seed"] for row in registry["rows"]), REGISTERED_SEEDS
        )
        self.assertEqual(
            tuple(row["seed_position"] for row in registry["rows"]), tuple(range(8))
        )

    def test_malformed_namespace_documents_are_rejected(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        for field, value in (
            ("terminal_environment_interactions", 20000),
            ("reference_depth_m", 16),
            ("gamma", 0.9),
            ("method_id", "matched_ppo"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_protocol({**protocol, field: value})
        with self.subTest("selection use"):
            broken = json.loads(json.dumps(protocol))
            broken["evaluation_population"]["selection_use"] = "stage1"
            with self.assertRaises(Exp1bSchemaError):
                validate_exp1b_protocol(broken)
        with self.subTest("test access"):
            broken = json.loads(json.dumps(protocol))
            broken["bridge_route"]["test_access"] = True
            with self.assertRaises(Exp1bSchemaError):
                validate_exp1b_protocol(broken)
        with self.subTest("seeds reordered"):
            broken = json.loads(json.dumps(protocol))
            broken["seeds"]["registered"] = list(reversed(REGISTERED_SEEDS))
            with self.assertRaises(Exp1bSchemaError):
                validate_exp1b_protocol(broken)


class Exp1bBootstrapTest(unittest.TestCase):
    """D2 and D3: the frozen resample plan and interval convention."""

    def test_plan_shape_and_determinism(self) -> None:
        plan = resample_plan(namespace=_NAMESPACE)
        self.assertEqual(len(plan), REPLICATES)
        self.assertTrue(all(len(row) == UNITS for row in plan))
        self.assertTrue(all(0 <= index < 8 for row in plan for index in row))
        self.assertEqual(plan, resample_plan(namespace=_NAMESPACE))

    def test_approved_namespace_literal_is_exact(self) -> None:
        """The owner-approved literal, pinned byte for byte."""

        self.assertEqual(BOOTSTRAP_RNG_NAMESPACE, "upi-trm-exp1b-seed-bootstrap-v1")
        self.assertEqual(
            BOOTSTRAP_RNG_NAMESPACE.encode("ascii"),
            b"upi-trm-exp1b-seed-bootstrap-v1",
        )
        self.assertEqual(len(BOOTSTRAP_RNG_NAMESPACE.encode("ascii")), 31)
        self.assertTrue(BOOTSTRAP_RNG_NAMESPACE.isascii())
        self.assertEqual(BOOTSTRAP_SEED, 3246702300714487323)
        self.assertEqual(BOOTSTRAP_SEED.to_bytes(8, "big").hex(), "2d0e9a8a6d87ce1b")

    def test_exact_hashed_preimage(self) -> None:
        """Pin the whole 45-byte preimage and its digest for (r=0, d=0).

        namespace(31) || 0x00(1) || seed(8) || replicate(4) || draw(1) = 45.
        A change to any field width, order, or separator moves this.
        """

        preimage = (
            b"upi-trm-exp1b-seed-bootstrap-v1"
            + b"\x00"
            + bytes.fromhex("2d0e9a8a6d87ce1b")
            + bytes.fromhex("00000000")
            + bytes.fromhex("00")
        )
        self.assertEqual(len(preimage), 45)
        self.assertEqual(
            preimage.hex(),
            "7570692d74726d2d65787031622d736565642d626f6f7473747261702d76"
            "31002d0e9a8a6d87ce1b0000000000",
        )
        self.assertEqual(
            hashlib.sha256(preimage).hexdigest(),
            "49d4e975e1b4ae1e0bb9570a9f41c76f769bb809bc0d5794c5f9c98d3f373df8",
        )
        self.assertEqual(hashlib.sha256(preimage).digest()[0], 73)
        self.assertEqual(hashlib.sha256(preimage).digest()[0] & 7, 1)
        self.assertEqual(resample_plan()[0][0], 1)

    def test_plan_matches_the_specified_digest_formula(self) -> None:
        """Recompute representative draws straight from the D2 formula."""

        namespace_bytes = BOOTSTRAP_RNG_NAMESPACE.encode("ascii")
        plan = resample_plan()
        for replicate, draw in (
            (0, 0),
            (0, 7),
            (1, 0),
            (1, 3),
            (4999, 4),
            (9999, 0),
            (9999, 7),
        ):
            with self.subTest(replicate=replicate, draw=draw):
                digest = hashlib.sha256(
                    namespace_bytes
                    + b"\x00"
                    + BOOTSTRAP_SEED.to_bytes(8, "big")
                    + replicate.to_bytes(4, "big")
                    + draw.to_bytes(1, "big")
                ).digest()
                self.assertEqual(plan[replicate][draw], digest[0] & 7)

    def test_golden_representative_draws(self) -> None:
        """Hard-coded indices under the approved namespace.

        Independent of the formula recomputation above: if the construction
        changes at all, these move even when the recomputation still agrees
        with itself.
        """

        plan = resample_plan()
        for (replicate, draw), expected in {
            (0, 0): 1,
            (0, 7): 0,
            (1, 0): 2,
            (1, 3): 6,
            (4999, 4): 3,
            (9999, 0): 7,
            (9999, 7): 5,
        }.items():
            with self.subTest(replicate=replicate, draw=draw):
                self.assertEqual(plan[replicate][draw], expected)

    def test_golden_replicate_rows_and_plan_digest(self) -> None:
        plan = resample_plan()
        self.assertEqual(plan[0], (1, 5, 7, 2, 1, 0, 1, 0))
        self.assertEqual(plan[1], (2, 0, 0, 6, 4, 5, 3, 1))
        self.assertEqual(plan[9999], (7, 0, 5, 4, 7, 6, 3, 5))
        interval = paired_seed_percentile_interval(seed_gaps=[1.0] * 8)
        self.assertEqual(
            interval.resample_plan_sha256,
            "ec582aaf77ca3e17d740a3eecb63e40d943ddacf1cb8a26a4706ea5e35f1c9ce",
        )

    def test_golden_interval_under_the_approved_namespace(self) -> None:
        """A fixed gap vector pins both endpoints and the replicate digest."""

        gaps = [-5.0, 3.0, 8.0, 0.0, -1.0, 2.0, 4.0, -3.0]
        result = paired_seed_percentile_interval(seed_gaps=gaps)
        self.assertEqual(result.rng_namespace, BOOTSTRAP_RNG_NAMESPACE)
        self.assertEqual(result.observed_mean, 1.0)
        self.assertEqual(result.interval_lower, -1.625)
        self.assertEqual(result.interval_upper, 3.75)
        self.assertEqual(
            result.replicate_vector_sha256,
            "9c905102f236899242cec159fdc12d7d78c82c5e35fe589b77af4c9219036720",
        )

    def test_namespace_and_seed_change_the_plan(self) -> None:
        base = resample_plan()
        self.assertEqual(base, resample_plan(namespace=BOOTSTRAP_RNG_NAMESPACE))
        self.assertNotEqual(base, resample_plan(namespace=_OTHER_NAMESPACE))
        self.assertNotEqual(base, resample_plan(namespace=_NAMESPACE + "x"))
        self.assertNotEqual(base, resample_plan(seed=1))

    def test_public_interval_exposes_no_namespace_or_seed_override(self) -> None:
        """Review finding 5: a registered result cannot carry a non-D2 plan."""

        import inspect

        signature = inspect.signature(paired_seed_percentile_interval)
        self.assertEqual(list(signature.parameters), ["seed_gaps"])
        for override in ({"namespace": _OTHER_NAMESPACE}, {"seed": 1}):
            with self.subTest(override=override):
                with self.assertRaises(TypeError):
                    paired_seed_percentile_interval(
                        seed_gaps=[1.0] * 8, **override
                    )
        self.assertEqual(
            paired_seed_percentile_interval(seed_gaps=[1.0] * 8).rng_namespace,
            BOOTSTRAP_RNG_NAMESPACE,
        )
        self.assertEqual(resample_plan(), resample_plan(namespace=_NAMESPACE))

    def test_malformed_namespace_is_rejected(self) -> None:
        for bad in (None, 0, b"", "", "né", 3.5, True):
            with self.subTest(bad=bad):
                with self.assertRaises(Exp1bBootstrapError):
                    resample_plan(namespace=bad)  # type: ignore[arg-type]

    def test_replicate_and_unit_counts_are_frozen(self) -> None:
        for replicates in (1, 999, 10001):
            with self.subTest(replicates=replicates):
                with self.assertRaisesRegex(Exp1bBootstrapError, "exactly 10000"):
                    resample_plan(namespace=_NAMESPACE, replicates=replicates)
        with self.assertRaisesRegex(Exp1bBootstrapError, "exactly 8"):
            resample_plan(namespace=_NAMESPACE, units=7)

    def test_index_mapping_is_uniform_over_eight_seeds(self) -> None:
        """256 is divisible by 8, so digest[0] & 7 is exactly uniform."""

        plan = resample_plan(namespace=_NAMESPACE)
        counts = [0] * 8
        for row in plan:
            for index in row:
                counts[index] += 1
        self.assertEqual(sum(counts), REPLICATES * UNITS)
        # 80,000 draws over 8 buckets: a few percent of 10,000 expected each.
        for index, count in enumerate(counts):
            with self.subTest(index=index):
                self.assertLess(abs(count - 10000), 500)

    def test_type7_quantile_matches_hand_computation(self) -> None:
        ordered = [float(value) for value in range(10)]
        # h = (10-1)*0.5 = 4.5 -> midway between 4 and 5.
        self.assertEqual(type7_quantile(ordered, 0.5), 4.5)
        self.assertEqual(type7_quantile(ordered, 0.0), 0.0)
        self.assertEqual(type7_quantile(ordered, 1.0), 9.0)
        # h = 9*0.025 = 0.225 -> 0 + 0.225*(1-0)
        self.assertAlmostEqual(type7_quantile(ordered, 0.025), 0.225, places=12)

    def test_interval_reads_the_frozen_order_statistics(self) -> None:
        """At B=10,000, p=0.025/0.975 read indices 249/250 and 9749/9750."""

        ordered = [float(value) for value in range(REPLICATES)]
        lower = type7_quantile(ordered, 0.025)
        upper = type7_quantile(ordered, 0.975)
        self.assertAlmostEqual(lower, 249.0 + 0.975 * 1.0, places=9)
        self.assertAlmostEqual(upper, 9749.0 + 0.025 * 1.0, places=9)

    def test_interval_over_eight_gaps(self) -> None:
        gaps = [-5.0, 3.0, 8.0, 0.0, -1.0, 2.0, 4.0, -3.0]
        result = paired_seed_percentile_interval(seed_gaps=gaps)
        self.assertEqual(result.replicates, REPLICATES)
        self.assertEqual(len(result.replicate_means), REPLICATES)
        self.assertEqual(result.observed_mean, sum(gaps) / 8)
        self.assertLessEqual(result.interval_lower, result.observed_mean)
        self.assertGreaterEqual(result.interval_upper, result.observed_mean)
        self.assertEqual(
            result.replicate_vector_sha256, replicate_digest(result.replicate_means)
        )
        again = paired_seed_percentile_interval(seed_gaps=gaps)
        self.assertEqual(result.interval_lower, again.interval_lower)
        self.assertEqual(result.interval_upper, again.interval_upper)
        self.assertEqual(
            result.replicate_vector_sha256, again.replicate_vector_sha256
        )

    def test_constant_gaps_give_a_degenerate_interval(self) -> None:
        result = paired_seed_percentile_interval(seed_gaps=[2.0] * 8)
        self.assertEqual(result.interval_lower, 2.0)
        self.assertEqual(result.interval_upper, 2.0)
        self.assertEqual(result.observed_mean, 2.0)

    def test_gap_order_matters(self) -> None:
        gaps = [-5.0, 3.0, 8.0, 0.0, -1.0, 2.0, 4.0, -3.0]
        forward = paired_seed_percentile_interval(seed_gaps=gaps)
        reversed_gaps = paired_seed_percentile_interval(
            seed_gaps=list(reversed(gaps))
        )
        self.assertEqual(forward.observed_mean, reversed_gaps.observed_mean)
        self.assertNotEqual(
            forward.replicate_vector_sha256, reversed_gaps.replicate_vector_sha256
        )

    def test_wrong_unit_count_is_refused(self) -> None:
        for count in (0, 7, 9):
            with self.subTest(count=count):
                with self.assertRaisesRegex(Exp1bBootstrapError, "exactly 8"):
                    paired_seed_percentile_interval(seed_gaps=[1.0] * count)

    def test_nonfinite_gap_is_refused(self) -> None:
        for bad in (float("nan"), float("inf"), True, None, "0.0"):
            with self.subTest(bad=bad):
                with self.assertRaises(Exp1bBootstrapError):
                    paired_seed_percentile_interval(
                        seed_gaps=[bad, *([1.0] * 7)]  # type: ignore[list-item]
                    )

    def test_no_dependency_on_the_v2_statistics_module(self) -> None:
        source = (
            _ROOT / "scripts" / "policy_improvement_exp1b_bootstrap.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("scripts.policy_improvement_statistics", imported)
        for forbidden in (
            "paired_seed_cluster_puzzle_bootstrap",
            "paired_seed_permutation_test",
            "3472274560",
        ):
            with self.subTest(forbidden=forbidden):
                # The docstring names them to explain the exclusion; no code
                # line may reference them.
                code_lines = [
                    line
                    for line in source.splitlines()
                    if forbidden in line and not line.lstrip().startswith(("#", "*"))
                ]
                self.assertEqual(
                    [line for line in code_lines if "``" not in line], []
                )




class Exp1bAuthenticatedRouteTest(unittest.TestCase):
    """Findings 2 and 4: trust anchors are computed, never accepted."""

    def test_production_factory_opens_and_serves_the_octet(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            self.assertEqual(route.access_state, "sealed_octet_complete")
            _serve_all(route, fixture)
            octet = route.sealed_octet()
            self.assertIsInstance(octet, Exp1bSealedOctet)
            self.assertEqual(len(octet.payloads), 8)
            self.assertEqual(octet.exp1b_protocol_sha256, fixture.protocol_sha)
            self.assertEqual(
                octet.runtime_attestation.runtime_authorization_sha256,
                _THEORY_AUTHORIZATION,
            )
            # Reloaded from disk, so the in-memory summary is deliberately
            # absent; the persisted result values are what aggregation consumes.
            for payload in octet.payloads:
                self.assertIsNone(payload.summary)
                self.assertEqual(payload.result_values().state_count, 128)
            self.assertEqual(len(octet.authenticated_checkpoints), 8)
            self.assertEqual(
                octet.effective_config_sha256, fixture.effective_config_sha256
            )

    def test_direct_route_construction_is_refused(self) -> None:
        """The constructor whose anchors were all caller values is closed."""

        with self.assertRaisesRegex(Exp1bBridgeError, "direct construction is refused"):
            Exp1bBridgeRoute(
                provenance={},
                schedule=None,  # type: ignore[arg-type]
                census=None,  # type: ignore[arg-type]
                population_registry={},
                generation=None,  # type: ignore[arg-type]
                checkpoints=(),
                attestation=_theory_attestation(),
                evaluator_attestation={},
                exp1b_protocol_sha256=_digest("p"),
                exp1b_registry_sha256=_digest("r"),
                exp1b_amendment_sha256=_digest("a"),
            )

    def test_forged_document_digests_are_refused(self) -> None:
        """Review reproducer: self-asserted digests no longer open the route."""

        for field in (
            "exp1b_protocol_sha256",
            "exp1b_registry_sha256",
            "exp1b_amendment_sha256",
        ):
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                fixture = _RouteFixture(Path(tmp), **{field: _digest("forged")})
                with self.assertRaisesRegex(
                    Exp1bBridgeError, "does not bind the authenticated"
                ):
                    fixture.open()

    def test_a_forged_producer_identity_is_refused(self) -> None:
        """Review reproducer: well-formed but unadmitted producer digests."""

        for field in (
            "source_git_commit",
            "runtime_sha256",
            "launcher_sha256",
            "runtime_authorization_sha256",
        ):
            forged = "f" * (40 if field == "source_git_commit" else 64)
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                fixture = _RouteFixture(Path(tmp))
                producer = dict(fixture.provenance["producer_attestation"])
                producer[field] = forged
                fixture.rewrite_provenance(producer_attestation=producer)
                with self.assertRaisesRegex(
                    Exp1bBridgeError, "did not authorize for the producer role"
                ):
                    fixture.open()

    def test_the_two_roles_are_authenticated_separately(self) -> None:
        """Finding 1: Stage A evidence must open under the distinct Stage B runtime.

        The previous route compared the provenance's producer identity against
        the *live evaluator* attestation. Since the admission requires the two
        runtime digests to differ, that comparison could never succeed and Stage
        B could not consume any real Stage A generation. This is the end-to-end
        reproducer: finalize under the full role, open under the theory role.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            producer = fixture.provenance["producer_attestation"]
            self.assertEqual(producer["role"], FULL_ROLE)
            self.assertEqual(producer["runtime_sha256"], _RUNTIME)
            self.assertNotEqual(_RUNTIME, _THEORY_RUNTIME)

            route = fixture.open(attestation=_theory_attestation())
            self.assertEqual(route.access_state, "sealed_octet_complete")
            _serve_all(route, fixture)
            octet = route.sealed_octet()
            self.assertEqual(octet.producer_attestation["role"], FULL_ROLE)
            self.assertEqual(octet.producer_attestation["runtime_sha256"], _RUNTIME)
            self.assertEqual(
                octet.evaluator_attestation["role"], THEORY_BRIDGE_ROLE
            )
            self.assertEqual(
                octet.evaluator_attestation["runtime_sha256"], _THEORY_RUNTIME
            )
            result = aggregate_exp1b_result(octet)
            self.assertEqual(result.producer_attestation["runtime_sha256"], _RUNTIME)
            self.assertEqual(
                result.evaluator_attestation["runtime_sha256"], _THEORY_RUNTIME
            )

    def test_opening_under_the_producer_identity_is_refused(self) -> None:
        """The evaluator PAR is the only thing allowed to serve the route."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            with self.assertRaisesRegex(
                Exp1bBridgeError, "authorized for the theory-bridge role"
            ):
                fixture.open(attestation=_attestation())

    def test_forged_census_digests_are_refused(self) -> None:
        for field in ("ordered_population_sha256", "population_binding_sha256"):
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                fixture = _RouteFixture(Path(tmp), **{field: _digest("forged")})
                with self.assertRaisesRegex(
                    Exp1bBridgeError, "different validation_bridge census"
                ):
                    fixture.open()

    def test_census_identity_is_recomputed_not_declared(self) -> None:
        """Review reproducer: swapping a member no longer passes."""

        population = _load(_V2 / "populations.json")["populations"]["validation_bridge"]
        census = census_from_population(population, state_id_for=_state_id_for)
        swapped = (
            PopulationMember(
                state_id="tampered",
                record_index=9999,
                dataset_record_sha256=_digest("tampered"),
            ),
            *census.members[1:],
        )
        with self.assertRaisesRegex(Exp1bBridgeError, "ordering digest"):
            Exp1bCensus(
                population_id="validation_bridge",
                split="validation",
                members=swapped,
                ordered_record_sha256=census.ordered_record_sha256,
                binding_sha256=census.binding_sha256,
                member_ordering_sha256=census.member_ordering_sha256,
            )

    def test_stale_parent_digest_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            stale = dict(fixture.protocol["parent"])
            stale["registry_sha256"] = _digest("stale")
            fixture.protocol["parent"] = stale
            fixture.protocol_path.write_bytes(
                canonical_json_bytes(fixture.protocol) + b"\n"
            )
            with self.assertRaises(Exp1bBridgeError):
                fixture.open()

    def test_stable_file_digest_rejects_relative_and_missing_paths(self) -> None:
        with self.assertRaisesRegex(Exp1bBridgeError, "absolute"):
            stable_file_digest("configs/policy_improvement_exp1b/protocol.json", label="x")
        with self.assertRaisesRegex(Exp1bBridgeError, "unavailable"):
            stable_file_digest("/nonexistent/exp1b/protocol.json", label="x")
        digest = stable_file_digest(_EXP1B / "protocol.json", label="protocol")
        self.assertEqual(
            digest.sha256, hashlib.sha256((_EXP1B / "protocol.json").read_bytes()).hexdigest()
        )

    def test_route_refuses_validation_select_and_test(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            for population in ("validation_select", "confirmatory_test", "test"):
                with self.subTest(population=population):
                    with self.assertRaisesRegex(Exp1bBridgeError, "never opens"):
                        route.serve(
                            seed_position=0,
                            seed=fixture.sealed[0]["seed"],
                            run_id=fixture.sealed[0]["run_id"],
                            checkpoint_sha256=fixture.sealed[0]["checkpoint_sha256"],
                            evaluation_population=population,
                            **_evaluator(0),
                        )


class Exp1bDurableScheduleTest(unittest.TestCase):
    """Finding 3: durable, transactional, explicit pre-emission crash policy."""

    def _fixture(self, tmp: str) -> _RouteFixture:
        return _RouteFixture(Path(tmp))

    def test_pre_emission_failure_does_not_consume_the_request(self) -> None:
        """Review reproducer: an evaluator error left the seed unusable."""

        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            route = fixture.open()
            first = fixture.sealed[0]

            def exploding(state_id: str, depth: int) -> float:
                raise RuntimeError("synthetic evaluator failure")

            evaluator = _evaluator(0)
            evaluator["endpoint_values"] = exploding
            with self.assertRaises(RuntimeError):
                route.serve(
                    seed_position=0,
                    seed=first["seed"],
                    run_id=first["run_id"],
                    checkpoint_sha256=first["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **evaluator,
                )
            # The claim was released; position 0 is still the next expected one.
            self.assertEqual(route.access_state, "sealed_octet_complete")
            payload = route.serve(
                seed_position=0,
                seed=first["seed"],
                run_id=first["run_id"],
                checkpoint_sha256=first["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            self.assertEqual(payload.seed_position, 0)

    def test_state_survives_a_process_boundary(self) -> None:
        """Eight launcher processes share one durable schedule."""

        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            first = fixture.sealed[0]
            route = fixture.open()
            route.serve(
                seed_position=0,
                seed=first["seed"],
                run_id=first["run_id"],
                checkpoint_sha256=first["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            # A fresh route object stands in for a fresh launcher process.
            reopened = fixture.open()
            self.assertEqual(reopened.access_state, "bridge_open")
            with self.assertRaisesRegex(Exp1bSchemaError, "post-payload retry"):
                reopened.serve(
                    seed_position=0,
                    seed=first["seed"],
                    run_id=first["run_id"],
                    checkpoint_sha256=first["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **_evaluator(0),
                )

    def test_attempt_counts_are_recorded_and_published(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            route = fixture.open()
            first = fixture.sealed[0]
            evaluator = _evaluator(0)
            evaluator["endpoint_values"] = lambda s, d: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
            with self.assertRaises(RuntimeError):
                route.serve(
                    seed_position=0,
                    seed=first["seed"],
                    run_id=first["run_id"],
                    checkpoint_sha256=first["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **evaluator,
                )
            _serve_all(route, fixture)
            octet = route.sealed_octet()
            # The retried seed shows two attempts; the rest show one.
            self.assertEqual(octet.attempt_counts[0], 2)
            self.assertEqual(octet.attempt_counts[1:], (1,) * 7)

    def test_schedule_state_is_bound_to_its_provenance(self) -> None:
        """A state file cannot be reused under a different provenance."""

        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            fixture.open()
            generation = fixture.generation()
            producer = {
                **fixture.provenance["producer_attestation"],
                "source_git_commit": "e" * 40,
            }
            with self.assertRaises(Exp1bScheduleStateError):
                Exp1bRequestSchedule(
                    {**fixture.provenance, "producer_attestation": producer},
                    generation=generation,
                )

    def test_schedule_state_is_bound_to_its_generation(self) -> None:
        """The one derived state path is not reusable across generations."""

        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            fixture.open()
            other_root = Path(tmp) / "other-evidence"
            other_generation_directory = other_root / fixture.generation_id
            (other_generation_directory / "payloads").mkdir(parents=True)
            (other_generation_directory / "checkpoints").mkdir(parents=True)
            other = open_evidence_generation(
                evidence_root=other_root, generation_id=fixture.generation_id
            )
            self.assertNotEqual(
                other.generation_sha256, fixture.generation().generation_sha256
            )
            # Move the first generation's durable state under the second one.
            other.schedule_path.write_bytes(
                fixture.schedule_state_path.read_bytes()
            )
            with self.assertRaisesRegex(Exp1bScheduleStateError, "generation"):
                Exp1bRequestSchedule(fixture.provenance, generation=other)

    def test_corrupt_schedule_state_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            fixture.open()
            fixture.schedule_state_path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(Exp1bScheduleStateError, "not JSON"):
                fixture.open()

    def test_a_served_slot_without_its_payload_is_refused(self) -> None:
        """A served marker with no durable payload behind it is not evidence."""

        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            fixture.open()
            document = json.loads(fixture.schedule_state_path.read_text())
            document["slots"][3]["state"] = "served"
            document["slots"][3]["payload_sha256"] = _digest("x")
            fixture.schedule_state_path.write_bytes(
                canonical_json_bytes(document) + b"\n"
            )
            with self.assertRaisesRegex(
                Exp1bScheduleStateError, "no persisted payload"
            ):
                fixture.open()

    def test_served_slots_must_be_a_prefix(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            route = fixture.open()
            for offset in (0, 1):
                item = fixture.sealed[offset]
                route.serve(
                    seed_position=offset,
                    seed=item["seed"],
                    run_id=item["run_id"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **_evaluator(offset),
                )
            # Slot 1 stays served; slot 0 loses both its marker and its
            # payload, so the served set is {1} and no longer a prefix.
            payload_path = fixture.generation().payload_path(0)
            payload_path.rename(payload_path.with_name("seed-0.json.orphan"))
            document = json.loads(fixture.schedule_state_path.read_text())
            document["slots"][0]["state"] = "pending"
            document["slots"][0]["payload_sha256"] = None
            fixture.schedule_state_path.write_bytes(
                canonical_json_bytes(document) + b"\n"
            )
            with self.assertRaisesRegex(Exp1bScheduleStateError, "prefix"):
                fixture.open()


class Exp1bAggregationBarrierTest(unittest.TestCase):
    """Finding 2: only a completed authenticated route can produce a result."""

    def _octet(self, tmp: str, **kwargs: Any) -> Exp1bSealedOctet:
        fixture = _RouteFixture(Path(tmp))
        route = fixture.open()
        _serve_all(route, fixture, **kwargs)
        return route.sealed_octet()

    def test_loose_payload_objects_are_refused(self) -> None:
        """Review reproducer: eight SimpleNamespace envelopes were accepted."""

        forged = [
            SimpleNamespace(
                seed_position=offset,
                seed=seed,
                run_id=f"forged-{offset}",
                checkpoint_sha256=f"not-a-digest-{offset}",
                census_ordering_sha256="ignored",
                summary=None,
            )
            for offset, seed in enumerate(REGISTERED_SEEDS)
        ]
        with self.assertRaisesRegex(Exp1bAggregateError, "only a sealed octet"):
            aggregate_exp1b_result(forged)
        for wrong in ([], None, "octet", 8, {}):
            with self.subTest(wrong=type(wrong).__name__):
                with self.assertRaises(Exp1bAggregateError):
                    aggregate_exp1b_result(wrong)

    def test_aggregate_of_a_real_octet_carries_full_provenance(self) -> None:
        with TemporaryDirectory() as tmp:
            octet = self._octet(tmp)
            result = aggregate_exp1b_result(octet)
            self.assertEqual(result.schema_name, RESULT_SCHEMA_NAME)
            self.assertEqual(result.provenance_sha256, octet.provenance_sha256)
            self.assertEqual(result.schedule_sha256, octet.schedule_sha256)
            self.assertEqual(result.census_ordering_sha256, octet.census_ordering_sha256)
            self.assertEqual(
                result.evaluator_attestation["runtime_authorization_sha256"],
                _THEORY_AUTHORIZATION,
            )
            self.assertEqual(result.producer_attestation["source_git_commit"], _COMMIT)
            document = result.as_document()
            for field in (
                "provenance_sha256",
                "schedule_sha256",
                "census_ordering_sha256",
                "producer_attestation",
                "evaluator_attestation",
                "attempt_counts",
            ):
                self.assertIn(field, document)

    def test_reused_snapshot_behind_different_envelopes_is_refused(self) -> None:
        """Review reproducer: one real seed's result presented eight times."""

        with TemporaryDirectory() as tmp:
            octet = self._octet(tmp)
            first = octet.payloads[0]
            cloned = tuple(
                replace(
                    payload,
                    checkpoint_sha256=first.checkpoint_sha256,
                    persisted=first.persisted,
                )
                for payload in octet.payloads
            )
            with self.assertRaisesRegex(
                Exp1bAggregateError, "registered checkpoint identity"
            ):
                aggregate_exp1b_result(_reissue(octet, payloads=cloned))

    def test_a_summary_from_another_snapshot_is_refused(self) -> None:
        """The in-memory path still cross-checks summary against identity."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            served = [
                route.serve(
                    seed_position=item["seed_position"],
                    seed=item["seed"],
                    run_id=item["run_id"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **_evaluator(offset),
                )
                for offset, item in enumerate(fixture.sealed)
            ]
            octet = route.sealed_octet()
            # Re-attach the live summaries, then point one at another snapshot.
            rebuilt = tuple(
                replace(reloaded, summary=live.summary)
                for reloaded, live in zip(octet.payloads, served)
            )
            tampered = (
                replace(rebuilt[0], summary=served[1].summary),
                *rebuilt[1:],
            )
            with self.assertRaisesRegex(
                Exp1bAggregateError, "differs from the snapshot"
            ):
                aggregate_exp1b_result(_reissue(octet, payloads=tampered))

    def test_one_state_population_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            octet = self._octet(tmp)
            member = PopulationMember(
                state_id=_member(0),
                record_index=0,
                dataset_record_sha256=_digest("only"),
            )
            observation = PairedEndpointObservation(
                state_id=member.state_id,
                record_index=member.record_index,
                dataset_record_sha256=member.dataset_record_sha256,
                snapshot_id=octet.payloads[0].checkpoint_sha256,
                deployed_depth_n=2,
                reference_depth_m=8,
                action_mask=(True, True),
                base_probabilities=(0.5, 0.5),
                endpoint_value_n=0.0,
                endpoint_value_m=1.0,
                action_values_n=(0.0, 0.0),
                action_values_m=(1.0, 1.0),
                gamma=0.99,
            )
            tiny = paired_population_summary(
                snapshot_id=octet.payloads[0].checkpoint_sha256,
                population_id="validation_bridge",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=(member,),
                state_diagnostics=[paired_state_diagnostic(observation)],
            )
            self.assertEqual(tiny.state_count, 1)
            # Review reproducer: replacing a persisted scalar no longer leaves
            # the payload digest unchanged, so the substitution is caught before
            # the state-count check is even reached.
            shrunk = (
                replace(
                    octet.payloads[0],
                    persisted=replace(octet.payloads[0].persisted, state_count=1),
                ),
                *octet.payloads[1:],
            )
            with self.assertRaisesRegex(Exp1bAggregateError, "digest differs"):
                aggregate_exp1b_result(_reissue(octet, payloads=shrunk))
            # And the population-size rule itself still bites on the in-memory
            # path, where no persisted digest exists to catch it first.
            with self.assertRaisesRegex(
                Exp1DiagnosticsError, "population size differs"
            ):
                paired_population_summary(
                    snapshot_id=octet.payloads[0].checkpoint_sha256,
                    population_id="validation_bridge",
                    deployed_depth_n=2,
                    reference_depth_m=8,
                    expected_population=self._census_members(),
                    state_diagnostics=[paired_state_diagnostic(observation)],
                )

    def _census_members(self) -> tuple[PopulationMember, ...]:
        population = _load(_V2 / "populations.json")["populations"][
            "validation_bridge"
        ]
        return census_from_population(
            population, state_id_for=_state_id_for
        ).members

    def test_payload_from_another_census_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            octet = self._octet(tmp)
            tampered = (
                replace(octet.payloads[0], census_ordering_sha256=_digest("other")),
                *octet.payloads[1:],
            )
            with self.assertRaisesRegex(
                Exp1bAggregateError, "different census ordering"
            ):
                aggregate_exp1b_result(_reissue(octet, payloads=tampered))

    def test_partial_or_reordered_octet_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            octet = self._octet(tmp)
            with self.assertRaisesRegex(Exp1bAggregateError, "exactly 8"):
                aggregate_exp1b_result(_reissue(octet, payloads=octet.payloads[:7]))
            with self.assertRaises(Exp1bAggregateError):
                aggregate_exp1b_result(
                    replace(octet, payloads=tuple(reversed(octet.payloads)))
                )

    def test_interval_is_mandatory_and_carries_the_full_vector(self) -> None:
        """Findings 5 and 6, at the result boundary."""

        with TemporaryDirectory() as tmp:
            result = aggregate_exp1b_result(self._octet(tmp))
            interval = result.interval
            self.assertEqual(interval["status"], "available")
            self.assertEqual(interval["rng_namespace"], BOOTSTRAP_RNG_NAMESPACE)
            self.assertEqual(interval["bootstrap_seed"], BOOTSTRAP_SEED)
            self.assertEqual(interval["replicates"], REPLICATES)
            self.assertEqual(interval["replicate_encoding"], "float_hex_v1")
            vector = interval["replicate_means_hex"]
            self.assertEqual(len(vector), REPLICATES)
            decoded = decode_replicate_vector(vector)
            self.assertEqual(replicate_digest(decoded), interval["replicate_vector_sha256"])
            # An auditor can recompute both endpoints from the persisted vector.
            ordered = sorted(decoded)
            self.assertEqual(type7_quantile(ordered, 0.025), interval["interval_lower"])
            self.assertEqual(type7_quantile(ordered, 0.975), interval["interval_upper"])
            # And there is no way to suppress it.
            import inspect

            self.assertEqual(
                list(inspect.signature(aggregate_exp1b_result).parameters),
                ["sealed_octet"],
            )


class Exp1bTrainingSessionTest(unittest.TestCase):
    """Finding 7: opaque registration, authenticated load, restore before trainer."""

    class _Base:
        checkpoint_sha256 = _digest("admitted-base-checkpoint")
        model_state_sha256 = _digest("admitted-base-model")
        initialization_kind = "train_only_pretrained_base_policy"

    def setUp(self) -> None:
        self.protocol = _load(_EXP1B / "protocol.json")
        self.registry = _load(_EXP1B / "registry.json")
        self.training = self.protocol["training_population"]
        self.project_root = _ROOT
        self._registrations: dict[int, Exp1bTrainingRegistration] = {}

    def _registration(self, seed: int | None = None) -> Exp1bTrainingRegistration:
        # Cached: the session binds a guard to its registration by identity, so
        # helpers that mint a second one would not compose.
        key = seed if seed is not None else REGISTERED_SEEDS[0]
        if key not in self._registrations:
            self._registrations[key] = register_exp1b_training(
                protocol=self.protocol,
                registry=self.registry,
                project_root=self.project_root,
                seed=key,
            )
        return self._registrations[key]

    def _loaded(self, root: Any, split: str, requested: int, **overrides: Any) -> Any:
        values = {
            "dataset": {"split": split, "count": requested},
            "dataset_root": root,
            "split": split,
            "count": requested,
            "ordered_record_sha256": self.training["ordered_record_sha256"],
            "dataset_manifest_sha256": self.training["dataset_manifest_sha256"],
            "split_manifest_sha256": self.training["split_manifest_sha256"],
        }
        values.update(overrides)
        return LoadedTrainSplit(**values)

    def _guard(
        self,
        registration: Exp1bTrainingRegistration | None = None,
        **overrides: Any,
    ) -> TrainOnlyDatasetGuard:
        target = registration if registration is not None else self._registration()
        return TrainOnlyDatasetGuard(
            registration=target,
            loader=lambda root, split, count: self._loaded(
                root, split, count, **overrides
            ),
        )

    def _harness(self, *, restore_ok: bool = True) -> dict[str, Any]:
        log: list[str] = []
        seeds: list[int] = []
        state = {"digest": _digest("freshly-constructed-untrained")}

        class _Model:
            pass

        model = _Model()

        def model_factory(dataset: Any) -> Any:
            log.append(f"model_factory:{dataset['count']}")
            return model

        def restore_base_policy(target: Any, base: Any) -> str:
            log.append("restore")
            if restore_ok:
                state["digest"] = base.model_state_sha256
            return state["digest"]

        def trainer_factory(target: Any, dataset: Any, config_sha256: str) -> Any:
            log.append(f"build_trainer:{state['digest']}:{config_sha256}")
            return {"snapshot_digest": state["digest"]}

        def apply_seed(seed: int) -> None:
            log.append(f"seed:{seed}")
            seeds.append(seed)

        return {
            "log": log,
            "seeds": seeds,
            "apply_seed": apply_seed,
            "model_factory": model_factory,
            "restore_base_policy": restore_base_policy,
            "trainer_factory": trainer_factory,
            "model_state_digest": lambda target: state["digest"],
        }

    def _build(self, **overrides: Any) -> Any:
        harness = overrides.pop("harness", None) or self._harness()
        registration = overrides.pop("registration", None) or self._registration()
        guard = overrides.pop("dataset_guard", None)
        if guard is None:
            guard = self._guard(registration)
        arguments = {
            "registration": registration,
            "dataset_guard": guard,
            "base_policy": self._Base(),
            "apply_seed": harness["apply_seed"],
            "model_factory": harness["model_factory"],
            "restore_base_policy": harness["restore_base_policy"],
            "trainer_factory": harness["trainer_factory"],
            "model_state_digest": harness["model_state_digest"],
        }
        arguments.update(overrides)
        return harness, guard, build_exp1b_training_session(**arguments)

    def test_registered_row_load_restore_then_trainer(self) -> None:
        registration = self._registration()
        harness, guard, session = self._build(registration=registration)
        self.assertEqual(
            session.call_order,
            (
                "seed_applied",
                "train_split_loaded",
                "model_constructed",
                "base_policy_restored",
                "trainer_built",
            ),
        )
        self.assertEqual(
            harness["log"],
            [
                f"seed:{REGISTERED_SEEDS[0]}",
                f"model_factory:{TRAINING_RECORD_COUNT}",
                "restore",
                (
                    f"build_trainer:{self._Base.model_state_sha256}"
                    f":{registration.effective_config_sha256}"
                ),
            ],
        )
        self.assertEqual(
            guard.resolved, [(registration.dataset_root, "train", 1024)]
        )
        self.assertEqual(session.train_record_count, TRAINING_RECORD_COUNT)
        self.assertEqual(session.seed, REGISTERED_SEEDS[0])
        self.assertEqual(
            session.effective_config_sha256,
            self.protocol["effective_config"]["effective_config_sha256"],
        )
        self.assertEqual(
            session.train_ordered_record_sha256,
            self.training["ordered_record_sha256"],
        )
        self.assertEqual(
            session.trainer["snapshot_digest"], self._Base.model_state_sha256
        )

    def test_the_session_consumes_only_an_opaque_registration(self) -> None:
        """Review reproducer: a hand-assembled row can no longer build a session."""

        registry = _load(_EXP1B / "registry.json")
        row = dict(registry["rows"][0])
        harness = self._harness()
        with self.assertRaisesRegex(Exp1bSessionError, "minted training registration"):
            build_exp1b_training_session(
                registration=row,  # type: ignore[arg-type]
                dataset_guard=self._guard(),
                base_policy=self._Base(),
                apply_seed=harness["apply_seed"],
                model_factory=harness["model_factory"],
                restore_base_policy=harness["restore_base_policy"],
                trainer_factory=harness["trainer_factory"],
                model_state_digest=harness["model_state_digest"],
            )
        # And a directly constructed registration carries no issuance token.
        with self.assertRaisesRegex(Exp1bSessionError, "must be minted"):
            Exp1bTrainingRegistration(
                _issued_by=object(),
                _row=row,
                project_root=self.project_root,
                dataset_root=self.project_root,
                protocol_sha256=_digest("p"),
                registry_sha256=_digest("r"),
                effective_config_sha256=_digest("c"),
                dataset_name="policy-improvement-hard-4x4-v1",
                training_population_id="train_full",
                ordered_record_sha256=_digest("o"),
                dataset_manifest_sha256=_digest("m"),
                split_manifest_sha256=_digest("s"),
                train_record_count=TRAINING_RECORD_COUNT,
            )

    def test_unregistered_run_is_refused(self) -> None:
        """Review reproducer: run_id='unregistered-run', seed=999 was accepted."""

        with self.assertRaisesRegex(Exp1bSessionError, "not exactly one registered"):
            register_exp1b_training(
                protocol=self.protocol,
                registry=self.registry,
                project_root=self.project_root,
                seed=999,
            )

    def test_registration_refuses_a_registry_from_another_protocol(self) -> None:
        stale = {**self.registry, "protocol_sha256": _digest("stale")}
        with self.assertRaises(Exp1bSchemaError):
            register_exp1b_training(
                protocol=self.protocol,
                registry=stale,
                project_root=self.project_root,
                seed=REGISTERED_SEEDS[0],
            )

    def test_guard_rejects_a_wrong_record_count(self) -> None:
        """Review reproducer: load('/registered', 'train', 1) was accepted."""

        registration = self._registration()
        guard = self._guard(registration)
        for count in (0, 1, 128, 1023, 1025):
            with self.subTest(count=count):
                with self.assertRaisesRegex(Exp1bSessionError, "exactly 1024"):
                    guard.load(registration.dataset_root, "train", count)
        self.assertEqual(guard.resolved, [])
        self.assertEqual(len(guard.refused), 5)

    def test_guard_requires_authenticated_loader_metadata(self) -> None:
        """Finding 7: 1,024 records from an unregistered split is still wrong."""

        registration = self._registration()
        cases = {
            "ordered_record_sha256": "ordered_record_sha256",
            "dataset_manifest_sha256": "dataset_manifest_sha256",
            "split_manifest_sha256": "split_manifest_sha256",
        }
        for field, message in cases.items():
            with self.subTest(field=field):
                guard = self._guard(registration, **{field: _digest("other")})
                with self.assertRaisesRegex(Exp1bSessionError, message):
                    guard.load(registration.dataset_root, "train", 1024)
                self.assertEqual(guard.resolved, [])
                self.assertEqual(len(guard.refused), 1)

    def test_guard_refuses_a_loader_that_returns_no_metadata(self) -> None:
        registration = self._registration()
        guard = TrainOnlyDatasetGuard(
            registration=registration,
            loader=lambda root, split, count: {"split": split, "count": count},
        )
        with self.assertRaisesRegex(Exp1bSessionError, "authenticated split metadata"):
            guard.load(registration.dataset_root, "train", 1024)

    def test_guard_refuses_a_metadata_count_that_disagrees(self) -> None:
        registration = self._registration()
        guard = self._guard(registration, count=1023)
        with self.assertRaisesRegex(Exp1bSessionError, "wrong record count"):
            guard.load(registration.dataset_root, "train", 1024)

    def test_metadata_is_authenticated_before_the_model_exists(self) -> None:
        registration = self._registration()
        harness = self._harness()
        guard = self._guard(registration, ordered_record_sha256=_digest("other"))
        with self.assertRaises(Exp1bSessionError):
            build_exp1b_training_session(
                registration=registration,
                dataset_guard=guard,
                base_policy=self._Base(),
                apply_seed=harness["apply_seed"],
                model_factory=harness["model_factory"],
                restore_base_policy=harness["restore_base_policy"],
                trainer_factory=harness["trainer_factory"],
                model_state_digest=harness["model_state_digest"],
            )
        # The seed is applied first by design; nothing after the load ran.
        self.assertEqual(harness["log"], [f"seed:{registration.seed}"])
        self.assertEqual(harness["seeds"], [registration.seed])

    def test_session_performs_the_load_itself(self) -> None:
        """The builder loads; it does not merely check that someone else did."""

        registration = self._registration()
        guard = self._guard(registration)
        self.assertEqual(guard.resolved, [])
        self._build(registration=registration, dataset_guard=guard)
        self.assertEqual(guard.resolved_splits, ("train",))
        self.assertFalse(guard.touched_evaluation_data)

    def test_a_pre_used_guard_is_refused(self) -> None:
        registration = self._registration()
        harness = self._harness()
        guard = self._guard(registration)
        guard.load(registration.dataset_root, "train", 1024)
        with self.assertRaisesRegex(Exp1bSessionError, "fresh loader guard"):
            build_exp1b_training_session(
                registration=registration,
                dataset_guard=guard,
                base_policy=self._Base(),
                apply_seed=harness["apply_seed"],
                model_factory=harness["model_factory"],
                restore_base_policy=harness["restore_base_policy"],
                trainer_factory=harness["trainer_factory"],
                model_state_digest=harness["model_state_digest"],
            )

    def test_a_guard_bound_to_another_registration_is_refused(self) -> None:
        harness = self._harness()
        first = self._registration(REGISTERED_SEEDS[0])
        second = self._registration(REGISTERED_SEEDS[1])
        with self.assertRaisesRegex(Exp1bSessionError, "different registration"):
            build_exp1b_training_session(
                registration=first,
                dataset_guard=self._guard(second),
                base_policy=self._Base(),
                apply_seed=harness["apply_seed"],
                model_factory=harness["model_factory"],
                restore_base_policy=harness["restore_base_policy"],
                trainer_factory=harness["trainer_factory"],
                model_state_digest=harness["model_state_digest"],
            )

    def test_missing_base_artifact_is_a_hard_error(self) -> None:
        harness = self._harness()
        with self.assertRaisesRegex(
            Exp1bSessionError, "no random-initialization fallback"
        ):
            build_exp1b_training_session(
                registration=self._registration(),
                dataset_guard=self._guard(),
                base_policy=None,
                apply_seed=harness["apply_seed"],
                model_factory=harness["model_factory"],
                restore_base_policy=harness["restore_base_policy"],
                trainer_factory=harness["trainer_factory"],
                model_state_digest=harness["model_state_digest"],
            )
        self.assertEqual(harness["log"], [])

    def test_a_restore_that_does_not_land_is_refused(self) -> None:
        harness = self._harness(restore_ok=False)
        with self.assertRaisesRegex(Exp1bSessionError, "differs from its registered"):
            build_exp1b_training_session(
                registration=self._registration(),
                dataset_guard=self._guard(),
                base_policy=self._Base(),
                apply_seed=harness["apply_seed"],
                model_factory=harness["model_factory"],
                restore_base_policy=harness["restore_base_policy"],
                trainer_factory=harness["trainer_factory"],
                model_state_digest=harness["model_state_digest"],
            )

    def test_train_validation_separation(self) -> None:
        registration = self._registration()
        guard = self._guard(registration)
        for root, split in (
            (registration.dataset_root, "validation"),
            (registration.dataset_root, "test"),
            (Path("/other/root"), "train"),
        ):
            with self.subTest(root=str(root), split=split):
                with self.assertRaisesRegex(Exp1bSessionError, "non-train path"):
                    guard.load(root, split, 1024)
        self.assertEqual(guard.resolved, [])
        self.assertEqual(len(guard.refused), 3)


class Exp1bStrictJsonTypeTest(unittest.TestCase):
    """Finding 9: bool and float lookalikes are refused in every document."""

    def test_protocol_integer_and_number_lookalikes(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        for field, value in (
            ("bellman_horizon_K", True),
            ("bellman_horizon_K", 1.0),
            ("terminal_environment_interactions", 10000.0),
            ("deployed_depth_n", 2.0),
            ("reference_depth_m", True),
            ("schema_version", 1.0),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_protocol({**protocol, field: value})

    def test_protocol_nested_lookalikes(self) -> None:
        protocol = _load(_EXP1B / "protocol.json")
        for path, value in (
            (("bridge_route", "request_count"), 8.0),
            (("bridge_route", "depths_per_request"), [2.0, 8.0]),
            (("bridge_route", "requires_sealed_octet"), 1),
            (("analysis", "bootstrap", "replicates"), 10000.0),
            (("analysis", "bootstrap", "units"), 8.0),
            (("analysis", "bootstrap", "seed"), float(BOOTSTRAP_SEED)),
            (("analysis", "interval", "lower_probability"), True),
            (("training_population", "count"), 1024.0),
            (("evaluation_population", "count"), 128.0),
            (("seeds", "registered"), [float(s) for s in REGISTERED_SEEDS]),
            (("parent", "protocol_schema_version"), 2.0),
        ):
            with self.subTest(path=path, value=value):
                broken = json.loads(json.dumps(protocol))
                target = broken
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_protocol(broken)

    def test_registry_lookalikes(self) -> None:
        registry = _load(_EXP1B / "registry.json")
        for field, value in (("row_count", 8.0), ("schema_version", True)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_registry({**registry, field: value})
        for field, value in (
            ("seed", float(REGISTERED_SEEDS[0])),
            ("seed_position", True),
            ("n", 2.0),
            ("K", 1.0),
            ("terminal_environment_interactions", 10000.0),
            ("paper_evidence_eligible", 1),
        ):
            with self.subTest(row_field=field, value=value):
                broken = json.loads(json.dumps(registry))
                broken["rows"][0][field] = value
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_registry(broken)

    def test_amendment_lookalikes(self) -> None:
        amendment = _load(_EXP1B / "amendments" / "reduced_study_exp1b.json")
        for field, value in (("schema_version", 1.0), ("schema_version", True)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(Exp1bSchemaError):
                    validate_exp1b_amendment({**amendment, field: value})

    def test_provenance_lookalikes(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            base = fixture.provenance
            for field, value in (
                ("schema_version", 1.0),
                ("schema_version", True),
                ("seed_ids", [float(s) for s in REGISTERED_SEEDS]),
            ):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(Exp1bSchemaError):
                        validate_substitute_provenance({**base, field: value})
            for field, value in (
                ("environment_interactions", 10000.0),
                ("checkpoint_size_bytes", 4096.0),
                ("sealed", 1),
                ("seed_position", True),
            ):
                with self.subTest(checkpoint_field=field, value=value):
                    broken = json.loads(json.dumps(base))
                    broken["sealed_checkpoints"][0][field] = value
                    with self.assertRaises(Exp1bSchemaError):
                        validate_substitute_provenance(broken)
            for field, value in (
                ("checkpoint_size_bytes", 8192.0),
                ("shared_across_seeds", 1),
                ("not_selected_by_validation_or_test", 1),
            ):
                with self.subTest(base_field=field, value=value):
                    broken = json.loads(json.dumps(base))
                    broken["base_policy_artifact"][field] = value
                    with self.assertRaises(Exp1bSchemaError):
                        validate_substitute_provenance(broken)


class Exp1bLauncherDispatchTest(unittest.TestCase):
    """Finding 1: both purposes reach an authenticated, launcher-owned route."""

    def setUp(self) -> None:
        from confirmatory_runtime_launcher import ConfirmatoryRuntimeError
        import phase4_runtime_launcher as launcher

        self.launcher = launcher
        self.error = ConfirmatoryRuntimeError
        self.digest = "0" * 64

    def test_caller_arguments_do_not_pass_through(self) -> None:
        """Review reproducer: the caller list was returned unchanged."""

        for purpose in (
            self.launcher.POLICY_IMPROVEMENT_EXP1B_TRAINING_PURPOSE,
            self.launcher.POLICY_IMPROVEMENT_EXP1B_BRIDGE_PURPOSE,
        ):
            with self.subTest(purpose=purpose):
                with self.assertRaises(self.error):
                    self.launcher._normalize_child_args(
                        purpose,
                        "/root",
                        [
                            "--caller-owned-option",
                            "value",
                            "--policy-improvement-exp1b-entrypoint",
                        ],
                        runtime_authorization_sha256=self.digest,
                        policy_protocol_v2=True,
                    )

    def test_training_dispatch_injects_marker_and_owned_paths(self) -> None:
        out = self.launcher._normalize_child_args(
            self.launcher.POLICY_IMPROVEMENT_EXP1B_TRAINING_PURPOSE,
            "/root",
            [
                "--base-policy-artifact", "/a/base.pt",
                "--base-policy-amendment", "/a/amend.json",
                "--execution-admission", "/owner/admission.json",
                "--evidence-root", "/ev",
                "--evidence-generation", "gen-0001",
                "--row-id", "exp1b-fixed-base-exact-persistent-seed2081976412",
                "--seed", "2081976412",
            ],
            runtime_authorization_sha256=self.digest,
            policy_protocol_v2=True,
        )
        self.assertEqual(out[0], "--policy-improvement-exp1b-entrypoint")
        self.assertIn("--policy-improvement-exp1b-training", out)
        self.assertIn("/root/configs/policy_improvement_exp1b/protocol.json", out)
        self.assertIn("/root/configs/policy_improvement_exp1b/registry.json", out)
        # The signed admission is an external owner artifact; the launcher owns
        # the option and refuses a path inside the checkout, but never invents
        # a location inside it. The dataset root is derived from the registered
        # training population rather than taken from the caller.
        self.assertEqual(out[out.index("--execution-admission") + 1],
                         "/owner/admission.json")
        self.assertNotIn("--dataset-root", out)

    def test_an_admission_inside_the_checkout_is_refused(self) -> None:
        """Review reproducer: a signed file in the tree fails source authorization."""

        with self.assertRaisesRegex(self.error, "outside the source checkout"):
            self.launcher._normalize_child_args(
                self.launcher.POLICY_IMPROVEMENT_EXP1B_TRAINING_PURPOSE,
                "/root",
                [
                    "--base-policy-artifact", "/a/base.pt",
                    "--base-policy-amendment", "/a/amend.json",
                    "--execution-admission",
                    "/root/configs/policy_improvement_exp1b/amendments/a.json",
                    "--evidence-root", "/ev",
                    "--evidence-generation", "gen-0001",
                    "--row-id", "r",
                    "--seed", "1",
                ],
                runtime_authorization_sha256=self.digest,
                policy_protocol_v2=True,
            )

    def test_the_exp1b_bridge_uses_the_producer_checkout(self) -> None:
        """Review reproducer: bridge child paths came from the evaluator root."""

        tree = ast.parse(
            (_ROOT / "phase4_runtime_launcher.py").read_text(encoding="utf-8")
        )
        selector = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "policy_project_root"
            and isinstance(node.value, ast.IfExp)
        )
        names = {
            element.id
            for element in ast.walk(selector.value.test)
            if isinstance(element, ast.Name)
        }
        self.assertIn("POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE", names)
        self.assertIn("POLICY_IMPROVEMENT_EXP1B_BRIDGE_PURPOSE", names)

    def test_bridge_dispatch_injects_marker_and_parent_paths(self) -> None:
        out = self.launcher._normalize_child_args(
            self.launcher.POLICY_IMPROVEMENT_EXP1B_BRIDGE_PURPOSE,
            "/root",
            [
                "--execution-admission", "/owner/admission.json",
                "--evidence-root", "/ev",
                "--evidence-generation", "gen-0001",
                "--seed-position", "0",
            ],
            runtime_authorization_sha256=self.digest,
            policy_protocol_v2=True,
        )
        self.assertEqual(out[0], "--policy-improvement-exp1b-bridge-entrypoint")
        self.assertIn("/root/configs/policy_improvement_v2/populations.json", out)
        self.assertIn(
            "/root/configs/policy_improvement_exp1b/amendments/"
            "reduced_study_exp1b.json",
            out,
        )
        # The provenance and the schedule state are derived from the evidence
        # generation, never supplied: there is one state artifact per
        # generation and no alternate path to it.
        self.assertNotIn("--substitute-provenance", out)
        self.assertNotIn("--schedule-state", out)

    def test_held_out_and_publication_options_are_refused(self) -> None:
        for bad in ("--test-open", "--validation-split", "--evaluation-split"):
            with self.subTest(option=bad):
                with self.assertRaisesRegex(self.error, "held-out"):
                    self.launcher._normalize_child_args(
                        self.launcher.POLICY_IMPROVEMENT_EXP1B_TRAINING_PURPOSE,
                        "/root",
                        [
                            bad, "x",
                            "--base-policy-artifact", "/a",
                            "--base-policy-amendment", "/b",
                            "--execution-admission", "/owner/admission.json",
                            "--evidence-root", "/e",
                            "--evidence-generation", "gen-0001",
                            "--row-id", "r",
                            "--seed", "1",
                        ],
                        runtime_authorization_sha256=self.digest,
                        policy_protocol_v2=True,
                    )

    def test_missing_authorization_digest_is_refused(self) -> None:
        with self.assertRaisesRegex(self.error, "authorization digest"):
            self.launcher._normalize_child_args(
                self.launcher.POLICY_IMPROVEMENT_EXP1B_TRAINING_PURPOSE,
                "/root",
                [
                    "--base-policy-artifact", "/a",
                    "--base-policy-amendment", "/b",
                    "--execution-admission", "/owner/admission.json",
                    "--evidence-root", "/e",
                    "--evidence-generation", "gen-0001",
                    "--row-id", "r",
                    "--seed", "1",
                ],
                runtime_authorization_sha256=None,
                policy_protocol_v2=True,
            )

    def test_exp1b_purposes_join_every_gating_purpose_set(self) -> None:
        """Both purposes must appear wherever their model purpose appears.

        The training purpose shadows ``policy-improvement-throughput`` and the
        bridge purpose shadows ``policy-improvement-theory-bridge``. A gating
        set that lists the model but omits the Experiment 1B purpose is exactly
        the hole the review found: authorization arguments present, never
        consumed.
        """

        tree = ast.parse(
            (_ROOT / "phase4_runtime_launcher.py").read_text(encoding="utf-8")
        )
        pairs = (
            (
                "POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE",
                "POLICY_IMPROVEMENT_EXP1B_TRAINING_PURPOSE",
            ),
            (
                "POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE",
                "POLICY_IMPROVEMENT_EXP1B_BRIDGE_PURPOSE",
            ),
        )
        # Scope to the gating logic. `_normalize_child_args` deliberately
        # routes the Experiment 1B purposes through their own branch rather
        # than the v2 full/theory one, so its dispatch sets are excluded.
        dispatch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_normalize_child_args"
        )
        dispatch_sets = {
            id(node) for node in ast.walk(dispatch) if isinstance(node, ast.Set)
        }
        sets = [
            {
                element.id
                for element in node.elts
                if isinstance(element, ast.Name)
            }
            for node in ast.walk(tree)
            if isinstance(node, ast.Set) and id(node) not in dispatch_sets
        ]
        for model, shadow in pairs:
            gating = [names for names in sets if model in names]
            self.assertTrue(gating, f"no gating set mentions {model}")
            for names in gating:
                with self.subTest(model=model, members=sorted(names)):
                    self.assertIn(shadow, names)

    def test_runtime_handlers_exist_and_fail_closed(self) -> None:
        import scripts.policy_improvement_exp1b_runtime as runtime

        self.assertTrue(callable(runtime.main))
        self.assertTrue(callable(runtime.bridge_main))
        with self.assertRaisesRegex(runtime.Exp1bRuntimeError, "packaged entrypoint"):
            runtime.main(
                [
                    "--policy-improvement-exp1b-training",
                    "--project-root", "/p",
                    "--reduced-study-protocol", "/a",
                    "--reduced-study-registry", "/b",
                    "--reduced-study-amendment", "/c",
                    "--execution-admission", "/c2",
                    "--base-policy-artifact", "/d",
                    "--base-policy-amendment", "/e",
                    "--evidence-root", "/g",
                    "--evidence-generation", "gen-0001",
                    "--row-id", "x",
                    "--seed", "1",
                ]
            )
        with self.assertRaisesRegex(runtime.Exp1bRuntimeError, "packaged entrypoint"):
            runtime.bridge_main(
                [
                    "--policy-improvement-exp1b-bridge",
                    "--project-root", "/p",
                    "--reduced-study-protocol", "/a",
                    "--reduced-study-registry", "/b",
                    "--reduced-study-amendment", "/c",
                    "--execution-admission", "/c2",
                    "--parent-protocol", "/d",
                    "--parent-registry", "/e",
                    "--parent-populations", "/f",
                    "--parent-theory-amendment", "/g",
                    "--evidence-root", "/j",
                    "--evidence-generation", "gen-0001",
                    "--seed-position", "0",
                ]
            )

    def test_entrypoints_dispatch_and_strip_markers(self) -> None:
        full = (_ROOT / "policy_improvement_full_entrypoint.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts.policy_improvement_exp1b_runtime", full)
        self.assertIn("marked = throughput_requested or exp1b_requested", full)
        self.assertIn("runtime_attestation=_EXP1B_ATTESTATION", full)
        bridge = (_ROOT / "policy_improvement_theory_bridge_entrypoint.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--policy-improvement-exp1b-bridge-entrypoint", bridge)
        self.assertIn("bridge_main", bridge)
        self.assertIn("Experiment 1B bridge marker is not launcher-owned", bridge)


class Exp1bWiringTest(unittest.TestCase):
    """Runtime profile, BUCK graph, and the untouched v2 refusal."""

    def test_runtime_roles_v2_is_still_a_closed_six_tuple(self) -> None:
        from scripts.policy_improvement_schema import RUNTIME_ROLES_V2

        self.assertEqual(len(tuple(RUNTIME_ROLES_V2)), 6)
        for purpose in (
            "policy-improvement-exp1b-training",
            "policy-improvement-exp1b-bridge",
        ):
            self.assertNotIn(purpose, RUNTIME_ROLES_V2)

    def test_runtime_profile_carries_every_exp1b_source_and_config(self) -> None:
        import phase4_runtime_profile as profile

        for name in (
            profile.POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
            profile.POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
        ):
            paths = set(profile._EXACT_PROFILE_PATHS[name])
            for required in (
                "configs/policy_improvement_exp1b/protocol.json",
                "configs/policy_improvement_exp1b/registry.json",
                "configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json",
                "scripts/policy_improvement_exp1_diagnostics.py",
                "scripts/policy_improvement_exp1b_aggregate.py",
                "scripts/policy_improvement_exp1b_bootstrap.py",
                "scripts/policy_improvement_exp1b_bridge.py",
                "scripts/policy_improvement_exp1b_schema.py",
                "scripts/policy_improvement_exp1b_session.py",
            ):
                with self.subTest(profile=name, path=required):
                    self.assertIn(required, paths)
                    self.assertTrue((_ROOT / required).is_file())

    def test_pars_declare_the_dynamically_imported_runtime(self) -> None:
        """Review item: the PARs must actually contain the exp1b modules."""

        buck = (_ROOT / "BUCK").read_text(encoding="utf-8")
        for binary in ("policy_improvement_full", "policy_improvement_theory_bridge"):
            start = buck.index(f'name = "{binary}"')
            block = buck[start : buck.index("\n)\n", start)]
            with self.subTest(binary=binary):
                self.assertIn(":policy_improvement_exp1b_runtime", block)
                self.assertIn("configs/policy_improvement_exp1b/*.json", block)
                self.assertIn(
                    "configs/policy_improvement_exp1b/amendments/*.json", block
                )

    def test_exp1b_test_target_declares_every_file_it_reads(self) -> None:
        """The wiring tests read by path, so the runfiles must contain them."""

        buck = (_ROOT / "BUCK").read_text(encoding="utf-8")
        start = buck.index('name = "test_policy_improvement_exp1b"')
        block = buck[start : buck.index("\n)\n", start)]
        source = (
            _ROOT / "tests" / "test_policy_improvement_exp1b_unittest.py"
        ).read_text(encoding="utf-8")
        for read_by_path in (
            "BUCK",
            "phase4_runtime_launcher.py",
            "phase4_runtime_profile.py",
            "policy_improvement_full_entrypoint.py",
            "policy_improvement_theory_bridge_entrypoint.py",
            "scripts/policy_improvement_theory_backend_v2.py",
        ):
            with self.subTest(path=read_by_path):
                self.assertIn(f'"{read_by_path}"', block)
                self.assertTrue((_ROOT / read_by_path).is_file())
        self.assertIn("_ROOT /", source)

    def test_v2_validation_bridge_refusal_is_left_in_force(self) -> None:
        source = (
            _ROOT / "scripts" / "policy_improvement_theory_backend_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Validation-bridge execution remains blocked until the canonical", source
        )
        self.assertIn("require_stage1_selection=True", source)
        for module in (
            "policy_improvement_exp1b_bridge",
            "policy_improvement_exp1b_schema",
            "policy_improvement_exp1b_runtime",
        ):
            with self.subTest(module=module):
                self.assertNotIn(module, source)

    def test_exp1b_modules_import_no_torch_and_no_v2_backend(self) -> None:
        for name in (
            "policy_improvement_exp1b_schema",
            "policy_improvement_exp1b_bridge",
            "policy_improvement_exp1b_session",
            "policy_improvement_exp1b_aggregate",
            "policy_improvement_exp1b_bootstrap",
            "policy_improvement_exp1b_runtime",
        ):
            with self.subTest(module=name):
                tree = ast.parse(
                    (_ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
                )
                modules = {
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                } | {
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                self.assertNotIn(
                    "scripts.policy_improvement_theory_backend_v2", modules
                )
                self.assertNotIn("torch", modules)


class Exp1bExtremeArithmeticTest(unittest.TestCase):
    """Finding 10: the exact-then-round paths must survive binary64 extremes."""

    def _octet_with_gaps(self, tmp: str, gaps: tuple[float, ...]) -> Exp1bSealedOctet:
        """Serve a real octet whose evaluator produces the requested gaps.

        Editing a persisted payload is no longer a way to get here: the payload
        digest is recomputed from its own scalars, so a changed gap is refused.
        The gap therefore has to come out of the arithmetic, which is what makes
        this an end-to-end test rather than a record-substitution one.
        """

        fixture = _RouteFixture(Path(tmp))
        route = fixture.open()
        for offset, item in enumerate(fixture.sealed):
            route.serve(
                seed_position=item["seed_position"],
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_gap_evaluator(gaps[offset]),
            )
        return route.sealed_octet()

    def test_eight_maximal_gaps_aggregate_without_an_overflowing_total(self) -> None:
        """Kills the `sum(gaps)/8` mutant: the naive total is +inf."""

        self.assertEqual(sum((1e308,) * UNITS), math.inf)
        with TemporaryDirectory() as tmp:
            octet = self._octet_with_gaps(tmp, (1e308,) * UNITS)
            result = aggregate_exp1b_result(octet)
            observed = result.mean_signed_gap
            self.assertTrue(math.isfinite(observed))
            self.assertEqual(observed, result.signed_gaps[0])
            self.assertGreater(observed, 1e307)
            interval = result.interval
            self.assertEqual(interval["interval_lower"], observed)
            self.assertEqual(interval["interval_upper"], observed)
            for encoded in interval["replicate_means_hex"]:
                self.assertTrue(math.isfinite(float.fromhex(encoded)))

    def test_mixed_sign_maximal_gaps_average_to_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            magnitude = (1e308 * (1.0 - GAMMA)) / (1.0 - GAMMA)
            gaps = (1e308, -magnitude) * 4
            octet = self._octet_with_gaps(tmp, gaps)
            result = aggregate_exp1b_result(octet)
            self.assertEqual(result.signed_gaps, (magnitude, -magnitude) * 4)
            self.assertEqual(result.mean_signed_gap, 0.0)
            self.assertTrue(math.isfinite(result.interval["interval_lower"]))
            self.assertTrue(math.isfinite(result.interval["interval_upper"]))

    def test_type7_interpolation_never_forms_high_minus_low(self) -> None:
        """Kills the `low + h*(high-low)` mutant: that intermediate is +inf."""

        low, high = -1.7e308, 1.7e308
        self.assertEqual(high - low, math.inf)
        for probability, expected in (
            (0.0, low),
            (1.0, high),
            (0.5, 0.0),
            (0.25, -8.5e307),
            (0.75, 8.5e307),
        ):
            with self.subTest(probability=probability):
                observed = type7_quantile([low, high], probability)
                self.assertTrue(math.isfinite(observed))
                self.assertEqual(observed, expected)

    def test_type7_endpoints_stay_inside_the_order_statistics(self) -> None:
        ordered = [-1.7e308, -1.0, 0.0, 1.7e308]
        for probability in (0.025, 0.1, 0.5, 0.9, 0.975):
            with self.subTest(probability=probability):
                observed = type7_quantile(ordered, probability)
                self.assertTrue(math.isfinite(observed))
                self.assertGreaterEqual(observed, ordered[0])
                self.assertLessEqual(observed, ordered[-1])

    def test_exact_once_rounding_differs_from_scaled_fsum(self) -> None:
        """Kills `math.fsum(v / n for v in values)`.

        For ``n = 8`` the division is by a power of two and therefore exact for
        every normal input, so scaled ``fsum`` agrees with the exact-rational
        mean everywhere in the normal range -- which is why the overflow cases
        do not distinguish them. In the subnormal range ``v / 8`` rounds, and the
        two rules part company by one ULP.
        """

        values = tuple(5e-324 * k for k in (1, 2, 3, 4, 5, 6, 7, 9))
        exact = float(sum(Fraction(value) for value in values) / len(values))
        scaled = math.fsum(value / len(values) for value in values)
        self.assertNotEqual(exact.hex(), scaled.hex())
        self.assertEqual(_exact_mean(values).hex(), exact.hex())

        # And the same rule, reached through the public interval.
        interval = paired_seed_percentile_interval(seed_gaps=values)
        self.assertEqual(interval.observed_mean.hex(), exact.hex())
        revalidate_replicate_vector(interval, seed_gaps=values)

    def test_exact_mean_matches_the_rational_oracle_on_wide_exponents(self) -> None:
        values = (
            1e308, -1e308, 5e-324, -5e-324, 1.0, -1.0, 2.0**-1000, -(2.0**-1000),
        )
        exact = float(sum(Fraction(value) for value in values) / len(values))
        self.assertEqual(_exact_mean(values).hex(), exact.hex())

    def test_maximal_seed_gaps_produce_a_finite_interval_directly(self) -> None:
        interval = paired_seed_percentile_interval(seed_gaps=(1e308,) * UNITS)
        self.assertEqual(interval.observed_mean, 1e308)
        self.assertEqual(interval.interval_lower, 1e308)
        self.assertEqual(interval.interval_upper, 1e308)
        revalidate_replicate_vector(interval, seed_gaps=(1e308,) * UNITS)


class Exp1bVectorRevalidationTest(unittest.TestCase):
    """Finding 9/10: persisted-vector revalidation is bit-exact and complete."""

    GAPS = (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 6.0)

    def _interval(self) -> Any:
        return paired_seed_percentile_interval(seed_gaps=self.GAPS)

    def test_a_clean_interval_revalidates(self) -> None:
        revalidate_replicate_vector(self._interval(), seed_gaps=self.GAPS)

    def test_integral_float_and_boolean_aliases_are_rejected(self) -> None:
        """Finding 10: `10000.0 == 10000` and `True == 1` in Python."""

        interval = self._interval()
        for field, value in (
            ("replicates", float(REPLICATES)),
            ("units", float(UNITS)),
            ("bootstrap_seed", float(BOOTSTRAP_SEED)),
            ("observed_mean", 1),
            ("confidence_level", 1),
            ("interval_lower", 0),
            ("interval_upper", 0),
        ):
            with self.subTest(field=field):
                with self.assertRaises(Exp1bBootstrapError):
                    revalidate_replicate_vector(
                        replace(interval, **{field: value}), seed_gaps=self.GAPS
                    )
        integral = tuple(
            int(value) if value == int(value) else value
            for value in interval.replicate_means
        )
        with self.assertRaisesRegex(Exp1bBootstrapError, "binary64 float"):
            revalidate_replicate_vector(
                replace(interval, replicate_means=integral), seed_gaps=self.GAPS
            )

    def test_signed_zero_substitution_is_rejected(self) -> None:
        """Kills a numeric-only comparison: +0.0 == -0.0 in binary64.

        The digest is recomputed over the flipped vector, so the substitution is
        internally consistent and the digest check passes. Only the canonical
        ``float.hex()`` comparison can reject it. An earlier version of this test
        left the old digest in place and therefore stopped at the digest check,
        which a numeric-comparison mutant survived.
        """

        gaps = (0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0)
        interval = paired_seed_percentile_interval(seed_gaps=gaps)
        vector = list(interval.replicate_means_hex)
        zeros = [
            offset
            for offset, encoded in enumerate(vector)
            if float.fromhex(encoded) == 0.0 and not encoded.startswith("-")
        ]
        self.assertTrue(zeros, "fixture produced no positive zero to flip")
        flipped = float(len(zeros))
        for offset in zeros:
            vector[offset] = (-0.0).hex()
        self.assertEqual(flipped, float(len(zeros)))
        substituted = tuple(vector)
        decoded = decode_replicate_vector(substituted)
        # Numerically identical to the genuine vector...
        self.assertEqual(
            [float.fromhex(item) for item in substituted],
            [float.fromhex(item) for item in interval.replicate_means_hex],
        )
        self.assertEqual(list(decoded), list(interval.replicate_means))
        # ...and now carrying a digest that genuinely covers its own bytes.
        consistent = replace(
            interval,
            replicate_means=decoded,
            replicate_means_hex=substituted,
            replicate_vector_sha256=replicate_digest(decoded),
        )
        self.assertNotEqual(
            consistent.replicate_vector_sha256, interval.replicate_vector_sha256
        )
        # The specific refusal matters. A numeric-comparison mutant still
        # rejects this document, but only at the trailing digest-field
        # comparison; the bitwise vector check is the one that has to fire, and
        # asserting its message is what makes the mutant visible.
        with self.assertRaisesRegex(
            Exp1bBootstrapError, "does not reproduce from its seed gaps"
        ):
            revalidate_replicate_vector(consistent, seed_gaps=gaps)

    def test_a_recorded_digest_that_does_not_cover_the_vector_is_rejected(self) -> None:
        interval = self._interval()
        with self.assertRaises(Exp1bBootstrapError):
            revalidate_replicate_vector(
                replace(interval, replicate_vector_sha256=_digest("other")),
                seed_gaps=self.GAPS,
            )

    def test_every_frozen_metadata_field_is_revalidated(self) -> None:
        """A no-op revalidation would let each of these through."""

        interval = self._interval()
        for field, value in (
            ("confidence_level", 0.9),
            ("lower_probability", 0.05),
            ("upper_probability", 0.95),
            ("replicates", REPLICATES - 1),
            ("units", UNITS - 1),
            ("bootstrap_seed", BOOTSTRAP_SEED + 1),
            ("rng_namespace", _OTHER_NAMESPACE),
            ("scheme", "another-scheme"),
            ("interval_convention", "another-convention"),
            ("replicate_encoding", "float_decimal_v1"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(Exp1bBootstrapError):
                    revalidate_replicate_vector(
                        replace(interval, **{field: value}), seed_gaps=self.GAPS
                    )

    def test_a_truncated_vector_is_rejected(self) -> None:
        interval = self._interval()
        with self.assertRaises(Exp1bBootstrapError):
            revalidate_replicate_vector(
                replace(
                    interval,
                    replicate_means_hex=interval.replicate_means_hex[:-1],
                ),
                seed_gaps=self.GAPS,
            )

    def test_revalidation_rejects_a_vector_from_other_gaps(self) -> None:
        interval = self._interval()
        other = paired_seed_percentile_interval(seed_gaps=tuple(
            value + 1.0 for value in self.GAPS
        ))
        with self.assertRaises(Exp1bBootstrapError):
            revalidate_replicate_vector(
                replace(
                    interval,
                    replicate_means_hex=other.replicate_means_hex,
                    replicate_vector_sha256=other.replicate_vector_sha256,
                ),
                seed_gaps=self.GAPS,
            )


class Exp1bAdmissionTransitionTest(unittest.TestCase):
    """Finding 6: the transition that flips both blocked slots, and its gates."""

    def setUp(self) -> None:
        self.protocol = _load(_EXP1B / "protocol.json")
        self.registry = _load(_EXP1B / "registry.json")
        self.amendment = _load(_EXP1B / "amendments" / "reduced_study_exp1b.json")
        self.protocol_sha = exp1b_document_sha256(self.protocol)
        self.registry_sha = exp1b_document_sha256(self.registry)
        self.amendment_sha = exp1b_document_sha256(self.amendment)

    def _document(self, **overrides: Any) -> dict[str, Any]:
        document = _admission_document(
            protocol_sha256=self.protocol_sha,
            registry_sha256=self.registry_sha,
            prior_amendment_sha256=self.amendment_sha,
        )
        document.update(overrides)
        return document

    def _validated(self, **overrides: Any) -> Any:
        return validate_exp1b_admission(
            self._document(**overrides),
            protocol_sha256=self.protocol_sha,
            registry_sha256=self.registry_sha,
            prior_amendment_sha256=self.amendment_sha,
        )

    def test_the_committed_protocol_still_blocks_execution(self) -> None:
        """The on-disk document records the study as blocked, and stays that way."""

        checked = validate_exp1b_protocol(self.protocol)
        self.assertEqual(checked["base_policy_artifact"]["status"], "unavailable")
        self.assertEqual(checked["runtime_authorization"]["status"], "unavailable")
        self.assertIs(checked["execution_gate"]["execution_allowed"], False)
        self.assertEqual(checked["amendments"], [])

    def test_admission_resolves_both_slots_in_memory_only(self) -> None:
        admission = self._validated()
        effective = apply_exp1b_admission(self.protocol, admission)
        self.assertEqual(effective["base_policy_artifact"]["status"], "available")
        self.assertEqual(effective["runtime_authorization"]["status"], "available")
        self.assertIs(effective["execution_gate"]["execution_allowed"], True)
        self.assertEqual(effective["amendments"], [admission.admission_sha256])
        # The document on disk is untouched.
        self.assertEqual(
            exp1b_document_sha256(_load(_EXP1B / "protocol.json")), self.protocol_sha
        )

    def test_admission_must_cite_the_documents_in_hand(self) -> None:
        for field in (
            "protocol_sha256",
            "registry_sha256",
            "prior_amendment_sha256",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(Exp1bSchemaError, "does not admit"):
                    validate_exp1b_admission(
                        self._document(**{field: _digest("stale")}),
                        protocol_sha256=self.protocol_sha,
                        registry_sha256=self.registry_sha,
                        prior_amendment_sha256=self.amendment_sha,
                    )

    def test_admission_for_another_protocol_revision_is_refused(self) -> None:
        admission = self._validated()
        moved = {**self.protocol, "status": "registered-something-else"}
        with self.assertRaisesRegex(Exp1bSchemaError, "does not admit this protocol"):
            apply_exp1b_admission(moved, admission)

    def test_admission_names_exactly_one_purpose_and_gate(self) -> None:
        with self.assertRaisesRegex(Exp1bSchemaError, "identity differs"):
            self._validated(purpose="policy-improvement-v2-confirmatory")
        with self.assertRaisesRegex(Exp1bSchemaError, "different execution gate"):
            self._validated(
                execution_environment={
                    "environment_variable": "RUN_SOMETHING_ELSE",
                    "required_value": "1",
                }
            )

    def test_admission_may_resolve_only_the_two_registered_slots(self) -> None:
        for slots in ([], ["base_policy_artifact"], list(reversed(ADMISSIBLE_SLOTS))):
            with self.subTest(slots=slots):
                with self.assertRaisesRegex(Exp1bSchemaError, "exactly the two"):
                    self._validated(resolved_slots=slots)

    def test_an_unavailable_slot_cannot_be_admitted(self) -> None:
        with self.assertRaisesRegex(Exp1bSchemaError, "available owner input"):
            self._validated(
                base_policy_artifact={"status": "unavailable", "value": {}}
            )
        # And a {status, reason} slot is refused by field inventory, so a
        # regression to the unavailable shape cannot read as absent.
        with self.assertRaisesRegex(Exp1bSchemaError, "field inventory differs"):
            self._validated(
                base_policy_artifact={"status": "unavailable", "reason": "x"}
            )

    def test_execution_allowed_must_be_exactly_true(self) -> None:
        for value in (False, 1, "true", None):
            with self.subTest(value=value):
                with self.assertRaises(Exp1bSchemaError):
                    self._validated(execution_allowed=value)

    def test_the_execution_environment_gate_is_read_from_the_process(self) -> None:
        admission = self._validated()
        admission.require_execution_environment({EXECUTION_ENVIRONMENT_VARIABLE: "1"})
        for environment in ({}, {EXECUTION_ENVIRONMENT_VARIABLE: "0"},
                            {EXECUTION_ENVIRONMENT_VARIABLE: "true"}):
            with self.subTest(environment=environment):
                with self.assertRaisesRegex(Exp1bSchemaError, "execution requires"):
                    admission.require_execution_environment(environment)

    def test_the_admission_binds_each_role_separately(self) -> None:
        """Review reproducer: one runtime hash could not authorize two PARs."""

        admission = self._validated()
        full = {
            "source_git_commit": _COMMIT,
            "runtime_sha256": _RUNTIME,
            "launcher_sha256": _LAUNCHER,
            "runtime_authorization_sha256": _AUTHORIZATION,
        }
        theory = {
            "source_git_commit": _COMMIT,
            "runtime_sha256": _THEORY_RUNTIME,
            "launcher_sha256": _THEORY_LAUNCHER,
            "runtime_authorization_sha256": _THEORY_AUTHORIZATION,
        }
        admission.require_attestation(role=FULL_ROLE, **full)
        admission.require_attestation(role=THEORY_BRIDGE_ROLE, **theory)
        # Neither role may present the other's identity.
        with self.assertRaisesRegex(Exp1bSchemaError, "different runtime_sha256"):
            admission.require_attestation(role=FULL_ROLE, **theory)
        with self.assertRaisesRegex(Exp1bSchemaError, "different runtime_sha256"):
            admission.require_attestation(role=THEORY_BRIDGE_ROLE, **full)
        for field in full:
            forged = "f" * (40 if field == "source_git_commit" else 64)
            with self.subTest(field=field):
                with self.assertRaisesRegex(Exp1bSchemaError, f"different {field}"):
                    admission.require_attestation(
                        role=FULL_ROLE, **{**full, field: forged}
                    )
        with self.assertRaisesRegex(Exp1bSchemaError, "authorizes no"):
            admission.authorization_for("policy-improvement-audit")

    def test_one_runtime_cannot_authorize_both_roles(self) -> None:
        document = self._document()
        authorization = json.loads(json.dumps(document["runtime_authorization"]))
        authorization["value"][THEORY_BRIDGE_ROLE] = dict(
            authorization["value"][FULL_ROLE]
        )
        with self.assertRaisesRegex(Exp1bSchemaError, "distinct binaries"):
            self._validated(runtime_authorization=authorization)


class Exp1bRuntimeGateTest(unittest.TestCase):
    """Finding 6: both stages run the same four gates, in the same order."""

    def _study(self, tmp: Path, **admission_overrides: Any) -> dict[str, Any]:
        tmp = tmp / "producer"
        tmp.mkdir(parents=True, exist_ok=True)
        protocol = _load(_EXP1B / "protocol.json")
        registry = _load(_EXP1B / "registry.json")
        amendment = _load(_EXP1B / "amendments" / "reduced_study_exp1b.json")
        paths = {}
        for name, document in (
            ("protocol.json", protocol),
            ("registry.json", registry),
            ("amendment.json", amendment),
        ):
            path = tmp / name
            path.write_bytes(canonical_json_bytes(document) + b"\n")
            paths[name] = path
        admission = _admission_document(
            protocol_sha256=exp1b_document_sha256(protocol),
            registry_sha256=exp1b_document_sha256(registry),
            prior_amendment_sha256=exp1b_document_sha256(amendment),
        )
        admission.update(admission_overrides)
        # Outside the producer root, like a signed owner artifact must be.
        owner = tmp.parent / "owner"
        owner.mkdir(parents=True, exist_ok=True)
        admission_path = owner / "admission.json"
        admission_path.write_bytes(canonical_json_bytes(admission) + b"\n")
        return {
            "protocol_path": paths["protocol.json"],
            "registry_path": paths["registry.json"],
            "amendment_path": paths["amendment.json"],
            "admission_path": admission_path,
            "producer_root": tmp,
            "role": FULL_ROLE,
        }

    def test_admitting_the_study_passes_every_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            arguments = self._study(Path(tmp))
            study = exp1b_runtime.admit_exp1b_study(
                **arguments,
                attestation=_attestation(),
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )
            self.assertEqual(
                study.effective_protocol["execution_gate"]["execution_allowed"], True
            )
            self.assertEqual(study.admission.purpose, EXECUTION_PURPOSE)

    def test_a_missing_admission_refuses_both_stages(self) -> None:
        """The unavailable owner artifact is a hard stop, not a warning."""

        with TemporaryDirectory() as tmp:
            arguments = self._study(Path(tmp))
            arguments["admission_path"].unlink()
            with self.assertRaises(exp1b_runtime.Exp1bRuntimeError):
                exp1b_runtime.admit_exp1b_study(
                    **arguments,
                    attestation=_attestation(),
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )

    def test_the_execution_environment_gate_blocks_admission(self) -> None:
        with TemporaryDirectory() as tmp:
            arguments = self._study(Path(tmp))
            for environment in ({}, {EXECUTION_ENVIRONMENT_VARIABLE: "0"}):
                with self.subTest(environment=environment):
                    with self.assertRaisesRegex(
                        exp1b_runtime.Exp1bRuntimeError, "execution requires"
                    ):
                        exp1b_runtime.admit_exp1b_study(
                            **arguments,
                            attestation=_attestation(),
                            environment=environment,
                        )

    def test_an_attestation_the_admission_did_not_authorize_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            arguments = self._study(Path(tmp))
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "authorized a different"
            ):
                exp1b_runtime.admit_exp1b_study(
                    **arguments,
                    attestation=_attestation(runtime_sha256="e" * 64),
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )

    def test_a_document_outside_the_producer_root_is_refused(self) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as other:
            arguments = self._study(Path(tmp))
            stray = Path(other) / "protocol.json"
            stray.write_bytes(arguments["protocol_path"].read_bytes())
            arguments["protocol_path"] = stray
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "outside the producer root"
            ):
                exp1b_runtime.admit_exp1b_study(
                    **arguments,
                    attestation=_attestation(),
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )


class Exp1bEvidenceDurabilityTest(unittest.TestCase):
    """Finding 4: one no-replace path per artifact, serialized across processes."""

    def test_the_schedule_path_is_derived_not_supplied(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            self.assertEqual(
                generation.schedule_path, generation.directory / "schedule_state.json"
            )
            self.assertEqual(
                generation.payload_path(0),
                generation.directory / "payloads" / "seed-0.json",
            )
            # There is no parameter to point either of them elsewhere.
            import inspect

            self.assertEqual(
                list(inspect.signature(open_evidence_generation).parameters),
                ["evidence_root", "generation_id"],
            )

    def test_an_unsafe_generation_id_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            for bad in ("..", ".", "a/b", "", "gen 1", "gen\u00e9"):
                with self.subTest(generation_id=bad):
                    with self.assertRaisesRegex(
                        Exp1bEvidenceError, "safe identifier"
                    ):
                        open_evidence_generation(
                            evidence_root=Path(tmp), generation_id=bad
                        )

    def test_a_group_writable_generation_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.generation_directory.chmod(0o775)
            try:
                with self.assertRaisesRegex(
                    Exp1bEvidenceError, "group- or world-writable"
                ):
                    fixture.generation()
            finally:
                fixture.generation_directory.chmod(0o755)

    def test_a_payload_is_never_replaced(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            generation.write_payload(0, {"schema_name": "x", "value": 1})
            with self.assertRaisesRegex(Exp1bEvidenceError, "never replaced"):
                generation.write_payload(0, {"schema_name": "x", "value": 2})

    def test_a_deleted_payload_invalidates_a_served_slot(self) -> None:
        """Finding 4: durable state and durable payloads must agree."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            first = fixture.sealed[0]
            route.serve(
                seed_position=0,
                seed=first["seed"],
                run_id=first["run_id"],
                checkpoint_sha256=first["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            fixture.generation().payload_path(0).unlink()
            with self.assertRaisesRegex(
                Exp1bScheduleStateError, "no persisted payload"
            ):
                fixture.open()

    def test_a_renamed_payload_invalidates_a_served_slot(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            first = fixture.sealed[0]
            route.serve(
                seed_position=0,
                seed=first["seed"],
                run_id=first["run_id"],
                checkpoint_sha256=first["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            payload_path = fixture.generation().payload_path(0)
            payload_path.rename(payload_path.with_name("seed-0.json.bak"))
            with self.assertRaisesRegex(
                Exp1bScheduleStateError, "no persisted payload"
            ):
                fixture.open()

    def test_a_crash_between_payload_and_served_marker_is_recoverable(self) -> None:
        """The pre-emission crash window leaves a retryable, counted attempt."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            first = fixture.sealed[0]
            generation = fixture.generation()
            # Simulate the crash: the payload landed, the marker did not.
            document = json.loads(generation.schedule_path.read_text())
            route.serve(
                seed_position=0,
                seed=first["seed"],
                run_id=first["run_id"],
                checkpoint_sha256=first["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            self.assertTrue(generation.has_payload(0))
            generation.schedule_path.write_bytes(
                canonical_json_bytes(document) + b"\n"
            )
            # Review reproducer: the retry used to die on the O_EXCL payload
            # and strand the generation. The durable artifact is now adopted.
            reopened = fixture.open()
            self.assertEqual(reopened.access_state, "bridge_open")
            self.assertEqual(reopened.served_positions, (0,))
            # And the adopted slot is served, so a retry is permanently refused.
            with self.assertRaisesRegex(Exp1bSchemaError, "post-payload retry"):
                reopened.serve(
                    seed_position=0,
                    seed=first["seed"],
                    run_id=first["run_id"],
                    checkpoint_sha256=first["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **_evaluator(0),
                )
            # The reconciliation is itself durable.
            self.assertEqual(fixture.open().served_positions, (0,))

    def test_concurrent_writers_are_serialized_and_one_loses(self) -> None:
        """Two OS processes racing on the same seed: exactly one payload survives.

        Real processes, not threads: ``flock`` is advisory per open file
        description and ``O_EXCL`` is a kernel guarantee, so an in-process race
        would prove nothing about either. ``fork`` rather than ``spawn`` so the
        test is hermetic -- it needs no importable ``__main__`` and no second
        interpreter on PATH.
        """

        import multiprocessing

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()
            context = multiprocessing.get_context("fork")
            barrier = context.Barrier(2)
            results = context.Queue()
            workers = [
                context.Process(
                    target=_race_one_payload,
                    args=(
                        str(fixture.evidence_root),
                        fixture.generation_id,
                        barrier,
                        results,
                    ),
                )
                for _ in range(2)
            ]
            for worker in workers:
                worker.start()
            outcomes = sorted(results.get(timeout=120) for _ in workers)
            for worker in workers:
                worker.join(timeout=120)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(outcomes, ["refused", "wrote"])
            self.assertTrue(fixture.generation().has_payload(0))

    def test_the_lock_is_held_across_a_whole_transition(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            with generation.exclusive():
                self.assertTrue(generation.lock_path.exists())


def _publication_inputs(fixture: Any, route: Any, result: Any) -> dict[str, Any]:
    """The keyword inputs durable publication now takes.

    Publication no longer accepts a document. It takes the authenticated route
    and the study documents and serializes, validates, and audits the result
    itself, so a caller has no bytes to substitute.
    """

    return {
        "result": result,
        "route": route,
        "protocol": fixture.protocol,
        "registry": fixture.registry,
        "amendment": fixture.amendment,
        "admission": getattr(fixture, "admission_document", None),
    }


def _race_one_payload(root: str, generation_id: str, barrier: Any, results: Any) -> None:
    """Worker for the concurrent-writer test; module level so it is picklable."""

    generation = open_evidence_generation(
        evidence_root=Path(root), generation_id=generation_id
    )
    barrier.wait(timeout=60)
    try:
        generation.write_payload(0, {"schema_name": "race", "value": 1})
    except Exp1bEvidenceError:
        results.put("refused")
    else:
        results.put("wrote")


class Exp1bDurablePublicationTest(unittest.TestCase):
    """Finding 1: durable publication plus an independently validating consumer."""

    def _published(self, tmp: str) -> tuple[Any, Any, Any]:
        fixture = _RouteFixture(Path(tmp))
        route = fixture.open()
        _serve_all(route, fixture)
        result = aggregate_exp1b_result(route.sealed_octet())
        document = validated_exp1b_document(
            result,
            route=route,
            protocol=fixture.protocol,
            registry=fixture.registry,
            amendment=fixture.amendment,
            provenance=fixture.provenance,
            admission=fixture.admission_document,
        )
        route.generation.publish_result(**_publication_inputs(fixture, route, result))
        self._route = route
        return fixture, result, document

    def test_the_result_is_durable_and_reads_back_bit_for_bit(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, document = self._published(tmp)
            generation = fixture.generation()
            reloaded, digest = generation.read_published_result()
            self.assertEqual(reloaded, document)
            self.assertEqual(digest, result.document_sha256())
            self.assertTrue(generation.result_path.is_file())
            self.assertTrue(generation.result_digest_path.is_file())

    def test_a_second_publication_is_permanently_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, document = self._published(tmp)
            # An identical republication is a crash-retry and is adopted.
            fixture.generation().publish_result(
                **_publication_inputs(fixture, self._route, result)
            )
            # A different result for the same generation cannot even be
            # serialized past the validator, which re-derives every claim from
            # the durable generation.
            forged = replace(result, attempt_counts=(99,) * UNITS)
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "did not pass publication validation"
            ):
                fixture.generation().publish_result(
                    **_publication_inputs(fixture, self._route, forged)
                )
            # And if the durable document is edited underneath, the no-replace
            # floor refuses to overwrite it with the validated bytes.
            edited = document.replace(b'"seed_count":8', b'"seed_count":9')
            self.assertNotEqual(edited, document)
            generation = fixture.generation()
            generation.result_path.unlink()
            generation.result_path.write_bytes(edited)
            with self.assertRaisesRegex(Exp1bEvidenceError, "cannot be adopted"):
                generation.publish_result(
                    **_publication_inputs(fixture, self._route, result)
                )

    def test_a_tampered_published_result_is_caught_on_read(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, _result, document = self._published(tmp)
            generation = fixture.generation()
            edited = document.replace(b'"seed_count":8', b'"seed_count":7')
            self.assertNotEqual(edited, document)
            generation.result_path.write_bytes(edited)
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "does not match its recorded digest"
            ):
                generation.read_published_result()

    def test_the_independent_consumer_re_derives_every_number(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, document = self._published(tmp)
            report = audit_exp1b_result_document(
                document,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                population_registry=_parent_populations(),
            )
            self.assertEqual(report.document_sha256, result.document_sha256())
            self.assertEqual(report.recomputed_mean_signed_gap, result.mean_signed_gap)
            self.assertEqual(
                report.recomputed_interval_lower, result.interval["interval_lower"]
            )
            self.assertEqual(
                report.recomputed_interval_upper, result.interval["interval_upper"]
            )
            self.assertIn("per_seed_bound_algebra", report.checks)
            self.assertIn("frozen_bootstrap_interval", report.checks)

    def test_the_consumer_rejects_a_bound_its_own_maxima_do_not_imply(self) -> None:
        """The point of an independent consumer: it recomputes, not re-reads."""

        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            tampered = result.as_document()
            rows = [dict(row) for row in tampered["seed_rows"]]
            rows[0]["finite_reference_bound"] = (
                rows[0]["finite_reference_bound"] + 1.0
            )
            tampered["seed_rows"] = rows
            payload = canonical_json_bytes(tampered) + b"\n"
            with self.assertRaisesRegex(
                Exp1bAuditError, "not what its own published maxima imply"
            ):
                audit_exp1b_result_document(
                    payload,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_the_consumer_rejects_a_mean_that_is_not_the_equal_seed_mean(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            tampered = result.as_document()
            tampered["mean_signed_gap"] = tampered["mean_signed_gap"] + 1.0
            payload = canonical_json_bytes(tampered) + b"\n"
            with self.assertRaisesRegex(Exp1bAuditError, "equal-seed mean"):
                audit_exp1b_result_document(
                    payload,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_the_consumer_rejects_edited_interval_endpoints(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            for field in ("interval_lower", "interval_upper"):
                with self.subTest(field=field):
                    tampered = result.as_document()
                    interval = dict(tampered["interval"])
                    interval[field] = interval[field] + 1.0
                    tampered["interval"] = interval
                    payload = canonical_json_bytes(tampered) + b"\n"
                    with self.assertRaisesRegex(
                        Exp1bAuditError, "endpoints differ"
                    ):
                        audit_exp1b_result_document(
                            payload,
                            protocol=fixture.protocol,
                            registry=fixture.registry,
                            amendment=fixture.amendment,
                            provenance=fixture.provenance,
                            population_registry=_parent_populations(),
                        )

    def test_the_consumer_rejects_a_true_access_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            for field in (
                "scientific_selection",
                "validation_select_access",
                "test_access",
            ):
                with self.subTest(field=field):
                    tampered = {**result.as_document(), field: True}
                    payload = canonical_json_bytes(tampered) + b"\n"
                    with self.assertRaisesRegex(Exp1bAuditError, "exactly false"):
                        audit_exp1b_result_document(
                            payload,
                            protocol=fixture.protocol,
                            registry=fixture.registry,
                            amendment=fixture.amendment,
                            provenance=fixture.provenance,
                            population_registry=_parent_populations(),
                        )

    def test_the_consumer_binds_both_role_identities_to_the_admission(self) -> None:
        """Finding 1: the auditor must re-derive both roles, not one."""

        with TemporaryDirectory() as tmp:
            fixture, _result, document = self._published(tmp)
            report = audit_exp1b_result_document(
                document,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                population_registry=_parent_populations(),
                admission=fixture.admission_document,
            )
            self.assertIn("role_scoped_admission_binding", report.checks)
            # Without the admission the audit still runs, and says so.
            partial = audit_exp1b_result_document(
                document,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                population_registry=_parent_populations(),
            )
            self.assertNotIn("role_scoped_admission_binding", partial.checks)

    def test_the_consumer_rejects_one_runtime_for_both_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            tampered = result.as_document()
            tampered["evaluator_attestation"] = {
                **dict(result.producer_attestation),
                "role": THEORY_BRIDGE_ROLE,
            }
            payload = canonical_json_bytes(tampered) + b"\n"
            with self.assertRaisesRegex(
                Exp1bAuditError, "same artifact"
            ):
                audit_exp1b_result_document(
                    payload,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_the_consumer_rejects_a_swapped_producer_identity(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            tampered = result.as_document()
            tampered["producer_attestation"] = {
                **dict(result.producer_attestation),
                "runtime_sha256": _digest("some-other-full-par"),
            }
            payload = canonical_json_bytes(tampered) + b"\n"
            with self.assertRaisesRegex(
                Exp1bAuditError, "differs from the authenticated"
            ):
                audit_exp1b_result_document(
                    payload,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_the_two_result_readers_agree_on_the_schema_version(self) -> None:
        """A producer and a consumer that disagree here cannot both be right."""

        import scripts.policy_improvement_exp1b_aggregate as producer
        import scripts.policy_improvement_exp1b_auditor as consumer

        self.assertEqual(producer.RESULT_SCHEMA_NAME, consumer.RESULT_SCHEMA_NAME)
        self.assertEqual(
            producer.RESULT_SCHEMA_VERSION, consumer.RESULT_SCHEMA_VERSION
        )
        self.assertEqual(producer.RESULT_SCHEMA_VERSION, 2)

    def test_a_pre_split_result_document_is_refused(self) -> None:
        """A version-1 document cannot claim the version-2 contract.

        The pre-split shape carried four flat identity fields and no per-seed
        secondary diagnostics. Both inventories changed incompatibly, so the
        version bump has to be load-bearing rather than cosmetic.
        """

        with TemporaryDirectory() as tmp:
            fixture, result, _document = self._published(tmp)
            legacy = result.as_document()
            producer = dict(result.producer_attestation)
            legacy.pop("producer_attestation")
            legacy.pop("evaluator_attestation")
            legacy.update(
                {
                    "schema_version": 1,
                    "source_git_commit": producer["source_git_commit"],
                    "runtime_sha256": producer["runtime_sha256"],
                    "launcher_sha256": producer["launcher_sha256"],
                    "runtime_authorization_sha256": producer[
                        "runtime_authorization_sha256"
                    ],
                }
            )
            legacy["seed_rows"] = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "secondary_diagnostics"
                }
                for row in legacy["seed_rows"]
            ]
            payload = canonical_json_bytes(legacy) + b"\n"
            with self.assertRaisesRegex(Exp1bAuditError, "field inventory differs"):
                audit_exp1b_result_document(
                    payload,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_the_consumer_binds_the_study_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, _result, document = self._published(tmp)
            with self.assertRaisesRegex(Exp1bAuditError, "study document"):
                audit_exp1b_result_document(
                    document,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance={**fixture.provenance, "access_state": "closed"},
                    population_registry=_parent_populations(),
                )

    def test_the_consumer_imports_no_producer_module(self) -> None:
        """An audit that shared the producer's objects would agree with its bugs."""

        tree = ast.parse(
            (_ROOT / "scripts" / "policy_improvement_exp1b_auditor.py").read_text(
                encoding="utf-8"
            )
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in (
            "scripts.policy_improvement_exp1b_aggregate",
            "scripts.policy_improvement_exp1b_bridge",
            "scripts.policy_improvement_exp1b_evidence",
            "scripts.policy_improvement_exp1b_theory_backend",
            "scripts.policy_improvement_exp1_diagnostics",
            "torch",
        ):
            self.assertNotIn(forbidden, imported)
        self.assertEqual(
            imported,
            {
                "__future__",
                "argparse",
                "math",
                "sys",
                "collections.abc",
                "dataclasses",
                "pathlib",
                "typing",
                "scripts.policy_improvement_exp1b_bootstrap",
                "scripts.policy_improvement_exp1b_schema",
                "scripts.policy_improvement_schema",
            },
        )


class Exp1bResultImmutabilityTest(unittest.TestCase):
    """Finding 9: results are immutable in depth and validated at publication."""

    def _result(self, tmp: str) -> tuple[Any, Any]:
        fixture = _RouteFixture(Path(tmp))
        route = fixture.open()
        _serve_all(route, fixture)
        self._route = route
        return fixture, aggregate_exp1b_result(route.sealed_octet())

    def _publish(self, fixture: Any, result: Any) -> bytes:
        document = validated_exp1b_document(
            result,
            route=self._route,
            protocol=fixture.protocol,
            registry=fixture.registry,
            amendment=fixture.amendment,
            provenance=fixture.provenance,
            admission=fixture.admission_document,
        )
        return document

    def test_the_interval_cannot_be_cleared_after_aggregation(self) -> None:
        with TemporaryDirectory() as tmp:
            _fixture, result = self._result(tmp)
            with self.assertRaises(TypeError):
                result.interval["status"] = "unavailable"
            with self.assertRaises(TypeError):
                del result.interval["interval_lower"]

    def test_the_replicate_vector_cannot_be_truncated_in_place(self) -> None:
        with TemporaryDirectory() as tmp:
            _fixture, result = self._result(tmp)
            vector = result.interval["replicate_means_hex"]
            self.assertIsInstance(vector, tuple)
            with self.assertRaises(AttributeError):
                vector.pop()  # type: ignore[attr-defined]

    def test_a_result_with_a_suppressed_interval_is_not_publishable(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result = self._result(tmp)
            suppressed = replace(result, interval={"status": "unavailable"})
            with self.assertRaisesRegex(
                Exp1bAggregateError, "no available interval"
            ):
                self._publish(fixture, suppressed)

    def test_a_result_with_a_truncated_vector_is_not_publishable(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result = self._result(tmp)
            interval = dict(result.interval)
            interval["replicate_means_hex"] = list(
                interval["replicate_means_hex"]
            )[:-1]
            with self.assertRaisesRegex(
                Exp1bAggregateError, "does not carry the full vector"
            ):
                self._publish(fixture, replace(result, interval=interval))

    def test_a_result_whose_vector_lost_its_digest_is_not_publishable(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result = self._result(tmp)
            interval = dict(result.interval)
            interval["replicate_vector_sha256"] = _digest("other")
            with self.assertRaisesRegex(
                Exp1bAggregateError, "does not match its recorded digest"
            ):
                self._publish(fixture, replace(result, interval=interval))

    def test_a_flipped_access_flag_is_not_publishable(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, result = self._result(tmp)
            for field in (
                "scientific_selection",
                "validation_select_access",
                "test_access",
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        Exp1bAggregateError, "must be false"
                    ):
                        self._publish(fixture, replace(result, **{field: True}))


def _theory_backend_inputs(fixture: Any, route: Any) -> dict[str, Any]:
    """The census and split identity the Stage B factory binds."""

    population = fixture.protocol["evaluation_population"]
    return {
        "census": route.census_members,
        "dataset_root": fixture.directory / "dataset",
        "evaluation_split": str(population["split"]),
        "evaluation_split_manifest_sha256": _digest("validation-split-manifest"),
    }


class Exp1bProductionBackendTest(unittest.TestCase):
    """Finding 1: concrete methods on both authenticated production backends."""

    def test_the_full_backend_declares_both_stage_a_methods(self) -> None:
        """Torch-free check: the methods exist, are concrete, and are on the class."""

        tree = ast.parse(
            (_ROOT / "policy_improvement_full_backend.py").read_text(encoding="utf-8")
        )
        sealed = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "SealedFullRunBackend"
        )
        methods = {
            node.name for node in sealed.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("prepare_exp1b_training", methods)
        self.assertIn("run_exp1b_training", methods)
        for name in ("prepare_exp1b_training", "run_exp1b_training"):
            body = next(
                node
                for node in sealed.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            statements = [
                node
                for node in body.body
                if not isinstance(node, ast.Expr)
                or not isinstance(node.value, ast.Constant)
            ]
            self.assertTrue(statements, f"{name} is a stub")
            self.assertFalse(
                any(isinstance(node, ast.Pass) for node in statements),
                f"{name} is a stub",
            )
            self.assertFalse(
                any(
                    isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == "NotImplementedError"
                    for node in ast.walk(body)
                ),
                f"{name} raises NotImplementedError",
            )

    def test_the_sealed_backend_reads_only_attributes_it_assigns(self) -> None:
        """Regression: run_exp1b_training read self._module, which never existed.

        Every test that exercises Stage A substitutes its own backend class, so
        nothing ever ran the real SealedFullRunBackend.run_exp1b_training body.
        The typo therefore reached production and raised AttributeError on the
        last line of the training loop -- after the full 10,000-interaction
        budget had been spent and with nothing sealed. self._module belongs to
        TorchLearnedRunEngine; this class stores the same object as
        self._training_module.

        Torch-free by construction: this reads the module as source, so it runs
        wherever the other checks in this class run.
        """

        tree = ast.parse(
            (_ROOT / "policy_improvement_full_backend.py").read_text(encoding="utf-8")
        )
        sealed = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "SealedFullRunBackend"
        )
        # FullRunBackend is a Protocol carrying no instance state, so everything
        # this class reads off self must be assigned or defined right here.
        assigned = {
            target.attr
            for node in ast.walk(sealed)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        defined = {
            node.name
            for node in sealed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        read = {
            node.attr
            for node in ast.walk(sealed)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Load)
        }
        self.assertEqual(
            sorted(read - assigned - defined),
            [],
            "SealedFullRunBackend reads an attribute it never assigns.",
        )

    def test_the_full_backend_publishes_the_registered_budget(self) -> None:
        source = (_ROOT / "policy_improvement_full_backend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_EXP1B_TERMINAL_ENVIRONMENT_INTERACTIONS = 10000", source)
        self.assertIn("_EXP1B_TRAINING_RECORD_COUNT = 1024", source)

    def test_the_duplicated_budget_constants_match_the_schema(self) -> None:
        """The Torch module cannot import the reduced-study namespace."""

        tree = ast.parse(
            (_ROOT / "policy_improvement_full_backend.py").read_text(encoding="utf-8")
        )
        values = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(
            values["_EXP1B_TERMINAL_ENVIRONMENT_INTERACTIONS"],
            TERMINAL_ENVIRONMENT_INTERACTIONS,
        )
        self.assertEqual(
            values["_EXP1B_TRAINING_RECORD_COUNT"], TRAINING_RECORD_COUNT
        )

    def test_the_theory_backend_refuses_an_unauthenticated_octet(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            attestation = {
                "source_git_commit": _COMMIT,
                "runtime_sha256": _RUNTIME,
                "launcher_sha256": _LAUNCHER,
                "runtime_authorization_sha256": _AUTHORIZATION,
            }
            backend = create_exp1b_theory_backend(
                runtime_attestation=attestation,
                authenticated_checkpoints=route.authenticated_checkpoints,
                effective_config_sha256=fixture.effective_config_sha256,
                census_ordering_sha256=route.census_ordering_sha256,
                **_theory_backend_inputs(fixture, route),
            )
            self.assertEqual(
                backend.exp1b_state_id_for(0, _digest("x")),
                exp1b_state_id_for(0, _digest("x")),
            )
            for bad in ((), route.authenticated_checkpoints[:7]):
                with self.subTest(count=len(bad)):
                    with self.assertRaisesRegex(
                        Exp1bTheoryBackendError, "eight authenticated checkpoints"
                    ):
                        create_exp1b_theory_backend(
                            runtime_attestation=attestation,
                            authenticated_checkpoints=bad,
                            effective_config_sha256=(
                                fixture.effective_config_sha256
                            ),
                            census_ordering_sha256=route.census_ordering_sha256,
                            **_theory_backend_inputs(fixture, route),
                        )
            with self.assertRaisesRegex(
                Exp1bTheoryBackendError, "another configuration"
            ):
                create_exp1b_theory_backend(
                    runtime_attestation=attestation,
                    authenticated_checkpoints=route.authenticated_checkpoints,
                    effective_config_sha256=_digest("other"),
                    census_ordering_sha256=route.census_ordering_sha256,
                    **_theory_backend_inputs(fixture, route),
                )

    def test_the_theory_backend_refuses_a_mismatched_attestation(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            backend = create_exp1b_theory_backend(
                runtime_attestation={
                    "source_git_commit": _COMMIT,
                    "runtime_sha256": _RUNTIME,
                    "launcher_sha256": _LAUNCHER,
                    "runtime_authorization_sha256": _AUTHORIZATION,
                },
                authenticated_checkpoints=route.authenticated_checkpoints,
                effective_config_sha256=fixture.effective_config_sha256,
                census_ordering_sha256=route.census_ordering_sha256,
                **_theory_backend_inputs(fixture, route),
            )
            with self.assertRaisesRegex(
                Exp1bTheoryBackendError, "differs from the runtime"
            ):
                backend.prepare_exp1b_bridge(
                    seed_position=0,
                    runtime_attestation=_attestation(runtime_sha256="e" * 64),
                    effective_config_sha256=fixture.effective_config_sha256,
                )

    def test_the_sealed_restore_api_exists_and_the_call_site_binds(self) -> None:
        """Finding 4: the Stage B adapter is implemented, not an external input."""

        import inspect

        signature = inspect.signature(
            _production_signature(
                "open_exp1b_sealed_evaluation_session", inside_class=None
            )
        )
        # Exactly the arguments the Stage B backend passes.
        call_site = {
            "checkpoint_path",
            "expected_checkpoint_sha256",
            "expected_checkpoint_size_bytes",
            "expected_model_state_sha256",
            "expected_effective_config_sha256",
            "expected_environment_interactions",
            "run_id",
            "seed",
            "seed_position",
            "census_ordering_sha256",
            "census",
            "depths",
            "evaluation_population",
            "dataset_root",
            "evaluation_split",
            "evaluation_split_manifest_sha256",
            "training_module",
        }
        self.assertEqual(set(signature.parameters), call_site)
        signature.bind(**{name: object() for name in call_site})

        tree = ast.parse(
            (_ROOT / "policy_improvement_full_backend.py").read_text(encoding="utf-8")
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "open_exp1b_sealed_evaluation_session"
        )
        body = [
            node
            for node in function.body
            if not isinstance(node, ast.Expr)
            or not isinstance(node.value, ast.Constant)
        ]
        self.assertGreater(len(body), 20, "restore adapter is a stub")
        self.assertFalse(
            any(isinstance(node, ast.Pass) for node in body),
            "restore adapter is a stub",
        )
        # It must return an evaluation session exposing the four callables.
        evaluator = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "Exp1bSealedEvaluationSession"
        )
        methods = {
            node.name for node in evaluator.body if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "endpoint_values",
            "action_values",
            "action_mask",
            "base_probabilities",
        ):
            self.assertIn(name, methods)

    def test_the_theory_backend_names_a_missing_restore_api(self) -> None:
        """Fail-closed boundary: the sealed-restore API is a runtime input."""

        class _NoOpener:
            pass

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            backend = Exp1bTheoryBackend(
                runtime_attestation={
                    "source_git_commit": _COMMIT,
                    "runtime_sha256": _RUNTIME,
                    "launcher_sha256": _LAUNCHER,
                    "runtime_authorization_sha256": _AUTHORIZATION,
                },
                authenticated_checkpoints=route.authenticated_checkpoints,
                effective_config_sha256=fixture.effective_config_sha256,
                census_ordering_sha256=route.census_ordering_sha256,
                full_backend_module=_NoOpener(),
                **_theory_backend_inputs(fixture, route),
            )
            with self.assertRaisesRegex(
                Exp1bTheoryBackendError, "open_exp1b_sealed_evaluation_session"
            ):
                backend.prepare_exp1b_bridge(
                    seed_position=0,
                    runtime_attestation=_attestation(),
                    effective_config_sha256=fixture.effective_config_sha256,
                )

    def test_the_state_id_derivation_binds_position_and_record(self) -> None:
        first = exp1b_state_id_for(0, _digest("a"))
        self.assertNotEqual(first, exp1b_state_id_for(1, _digest("a")))
        self.assertNotEqual(first, exp1b_state_id_for(0, _digest("b")))
        for bad_index in (-1, True, 1.0, "0"):
            with self.subTest(index=bad_index):
                with self.assertRaises(Exp1bSchemaError):
                    exp1b_state_id_for(bad_index, _digest("a"))  # type: ignore[arg-type]
        for bad_digest in ("", "zz", _digest("a").upper(), None):
            with self.subTest(digest=bad_digest):
                with self.assertRaises(Exp1bSchemaError):
                    exp1b_state_id_for(0, bad_digest)  # type: ignore[arg-type]

    def test_the_state_id_derivation_lives_outside_the_stage_b_backend(self) -> None:
        """Finding 9: the full PAR must not carry the Stage B restore module."""

        tree = ast.parse(
            (_ROOT / "scripts" / "policy_improvement_exp1b_runtime.py").read_text(
                encoding="utf-8"
            )
        )
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn(
            "scripts.policy_improvement_exp1b_theory_backend", imported
        )
        self.assertIn("scripts.policy_improvement_exp1b_schema", imported)

    def test_the_theory_entrypoint_does_not_pass_the_v2_factory(self) -> None:
        """Review reproducer: the v2 backend factory was passed as the backend."""

        tree = ast.parse(
            (_ROOT / "policy_improvement_theory_bridge_entrypoint.py").read_text(
                encoding="utf-8"
            )
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "exp1b_main"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {
            keyword.arg: keyword.value
            for keyword in calls[0].keywords
            if keyword.arg
        }
        self.assertIn("backend_factory", keywords)
        self.assertNotIn("backend", keywords)
        self.assertEqual(keywords["backend_factory"].id, "exp1b_backend_factory")


class Exp1bPackagingTest(unittest.TestCase):
    """Finding 8: exact profiles and Buck targets carry what is imported."""

    def setUp(self) -> None:
        self.buck = (_ROOT / "BUCK").read_text(encoding="utf-8")
        self.buck_tree = ast.parse(self.buck)

    def _target(self, name: str) -> dict[str, Any]:
        for node in ast.walk(self.buck_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg
                }
                value = keywords.get("name")
                if isinstance(value, ast.Constant) and value.value == name:
                    return keywords
        raise AssertionError(f"no BUCK target named {name!r}")

    def _deps(self, name: str) -> set[str]:
        keywords = self._target(name)
        deps = keywords.get("deps")
        if deps is None:
            return set()
        return {
            element.value
            for element in deps.elts
            if isinstance(element, ast.Constant)
        }

    def test_every_declared_library_label_exists(self) -> None:
        declared = set()
        for node in ast.walk(self.buck_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                for keyword in node.keywords:
                    if keyword.arg == "name" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        declared.add(f":{keyword.value.value}")
        referenced = set()
        for node in ast.walk(self.buck_tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith(":"):
                    referenced.add(node.value)
        missing = sorted(referenced - declared)
        self.assertEqual(missing, [])

    def test_both_pars_declare_every_exp1b_library(self) -> None:
        for par in ("policy_improvement_full", "policy_improvement_theory_bridge"):
            with self.subTest(par=par):
                self.assertIn(":policy_improvement_exp1b_runtime", self._deps(par))

    def test_the_theory_par_declares_the_stage_b_backend(self) -> None:
        self.assertIn(
            ":policy_improvement_exp1b_theory_backend",
            self._deps("policy_improvement_theory_bridge"),
        )

    def test_the_bridge_library_declares_its_direct_imports(self) -> None:
        deps = self._deps("policy_improvement_exp1b_bridge")
        for required in (
            ":policy_improvement_exp1_diagnostics",
            ":policy_improvement_exp1b_evidence",
            ":policy_improvement_exp1b_schema",
            ":policy_improvement_schema",
        ):
            self.assertIn(required, deps)

    def test_the_aggregate_library_declares_the_bridge(self) -> None:
        self.assertIn(
            ":policy_improvement_exp1b_bridge",
            self._deps("policy_improvement_exp1b_aggregate"),
        )

    def test_the_auditor_has_its_own_binary_and_no_producer_deps(self) -> None:
        deps = self._deps("policy_improvement_exp1b_auditor_bin")
        self.assertIn(":policy_improvement_exp1b_auditor", deps)
        for forbidden in (
            ":policy_improvement_full_backend",
            ":policy_improvement_exp1b_bridge",
            ":policy_improvement_exp1b_runtime",
        ):
            self.assertNotIn(forbidden, deps)

    def test_the_exact_profiles_carry_every_exp1b_module(self) -> None:
        for profile in (
            profile_module.POLICY_IMPROVEMENT_FULL_PROFILE_PATHS,
            profile_module.POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS,
        ):
            for required in (
                "scripts/policy_improvement_exp1b_runtime.py",
                "scripts/policy_improvement_exp1b_evidence.py",
                "scripts/policy_improvement_exp1b_bridge.py",
                "scripts/policy_improvement_exp1b_aggregate.py",
                "scripts/policy_improvement_exp1b_session.py",
                "scripts/policy_improvement_exp1b_schema.py",
                "scripts/policy_improvement_exp1b_bootstrap.py",
                "scripts/policy_improvement_exp1_diagnostics.py",
                "configs/policy_improvement_exp1b/protocol.json",
                "configs/policy_improvement_exp1b/registry.json",
                "configs/policy_improvement_exp1b/amendments/"
                "reduced_study_exp1b.json",
            ):
                with self.subTest(path=required):
                    self.assertIn(required, profile)

    def test_the_theory_profile_carries_the_stage_b_backend(self) -> None:
        self.assertIn(
            "scripts/policy_improvement_exp1b_theory_backend.py",
            profile_module.POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS,
        )

    def test_the_consumer_profiles_are_not_regressed(self) -> None:
        """Adding Experiment 1B must not change what audit/analysis package."""

        for profile in (
            profile_module.POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS,
            profile_module.POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS,
        ):
            for path in profile:
                self.assertNotIn("policy_improvement_exp1b", path)

    def test_the_auditor_profile_is_separate_and_minimal(self) -> None:
        profile = profile_module.POLICY_IMPROVEMENT_EXP1B_AUDIT_PROFILE_PATHS
        self.assertIn("scripts/policy_improvement_exp1b_auditor.py", profile)
        for producer in (
            "policy_improvement_full_backend.py",
            "policy_improvement_full_entrypoint.py",
            "policy_improvement_theory_bridge_entrypoint.py",
            "scripts/policy_improvement_exp1b_bridge.py",
            "scripts/policy_improvement_exp1b_runtime.py",
        ):
            self.assertNotIn(producer, profile)

    def _closure(self, target: str) -> set[str]:
        """Every source file in one Buck target's transitive dependency closure."""

        targets: dict[str, dict[str, Any]] = {}
        for node in ast.walk(self.buck_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            keywords = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg
            }
            name = keywords.get("name")
            if isinstance(name, ast.Constant):
                targets[str(name.value)] = keywords

        def literal_sources(keywords: dict[str, Any]) -> set[str]:
            found: set[str] = set()
            for key in ("srcs", "main_src"):
                value = keywords.get(key)
                if isinstance(value, ast.Constant):
                    found.add(str(value.value))
                elif isinstance(value, (ast.List, ast.Tuple)):
                    found.update(
                        str(element.value)
                        for element in value.elts
                        if isinstance(element, ast.Constant)
                    )
            return found

        seen: set[str] = set()
        sources: set[str] = set()
        stack = [target]
        while stack:
            current = stack.pop()
            if current in seen or current not in targets:
                continue
            seen.add(current)
            keywords = targets[current]
            sources |= literal_sources(keywords)
            deps = keywords.get("deps")
            if isinstance(deps, (ast.List, ast.Tuple)):
                for element in deps.elts:
                    if isinstance(element, ast.Constant) and str(
                        element.value
                    ).startswith(":"):
                        stack.append(str(element.value)[1:])
        return sources

    def test_each_par_closure_matches_its_exact_profile(self) -> None:
        """Finding 9: a selected source the profile omits fails archive validation.

        The full PAR used to reach the Stage B theory backend transitively
        through the shared Experiment 1B runtime target, while only the theory
        profile listed it. Archive validation selects every
        ``scripts/policy_improvement_*.py`` source and requires exact equality,
        so the built PAR would have been rejected.

        Scoped to the ``scripts/policy_improvement_*.py`` rule. Several
        root-level producer modules are selected by the same predicate and are
        carried by both PARs without appearing in these two profile constants;
        that predates this change and is not what this test is about.
        """

        def scoped(path: str) -> bool:
            return path.startswith("scripts/policy_improvement_") and path.endswith(
                ".py"
            )

        for target, profile in (
            (
                "policy_improvement_full",
                profile_module.POLICY_IMPROVEMENT_FULL_PROFILE_PATHS,
            ),
            (
                "policy_improvement_theory_bridge",
                profile_module.POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS,
            ),
        ):
            with self.subTest(target=target):
                selected = {path for path in self._closure(target) if scoped(path)}
                declared = {path for path in profile if scoped(path)}
                self.assertEqual(
                    sorted(selected - declared),
                    [],
                    f"{target} carries selected sources its profile omits",
                )

    def test_the_full_par_does_not_carry_the_stage_b_backend(self) -> None:
        closure = self._closure("policy_improvement_full")
        self.assertNotIn(
            "scripts/policy_improvement_exp1b_theory_backend.py", closure
        )
        self.assertIn("scripts/policy_improvement_exp1b_runtime.py", closure)
        theory = self._closure("policy_improvement_theory_bridge")
        self.assertIn("scripts/policy_improvement_exp1b_theory_backend.py", theory)

    def test_the_exp1b_unittest_declares_its_direct_imports(self) -> None:
        """Resources are not dependencies; imported modules must be deps."""

        deps = self._deps("test_policy_improvement_exp1b")
        for required in (
            ":policy_improvement_exp1b_auditor",
            ":policy_improvement_exp1b_theory_backend",
            ":phase4_runtime_launcher_lib",
            ":phase4_runtime_profile",
            ":confirmatory_runtime_launcher_lib",
        ):
            self.assertIn(required, deps)

    def test_every_exp1b_profile_path_exists_on_disk(self) -> None:
        """Scoped to the paths this change added: the rest are covered upstream."""

        seen = set()
        for name, profile in (
            ("full", profile_module.POLICY_IMPROVEMENT_FULL_PROFILE_PATHS),
            (
                "theory",
                profile_module.POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS,
            ),
            (
                "exp1b-audit",
                profile_module.POLICY_IMPROVEMENT_EXP1B_AUDIT_PROFILE_PATHS,
            ),
        ):
            for path in profile:
                if "exp1b" not in path:
                    continue
                seen.add(path)
                with self.subTest(profile=name, path=path):
                    self.assertTrue((_ROOT / path).is_file(), path)
        # Every Experiment 1B module and config is declared somewhere.
        self.assertEqual(
            seen,
            {
                "configs/policy_improvement_exp1b/amendments/"
                "reduced_study_exp1b.json",
                "configs/policy_improvement_exp1b/protocol.json",
                "configs/policy_improvement_exp1b/registry.json",
                "scripts/policy_improvement_exp1b_aggregate.py",
                "scripts/policy_improvement_exp1b_auditor.py",
                "scripts/policy_improvement_exp1b_bootstrap.py",
                "scripts/policy_improvement_exp1b_bridge.py",
                "scripts/policy_improvement_exp1b_evidence.py",
                "scripts/policy_improvement_exp1b_runtime.py",
                "scripts/policy_improvement_exp1b_schema.py",
                "scripts/policy_improvement_exp1b_session.py",
                "scripts/policy_improvement_exp1b_theory_backend.py",
            },
        )


class Exp1bStageAEndToEndTest(unittest.TestCase):
    """Finding 1: the runtime is exercised against the production signature.

    The previous suite only asserted the backend methods existed. A neutral call
    through ``main`` therefore never happened, and a required keyword-only
    ``protocol`` argument the runtime did not pass went unnoticed until a review
    ran the real path.
    """

    def _backend(
        self,
        fixture: Any,
        recorder: dict[str, Any],
        *,
        seed_position: int = 0,
        variant: str = "",
    ) -> Any:
        import inspect

        signature = inspect.signature(
            _production_prepare_exp1b_training_signature()
        )

        class _Prepared:
            pass

        class _Backend:
            def prepare_exp1b_training(inner, **kwargs: Any) -> Any:
                # Bind against the real production signature: a runtime that
                # omits an argument fails here exactly as it would in the PAR.
                signature.bind(inner, **kwargs)
                recorder["prepare"] = dict(kwargs)
                prepared = _Prepared()
                prepared.base_policy = _StubAuthenticatedBase()
                prepared.apply_seed = lambda seed: recorder.setdefault(
                    "order", []
                ).append(f"seed:{seed}")
                prepared.load_split = lambda root, split, count: (
                    recorder.setdefault("order", []).append("load"),
                    _loaded_split(fixture, root, split, count),
                )[1]
                prepared.model_factory = lambda dataset: (
                    recorder.setdefault("order", []).append("model"),
                    object(),
                )[1]
                prepared.restore_base_policy = lambda model, base: (
                    recorder.setdefault("order", []).append("restore"),
                    base.model_state_sha256,
                )[1]
                prepared.trainer_factory = lambda model, dataset, config: (
                    recorder.setdefault("order", []).append("trainer"),
                    object(),
                )[1]
                prepared.model_state_digest = (
                    lambda model: _StubAuthenticatedBase.model_state_sha256
                )
                return prepared

            def run_exp1b_training(inner, *, session: Any) -> Any:
                recorder["session"] = session
                return _StubOutcome(seed_position, variant)

        return _Backend()

    def test_main_runs_the_registered_session_and_publishes_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            recorder: dict[str, Any] = {}
            code = exp1b_runtime.main(
                fixture.argv(),
                backend=self._backend(fixture, recorder),
                runtime_attestation=fixture.attestation,
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )
            self.assertEqual(code, 0)
            # Finding 1: the protocol reaches the backend.
            self.assertIn("protocol", recorder["prepare"])
            self.assertEqual(
                exp1b_document_sha256(recorder["prepare"]["protocol"]),
                fixture.protocol_sha,
            )
            # Finding 2: seeded before anything draws.
            self.assertEqual(
                recorder["order"],
                [f"seed:{REGISTERED_SEEDS[0]}", "load", "model", "restore", "trainer"],
            )
            session = recorder["session"]
            self.assertEqual(session.applied_seed, REGISTERED_SEEDS[0])
            self.assertEqual(
                session.call_order,
                (
                    "seed_applied",
                    "train_split_loaded",
                    "model_constructed",
                    "base_policy_restored",
                    "trainer_built",
                ),
            )
            # The run manifest and checkpoint are durably published.
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            directory = generation.checkpoint_directory_for(0)
            self.assertTrue((directory / "checkpoint.pt").is_file())
            manifest = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["applied_seed"], REGISTERED_SEEDS[0])
            self.assertEqual(
                manifest["initialization_artifact_sha256"],
                hashlib.sha256(_BASE_ARTIFACT_BYTES).hexdigest(),
            )
            self.assertEqual(manifest["counters"]["environment_interactions"], 10000)
            self.assertIs(manifest["resolved_evaluation_data"], False)

    def test_the_eighth_run_finalizes_the_generation(self) -> None:
        """Finding 6: Stage A now produces the evidence Stage B requires."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            for offset, row in enumerate(fixture.registry["rows"]):
                self.assertFalse(generation.provenance_path.exists())
                exp1b_runtime.main(
                    fixture.argv(seed_position=offset),
                    backend=self._backend(fixture, {}, seed_position=offset),
                    runtime_attestation=fixture.attestation,
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )
            # Only after the eighth.
            self.assertTrue(generation.provenance_path.exists())
            provenance = json.loads(
                generation.provenance_path.read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["admission_sha256"], fixture.admission_sha)
            self.assertEqual(
                provenance["generation_sha256"], generation.generation_sha256
            )
            self.assertEqual(len(provenance["sealed_checkpoints"]), UNITS)
            self.assertEqual(len(provenance["run_manifest_sha256s"]), UNITS)
            self.assertEqual(
                provenance["base_policy_artifact"]["checkpoint_sha256"],
                hashlib.sha256(_BASE_ARTIFACT_BYTES).hexdigest(),
            )
            self.assertTrue(
                (generation.directory / "base_policy" / "base_policy.pt").is_file()
            )
            # The document the finalizer wrote validates as real provenance.
            validate_substitute_provenance(
                provenance, registry_rows=registry_rows_by_run_id(fixture.registry)
            )
            # And finalizing again is a no-op, not a replacement.
            exp1b_runtime._finalize_when_complete(
                generation=generation,
                study=exp1b_runtime.admit_exp1b_study(
                    protocol_path=root
                    / "configs/policy_improvement_exp1b/protocol.json",
                    registry_path=root
                    / "configs/policy_improvement_exp1b/registry.json",
                    amendment_path=root
                    / "configs/policy_improvement_exp1b/amendments/"
                    "reduced_study_exp1b.json",
                    admission_path=fixture.admission_path,
                    producer_root=root,
                    role=FULL_ROLE,
                    attestation=_attestation(),
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                ),
                base_policy=_StubAuthenticatedBase(),
                admitted_base_artifact=fixture.base_artifact_path,
                attestation=_attestation(),
            )

    def test_main_refuses_without_the_execution_environment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "execution requires"
            ):
                exp1b_runtime.main(
                    fixture.argv(),
                    backend=self._backend(fixture, {}),
                    runtime_attestation=fixture.attestation,
                    environment={},
                )

    def test_main_refuses_a_theory_role_attestation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "different runtime_sha256"
            ):
                exp1b_runtime.main(
                    fixture.argv(),
                    backend=self._backend(fixture, {}),
                    runtime_attestation={
                        **fixture.attestation,
                        "runtime_sha256": _THEORY_RUNTIME,
                    },
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )

    def test_main_refuses_a_base_artifact_the_admission_did_not_admit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            fixture.base_artifact_path.write_bytes(b"a different artifact")
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "admitted checkpoint digest"
            ):
                exp1b_runtime.main(
                    fixture.argv(),
                    backend=self._backend(fixture, {}),
                    runtime_attestation=fixture.attestation,
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )


def _production_prepare_exp1b_training_signature() -> Any:
    return _production_signature(
        "prepare_exp1b_training", inside_class="SealedFullRunBackend"
    )


def _production_signature(name: str, *, inside_class: str | None) -> Any:
    """One production function's real signature, without importing Torch.

    Compiled from source rather than imported: the module imports Torch at
    module scope and Torch is not installed here, but the *signature* is what a
    call site has to satisfy and that is available statically.
    """

    tree = ast.parse(
        (_ROOT / "policy_improvement_full_backend.py").read_text(encoding="utf-8")
    )
    if inside_class is None:
        scope: Any = tree
    else:
        scope = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == inside_class
        )
    function = next(
        node
        for node in scope.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    stub = ast.Module(
        body=[
            ast.FunctionDef(
                name=name,
                args=function.args,
                body=[ast.Pass()],
                decorator_list=[],
                returns=None,
                type_params=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(stub)
    namespace: dict[str, Any] = {}
    exec(compile(stub, "<signature>", "exec"), namespace)  # noqa: S102
    return namespace[name]


class _StubAuthenticatedBase:
    """The shape ``authenticate_base_policy_artifact`` returns."""

    checkpoint_sha256 = hashlib.sha256(_BASE_ARTIFACT_BYTES).hexdigest()
    model_state_sha256 = _BASE_MODEL_STATE_SHA
    initialization_kind = "train_only_pretrained_base_policy"
    architecture_sha256 = _digest("base-architecture")
    producer_git_commit = "c" * 40
    producer_source_manifest_sha256 = _digest("base-manifest")
    training_procedure_sha256 = _digest("base-procedure")
    training_split_ordered_record_sha256 = _digest("base-train-data")


class _StubOutcome:
    environment_interactions = 10000
    counters = {
        "environment_interactions": 10000,
        "train_step_count": 250,
        "value_optimizer_step_count": 250,
        "policy_optimizer_step_count": 125,
        "distill_optimizer_step_count": 0,
        "puzzle_optimizer_step_count": 0,
        "exact_centering_batch_count": 25,
    }

    def __init__(self, seed_position: int = 0, variant: str = "") -> None:
        # `variant` stands in for what a real rerun does: the same registered
        # inputs, different serialized bytes. Empty by default, so every existing
        # fixture keeps its exact digests.
        self.checkpoint_bytes = (
            f"exp1b-stage-a-synthetic-checkpoint-{seed_position}{variant}".encode(
                "ascii"
            )
            * 8
        )
        self.model_state_sha256 = _digest(
            f"stage-a-model-state-{seed_position}{variant}"
        )


def _loaded_split(fixture: Any, root: Any, split: str, count: int) -> Any:
    training = fixture.protocol["training_population"]
    return LoadedTrainSplit(
        dataset={"split": split, "count": count},
        dataset_root=root,
        split=split,
        count=count,
        ordered_record_sha256=training["ordered_record_sha256"],
        dataset_manifest_sha256=training["dataset_manifest_sha256"],
        split_manifest_sha256=training["split_manifest_sha256"],
    )


class _StageAFixture:
    """A producer checkout plus an external signed admission and evidence root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.protocol = _load(_EXP1B / "protocol.json")
        self.registry = _load(_EXP1B / "registry.json")
        self.amendment = _load(_EXP1B / "amendments" / "reduced_study_exp1b.json")
        self.protocol_sha = exp1b_document_sha256(self.protocol)
        for relative, document in (
            ("configs/policy_improvement_exp1b/protocol.json", self.protocol),
            ("configs/policy_improvement_exp1b/registry.json", self.registry),
            (
                "configs/policy_improvement_exp1b/amendments/"
                "reduced_study_exp1b.json",
                self.amendment,
            ),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_json_bytes(document) + b"\n")
        self.owner = root.parent / "owner"
        self.owner.mkdir(parents=True, exist_ok=True)
        self.admission_document = _admission_document(
            protocol_sha256=self.protocol_sha,
            registry_sha256=exp1b_document_sha256(self.registry),
            prior_amendment_sha256=exp1b_document_sha256(self.amendment),
        )
        self.admission_sha = exp1b_document_sha256(self.admission_document)
        self.admission_path = self.owner / "admission.json"
        self.admission_path.write_bytes(
            canonical_json_bytes(self.admission_document) + b"\n"
        )
        self.base_artifact_path = self.owner / "base_policy.pt"
        self.base_artifact_path.write_bytes(_BASE_ARTIFACT_BYTES)
        # The admission names this amendment by digest, so the fixture's
        # document has to be the one that hashes to it.
        self.base_amendment_document = _BASE_AMENDMENT_DOCUMENT
        self.base_amendment_path = self.owner / "base_amendment.json"
        self.base_amendment_path.write_bytes(
            canonical_json_bytes(self.base_amendment_document) + b"\n"
        )
        self.evidence_root = root.parent / "evidence"
        self.generation_id = "gen-0001"
        directory = self.evidence_root / self.generation_id
        (directory / "payloads").mkdir(parents=True, exist_ok=True)
        (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.attestation = {
            "source_git_commit": _COMMIT,
            "runtime_sha256": _RUNTIME,
            "launcher_sha256": _LAUNCHER,
            "runtime_authorization_sha256": _AUTHORIZATION,
        }

    def argv(self, *, seed_position: int = 0) -> list[str]:
        row = self.registry["rows"][seed_position]
        return [
            "--policy-improvement-exp1b-training",
            "--project-root", str(self.root),
            "--reduced-study-protocol",
            str(self.root / "configs/policy_improvement_exp1b/protocol.json"),
            "--reduced-study-registry",
            str(self.root / "configs/policy_improvement_exp1b/registry.json"),
            "--reduced-study-amendment",
            str(
                self.root
                / "configs/policy_improvement_exp1b/amendments/"
                "reduced_study_exp1b.json"
            ),
            "--execution-admission", str(self.admission_path),
            "--base-policy-artifact", str(self.base_artifact_path),
            "--base-policy-amendment", str(self.base_amendment_path),
            "--evidence-root", str(self.evidence_root),
            "--evidence-generation", self.generation_id,
            "--row-id", str(row["run_id"]),
            "--seed", str(row["seed"]),
        ]


def _torch_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _sealed_runtime_identity() -> dict[str, str]:
    """A field-exact sealed identity the production validator accepts.

    ``SealedRuntimeIdentity.from_mapping`` requires the exact nine-field
    inventory, an authorized role, real 64-hex digests, a 40-hex commit, and
    ``runtime_profile_sha256 == selected_source_manifest_sha256 ==
    source_manifest_sha256``. Building it here rather than stubbing the class
    keeps the constructor under test too.
    """

    manifest = _digest("sealed-source-manifest")
    return {
        "role": "policy-improvement-full",
        "runtime_sha256": _digest("sealed-runtime"),
        "source_git_commit": "0" * 39 + "1",
        "source_manifest_sha256": manifest,
        "producer_source_manifest_sha256": _digest("sealed-producer-manifest"),
        "runtime_profile_sha256": manifest,
        "selected_source_manifest_sha256": manifest,
        "runtime_authorization_sha256": _digest("sealed-authorization"),
        "launcher_sha256": _digest("sealed-launcher"),
    }


class _CountingTrainer:
    """Reports a monotone environment-step count without any Torch object.

    ``run_exp1b_training`` only ever asks the trainer for its step count and
    tells it to collect more, so a counter is a faithful stand-in for the loop
    control the method actually implements.
    """

    def __init__(self, *, steps_per_call: int) -> None:
        self._count = 0
        self._steps_per_call = steps_per_call
        self.requested_budgets: list[int] = []

    def get_env_step_count(self) -> int:
        return self._count

    def train_step(self, *, max_env_steps_to_collect: int) -> None:
        self.requested_budgets.append(int(max_env_steps_to_collect))
        self._count += min(self._steps_per_call, int(max_env_steps_to_collect))


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bRealSealedBackendTrainingLoopTest(unittest.TestCase):
    """Execute the *real* ``SealedFullRunBackend.run_exp1b_training`` body.

    This is the test that was missing. Every other Stage A test substitutes its
    own backend class (see ``run_exp1b_training`` definitions elsewhere in this
    file) or asserts only that the production method exists and is not a stub.
    None of them executes the production body, so ``self._module`` -- an
    attribute of ``TorchLearnedRunEngine``, never of ``SealedFullRunBackend`` --
    survived review and reached a real run, where it raised ``AttributeError``
    on the final statement of the method after the entire 10,000-interaction
    budget had already been spent.

    Hermetic: no owner artifact, no dataset, no checkpoint, no Torch tensor. The
    real class is constructed with the real identity validator, and only the
    two collaborators the method calls out to are stand-ins.
    """

    def _backend(self, module: object) -> Any:
        import policy_improvement_full_backend as backend

        return backend.SealedFullRunBackend(
            runtime_identity=_sealed_runtime_identity(),
            training_module=module,
            # A non-None engine keeps TorchLearnedRunEngine out of the picture;
            # run_exp1b_training never touches the engine.
            engine=SimpleNamespace(),
        )

    @contextmanager
    def _captured_seal(self) -> Any:
        """Swap the module-level sealer for a recorder, then put it back.

        Patching the callee does not weaken the test: the ``AttributeError``
        fires while the *argument* is evaluated, before any call happens.
        """

        import policy_improvement_full_backend as backend

        calls: list[dict[str, Any]] = []
        original = backend.seal_exp1b_training_checkpoint

        def recorder(*, session: Any, module: Any) -> Any:
            calls.append({"session": session, "module": module})
            return "sealed-outcome"

        backend.seal_exp1b_training_checkpoint = recorder
        try:
            yield calls
        finally:
            backend.seal_exp1b_training_checkpoint = original

    def _session(self, *, steps_per_call: int, touched: bool = False) -> Any:
        return SimpleNamespace(
            trainer=_CountingTrainer(steps_per_call=steps_per_call),
            dataset_guard=SimpleNamespace(touched_evaluation_data=touched),
        )

    def test_the_real_body_seals_with_this_backends_training_module(self) -> None:
        """The regression: the sealer must receive ``self._training_module``."""

        import policy_improvement_full_backend as backend

        module = SimpleNamespace(name="training-module")
        sealed_backend = self._backend(module)
        session = self._session(steps_per_call=2500)
        with self._captured_seal() as calls:
            outcome = sealed_backend.run_exp1b_training(session=session)
        self.assertEqual(outcome, "sealed-outcome")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["session"], session)
        self.assertIs(
            calls[0]["module"],
            module,
            "run_exp1b_training must seal with the module this backend holds.",
        )
        self.assertEqual(
            session.trainer.get_env_step_count(),
            backend._EXP1B_TERMINAL_ENVIRONMENT_INTERACTIONS,
        )

    def test_the_real_body_stops_exactly_on_the_registered_budget(self) -> None:
        import policy_improvement_full_backend as backend

        target = backend._EXP1B_TERMINAL_ENVIRONMENT_INTERACTIONS
        session = self._session(steps_per_call=3000)
        with self._captured_seal():
            self._backend(SimpleNamespace()).run_exp1b_training(session=session)
        trainer = session.trainer
        self.assertEqual(trainer.get_env_step_count(), target)
        # The remaining budget must shrink on every call, and the last request
        # must ask for exactly what is left rather than a fresh full budget.
        self.assertEqual(
            trainer.requested_budgets,
            [target, target - 3000, target - 6000, target - 9000],
        )

    def test_the_real_body_refuses_a_trainer_that_is_not_fresh(self) -> None:
        import policy_improvement_full_backend as backend

        session = self._session(steps_per_call=2500)
        session.trainer.train_step(max_env_steps_to_collect=2500)
        with self._captured_seal():
            with self.assertRaises(backend.FullBackendError) as caught:
                self._backend(SimpleNamespace()).run_exp1b_training(session=session)
        self.assertIn("fresh trainer", str(caught.exception))

    def test_the_real_body_refuses_a_trainer_that_stalls(self) -> None:
        import policy_improvement_full_backend as backend

        session = self._session(steps_per_call=0)
        with self._captured_seal():
            with self.assertRaises(backend.FullBackendError) as caught:
                self._backend(SimpleNamespace()).run_exp1b_training(session=session)
        self.assertIn("no progress", str(caught.exception))

    def test_the_real_body_refuses_a_run_that_resolved_evaluation_data(self) -> None:
        import policy_improvement_full_backend as backend

        session = self._session(steps_per_call=10000, touched=True)
        with self._captured_seal() as calls:
            with self.assertRaises(backend.FullBackendError) as caught:
                self._backend(SimpleNamespace()).run_exp1b_training(session=session)
        self.assertIn("resolved evaluation data", str(caught.exception))
        self.assertEqual(calls, [], "A tainted run must not be sealed.")


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bTorchRepeatabilityTest(unittest.TestCase):
    """Finding 2: the registered seed actually determines the run.

    Skipped where Torch is absent. Where it is present these are the tests that
    distinguish "the seed is recorded" from "the seed controls the draws".
    """

    def _draws(self, seed: int) -> tuple[float, ...]:
        import random

        import numpy
        import torch

        from utils.seeding import set_global_seed

        set_global_seed(seed)
        return (
            random.random(),
            float(numpy.random.rand()),
            float(torch.rand(1).item()),
            float(torch.nn.Linear(4, 4).weight.detach().reshape(-1)[0].item()),
        )

    def test_two_registered_seeds_produce_different_draws(self) -> None:
        first = self._draws(REGISTERED_SEEDS[0])
        second = self._draws(REGISTERED_SEEDS[1])
        self.assertNotEqual(first, second)

    def test_one_registered_seed_reproduces_exactly(self) -> None:
        first = self._draws(REGISTERED_SEEDS[0])
        # Disturb every generator, then reseed and repeat.
        self._draws(REGISTERED_SEEDS[7])
        self.assertEqual(self._draws(REGISTERED_SEEDS[0]), first)

    def test_the_session_seeds_before_the_model_is_constructed(self) -> None:
        """The ordering claim, measured rather than asserted."""

        import torch

        from utils.seeding import set_global_seed

        protocol = _load(_EXP1B / "protocol.json")
        registry = _load(_EXP1B / "registry.json")
        registration = register_exp1b_training(
            protocol=protocol,
            registry=registry,
            project_root=_ROOT,
            seed=REGISTERED_SEEDS[0],
        )
        observed: dict[str, Any] = {}

        def model_factory(dataset: Any) -> Any:
            observed["weight"] = float(
                torch.nn.Linear(4, 4).weight.detach().reshape(-1)[0].item()
            )
            return object()

        training = protocol["training_population"]
        guard = TrainOnlyDatasetGuard(
            registration=registration,
            loader=lambda root, split, count: LoadedTrainSplit(
                dataset={"split": split},
                dataset_root=root,
                split=split,
                count=count,
                ordered_record_sha256=training["ordered_record_sha256"],
                dataset_manifest_sha256=training["dataset_manifest_sha256"],
                split_manifest_sha256=training["split_manifest_sha256"],
            ),
        )

        class _Base:
            checkpoint_sha256 = _digest("base")
            model_state_sha256 = _digest("state")

        build_exp1b_training_session(
            registration=registration,
            dataset_guard=guard,
            base_policy=_Base(),
            apply_seed=set_global_seed,
            model_factory=model_factory,
            restore_base_policy=lambda model, base: base.model_state_sha256,
            trainer_factory=lambda model, dataset, config: object(),
            model_state_digest=lambda model: _Base.model_state_sha256,
        )
        # The same construction under the same seed, with no session involved.
        set_global_seed(REGISTERED_SEEDS[0])
        expected = float(
            torch.nn.Linear(4, 4).weight.detach().reshape(-1)[0].item()
        )
        self.assertEqual(observed["weight"], expected)


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bTorchSealedCheckpointTest(unittest.TestCase):
    """Findings 3 and 4: seal every module, restore it, and evaluate."""

    def test_the_sealed_checkpoint_carries_all_four_modules(self) -> None:
        import policy_improvement_full_backend as backend

        self.assertEqual(
            backend.EXP1B_CHECKPOINT_MODULES,
            ("model", "policy_model_old", "policy_model_candidate", "target_model"),
        )

    def test_restore_reproduces_every_module_digest(self) -> None:
        self.skipTest(
            "Requires the registered 4x4 dataset and a trained budget-final "
            "checkpoint; see the handoff's unresolved-risk section."
        )


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bTorchFourModuleRoundTripTest(unittest.TestCase):
    """The real ``torch.save`` four-module round trip, with no owner artifacts.

    ``Exp1bTorchSealedCheckpointTest.test_the_sealed_checkpoint_carries_all_four_modules``
    compares a four-name constant and nothing else, and
    ``test_restore_reproduces_every_module_digest`` is an unimplemented
    placeholder. Neither invokes ``seal_exp1b_training_checkpoint``, ``torch.save``,
    or the sanctioned loader, so a defect in the checkpoint payload, the per-module
    state digests, the folded identity, or the restore could sit undetected behind
    a green suite. A previous handoff claimed the round trip ran; it did not.

    Nothing here needs the registered dataset or a trained checkpoint. The
    serializer takes a session and a module, so four real modules and a stand-in
    session reach the production ``torch.save`` directly.

    What is genuinely production here: ``seal_exp1b_training_checkpoint``,
    ``load_data_only_checkpoint``, ``state_dict_sha256``, ``canonical_json_sha256``,
    and ``EXP1B_CHECKPOINT_MODULES``. What is not: the session/trainer scalars and
    counters, and ``_config_to_dict``. Config conversion is not the subject; the
    ``module`` argument is the real ``upi_trm_train`` module, so ``_config_dict``
    is production.

    The four modules share an architecture and differ only in their weights. That
    is deliberate: differing shapes would make a swap fail on ``load_state_dict``
    for the wrong reason, while identical shapes make a swap load cleanly and be
    caught only by the digests and the tensor comparison -- which is the property
    under test.
    """

    _NAMES = ("model", "policy_model_old", "policy_model_candidate", "target_model")

    def _modules(self) -> dict[str, Any]:
        import torch

        from utils.seeding import set_global_seed

        built: dict[str, Any] = {}
        for offset, name in enumerate(self._NAMES):
            # A distinct seed per module, so the four states are different while
            # the architecture is identical.
            set_global_seed(REGISTERED_SEEDS[0] + offset)
            built[name] = torch.nn.Sequential(
                torch.nn.Linear(6, 5), torch.nn.Linear(5, 3)
            )
        digests = {
            name: _module_digest(item) for name, item in built.items()
        }
        self.assertEqual(
            len(set(digests.values())), 4, "the four fixture modules must differ"
        )
        return built

    def _seal(self, modules: dict[str, Any]) -> Any:
        import policy_improvement_full_backend as backend
        import upi_trm_train

        session = _StubSealSession(modules)
        return backend.seal_exp1b_training_checkpoint(
            session=session, module=upi_trm_train
        )

    def _restore(self, payload: Any) -> dict[str, Any]:
        """Load every recorded state dict into a fresh module of the same shape."""

        import torch

        restored: dict[str, Any] = {}
        for name in _sorted_module_names(payload):
            fresh = torch.nn.Sequential(
                torch.nn.Linear(6, 5), torch.nn.Linear(5, 3)
            )
            fresh.load_state_dict(payload[f"{name}_state_dict"])
            restored[name] = fresh
        return restored

    def test_the_production_serializer_round_trips_all_four_modules(self) -> None:
        import io

        from policy_improvement_checkpoint_allowlist import (
            load_data_only_checkpoint,
        )
        import policy_improvement_full_backend as backend

        modules = self._modules()
        outcome = self._seal(modules)
        self.assertTrue(outcome.checkpoint_bytes)

        # Through the sanctioned loader, not a bare torch.load.
        payload = load_data_only_checkpoint(io.BytesIO(outcome.checkpoint_bytes))
        self.assertEqual(
            payload["schema_name"], backend.EXP1B_CHECKPOINT_SCHEMA_NAME
        )
        recorded = dict(payload["module_state_sha256s"])
        self.assertEqual(
            set(recorded), set(backend.EXP1B_CHECKPOINT_MODULES)
        )
        for name in backend.EXP1B_CHECKPOINT_MODULES:
            self.assertIn(f"{name}_state_dict", payload, name)

        restored = self._restore(payload)
        self.assertEqual(set(restored), set(backend.EXP1B_CHECKPOINT_MODULES))

        # 1. Each restored module reproduces its own recorded digest.
        for name, fresh in restored.items():
            with self.subTest(module=name):
                self.assertEqual(_module_digest(fresh), recorded[name])

        # 2. Each restored module is tensor-for-tensor the one it claims to be,
        #    and is *not* any of the other three. This is what a swap breaks.
        for name, fresh in restored.items():
            with self.subTest(module=name):
                self.assertTrue(_same_state(fresh, modules[name]))
                for other in self._NAMES:
                    if other != name:
                        self.assertFalse(_same_state(fresh, modules[other]))

        # 3. The folded identity is over the whole inventory, and it is the one
        #    the outcome and the descriptor carry.
        folded = _folded_digest(recorded)
        self.assertEqual(folded, payload["model_state_sha256"])
        self.assertEqual(folded, outcome.model_state_sha256)
        self.assertEqual(dict(outcome.module_state_sha256s), recorded)

    def test_a_swapped_module_state_is_caught(self) -> None:
        """Two modules exchanged: same shapes, so it loads and must still fail."""

        import io

        from policy_improvement_checkpoint_allowlist import (
            load_data_only_checkpoint,
        )

        modules = self._modules()
        outcome = self._seal(modules)
        payload = dict(
            load_data_only_checkpoint(io.BytesIO(outcome.checkpoint_bytes))
        )
        first, second = "policy_model_old", "policy_model_candidate"
        payload[f"{first}_state_dict"], payload[f"{second}_state_dict"] = (
            payload[f"{second}_state_dict"],
            payload[f"{first}_state_dict"],
        )

        restored = self._restore(payload)
        recorded = dict(payload["module_state_sha256s"])
        mismatched = [
            name
            for name, fresh in restored.items()
            if _module_digest(fresh) != recorded[name]
        ]
        self.assertEqual(sorted(mismatched), sorted([first, second]))
        # And the restored deployed actor is now the candidate's state.
        self.assertTrue(_same_state(restored[first], modules[second]))

    def test_a_dropped_module_state_is_caught(self) -> None:
        import io

        from policy_improvement_checkpoint_allowlist import (
            load_data_only_checkpoint,
        )
        import policy_improvement_full_backend as backend

        modules = self._modules()
        outcome = self._seal(modules)
        payload = dict(
            load_data_only_checkpoint(io.BytesIO(outcome.checkpoint_bytes))
        )
        dropped = "target_model"
        del payload[f"{dropped}_state_dict"]
        del payload["module_state_sha256s"][dropped]

        self.assertNotEqual(
            set(payload["module_state_sha256s"]),
            set(backend.EXP1B_CHECKPOINT_MODULES),
        )
        # The folded identity no longer reproduces the recorded one, which is
        # the check the production restore performs.
        self.assertNotEqual(
            _folded_digest(dict(payload["module_state_sha256s"])),
            payload["model_state_sha256"],
        )
        with self.assertRaises(KeyError):
            payload[f"{dropped}_state_dict"]

    def test_the_folded_digest_changes_when_any_single_module_changes(self) -> None:
        """The fold must cover all four, not just one."""

        modules = self._modules()
        baseline = _folded_digest(
            dict(self._seal(modules).module_state_sha256s)
        )
        import torch

        for name in self._NAMES:
            with self.subTest(module=name):
                perturbed = self._modules()
                with torch.no_grad():
                    next(perturbed[name].parameters()).add_(1.0)
                folded = _folded_digest(
                    dict(self._seal(perturbed).module_state_sha256s)
                )
                self.assertNotEqual(folded, baseline, name)


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bTorchSealedOpenerTest(unittest.TestCase):
    """Drive the real ``open_exp1b_sealed_evaluation_session``, hermetically.

    ``Exp1bTorchFourModuleRoundTripTest`` proves the *serializer* half: it calls
    the production seal, the real ``torch.save``, and the sanctioned loader. It
    then restores through a test-local helper, so the production authenticated
    validator at ``policy_improvement_full_backend.py:6364-6459`` and the
    four-module load/digest/freeze loop at ``:6592-6619`` never ran. A review
    proved that twice: replacing the opener with a counter that raises left all
    four tests green with ``opener_calls=0``, and setting ``schema_version`` to
    999 after serialization also left them green while the opener rejects it.

    This class calls the opener itself. Nothing is factored out of production;
    the dataset, model, environment, and trainer boundaries are injected exactly
    as the review's preferred route describes, so every check between the path
    authentication and the freeze loop is the production one.

    Injected (not production): ``training_module``'s dataset/manifest/checker/
    trainer factories, ``TinyRecursiveReasoningModel_ACTV1``, ``PlanEditEnv``,
    ``UPITrmTrainer`` (so the ``isinstance`` gate accepts the stub trainer), and
    ``Exp1bSealedEvaluationSession`` (replaced by a recorder, because its
    constructor digests a real dataset). Production and actually executed: the
    canonical-path check, ``authenticated_checkpoint_bytes``,
    ``load_data_only_checkpoint``, every payload field check, the census
    recomputation, every registered-RL gate, and the module load/digest/freeze
    loop.
    """

    _SHAPE = (6, 5, 3)

    # --- fixture ----------------------------------------------------------

    def _fresh_module(self) -> Any:
        import torch

        first, hidden, last = self._SHAPE
        return torch.nn.Sequential(
            torch.nn.Linear(first, hidden), torch.nn.Linear(hidden, last)
        )

    def _sealed_modules(self) -> dict[str, Any]:
        from utils.seeding import set_global_seed

        built: dict[str, Any] = {}
        for offset, name in enumerate(_EXP1B_MODULE_NAMES):
            set_global_seed(REGISTERED_SEEDS[0] + 40 + offset)
            built[name] = self._fresh_module()
        return built

    def _census(self) -> tuple[Any, tuple[Any, ...]]:
        from policy_improvement_full_backend import (
            canonical_json_sha256,
            Exp1bCensusMember,
        )

        members = tuple(
            Exp1bCensusMember(
                state_id=f"exp1b-state-{index:064d}",
                record_index=index,
                dataset_record_sha256=_digest(f"opener-record-{index}"),
            )
            for index in range(3)
        )
        ordering = canonical_json_sha256(
            [
                {
                    "state_id": member.state_id,
                    "record_index": member.record_index,
                    "dataset_record_sha256": member.dataset_record_sha256,
                }
                for member in members
            ]
        )
        return ordering, members

    def _drive(
        self,
        tmp: str,
        *,
        mutate: Any = None,
        arguments: Mapping[str, Any] | None = None,
        payload_object: Any = None,
    ) -> dict[str, Any]:
        """Seal, write the bytes out, and call the real opener.

        Returns the kwargs the opener passed to ``Exp1bSealedEvaluationSession``
        plus the modules the production freeze loop restored into.
        """

        import hashlib
        import io

        import policy_improvement_full_backend as backend
        from policy_improvement_checkpoint_allowlist import (
            load_data_only_checkpoint,
        )
        import torch
        import upi_trm_train

        sealed = self._sealed_modules()
        outcome = backend.seal_exp1b_training_checkpoint(
            session=_StubSealSession(sealed), module=upi_trm_train
        )
        payload_bytes = outcome.checkpoint_bytes
        if payload_object is not None:
            buffer = io.BytesIO()
            torch.save(payload_object, buffer)
            payload_bytes = buffer.getvalue()
        elif mutate is not None:
            payload = dict(load_data_only_checkpoint(io.BytesIO(payload_bytes)))
            mutate(payload)
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            payload_bytes = buffer.getvalue()

        path = Path(tmp) / "sealed_checkpoint.pt"
        path.write_bytes(payload_bytes)
        ordering, members = self._census()

        # Fresh targets: the freeze loop must actually change them.
        trainer_modules = {
            name: self._fresh_module() for name in _EXP1B_MODULE_NAMES
        }
        for name in _EXP1B_MODULE_NAMES:
            self.assertNotEqual(
                _module_digest(trainer_modules[name]), _module_digest(sealed[name])
            )

        call = {
            "checkpoint_path": path,
            "expected_checkpoint_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "expected_checkpoint_size_bytes": len(payload_bytes),
            "expected_model_state_sha256": outcome.model_state_sha256,
            "expected_effective_config_sha256": _StubSealSession(
                sealed
            ).effective_config_sha256,
            "expected_environment_interactions": TERMINAL_ENVIRONMENT_INTERACTIONS,
            "run_id": "exp1b-seal-round-trip",
            "seed": REGISTERED_SEEDS[0],
            "seed_position": 0,
            "census_ordering_sha256": ordering,
            "census": members,
            "depths": (DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M),
            "evaluation_population": "validation_bridge",
            "dataset_root": Path(tmp),
            "evaluation_split": "validation",
            "evaluation_split_manifest_sha256": _digest("opener-split-manifest"),
        }
        call.update(dict(arguments or {}))

        recorded: dict[str, Any] = {}
        with _injected_opener_boundaries(trainer_modules, recorded):
            backend.open_exp1b_sealed_evaluation_session(
                training_module=_StubOpenerTrainingModule(trainer_modules),
                **call,
            )
        recorded["restored_modules"] = trainer_modules
        recorded["sealed_modules"] = sealed
        return recorded

    # --- the positive path -------------------------------------------------

    def test_the_production_opener_authenticates_and_restores_all_four(self) -> None:
        with TemporaryDirectory() as tmp:
            recorded = self._drive(tmp)

        self.assertTrue(recorded.get("reached"), "the opener did not complete")
        restored = recorded["restored_modules"]
        sealed = recorded["sealed_modules"]

        # The production loop loaded each sealed state into its own target...
        for name in _EXP1B_MODULE_NAMES:
            with self.subTest(module=name):
                self.assertTrue(_same_state(restored[name], sealed[name]))
                for other in _EXP1B_MODULE_NAMES:
                    if other != name:
                        self.assertFalse(_same_state(restored[name], sealed[other]))

        # ...and froze it. This is the `:6613-6619` half of the loop.
        for name, module in restored.items():
            with self.subTest(module=name):
                self.assertFalse(module.training, f"{name} was left in train mode")
                for parameter in module.parameters():
                    self.assertFalse(parameter.requires_grad)

        # The identities the opener carried into the session.
        self.assertEqual(
            set(recorded["module_state_sha256s"]), set(_EXP1B_MODULE_NAMES)
        )
        self.assertEqual(
            recorded["model_state_sha256"],
            _folded_digest(recorded["module_state_sha256s"]),
        )
        self.assertEqual(recorded["run_id"], "exp1b-seal-round-trip")
        self.assertEqual(recorded["seed"], REGISTERED_SEEDS[0])
        self.assertEqual(
            recorded["environment_interactions"], TERMINAL_ENVIRONMENT_INTERACTIONS
        )

    # --- every pre-restore invariant, one mutation each ---------------------

    def test_each_payload_invariant_is_enforced_by_the_production_opener(self) -> None:
        """One mutation per invariant; each must raise from production code."""

        def field(name: str, value: Any) -> Any:
            def apply(payload: dict[str, Any]) -> None:
                payload[name] = value

            return apply

        def rl(name: str, value: Any) -> Any:
            def apply(payload: dict[str, Any]) -> None:
                settings = dict(payload["rl_config"])
                settings[name] = value
                payload["rl_config"] = settings
                payload["rl_config_sha256"] = _canonical_sha256(settings)

            return apply

        def swap_two_states(payload: dict[str, Any]) -> None:
            first, second = "policy_model_old", "target_model"
            payload[f"{first}_state_dict"], payload[f"{second}_state_dict"] = (
                payload[f"{second}_state_dict"],
                payload[f"{first}_state_dict"],
            )

        def drop_state(payload: dict[str, Any]) -> None:
            del payload["target_model_state_dict"]

        def drop_recorded_digest(payload: dict[str, Any]) -> None:
            recorded = dict(payload["module_state_sha256s"])
            del recorded["target_model"]
            payload["module_state_sha256s"] = recorded

        cases: tuple[tuple[str, Any, Mapping[str, Any]], ...] = (
            # Path and byte binding, before deserialization.
            ("non-canonical path", None, {"checkpoint_path": Path("relative.pt")}),
            (
                "wrong expected digest",
                None,
                {"expected_checkpoint_sha256": _digest("another-checkpoint")},
            ),
            ("wrong expected size", None, {"expected_checkpoint_size_bytes": 11}),
            # Schema and strict typing.
            ("schema name", field("schema_name", "other_schema"), {}),
            ("schema version", field("schema_version", 999), {}),
            # Run identity.
            ("run identity", None, {"run_id": "exp1b-some-other-run"}),
            ("seed", None, {"seed": REGISTERED_SEEDS[1]}),
            ("seed position", None, {"seed_position": 5}),
            ("applied seed", field("applied_seed", REGISTERED_SEEDS[1]), {}),
            # Budget, configuration, access.
            ("budget-final interactions", field("environment_interactions", 9999), {}),
            (
                "effective config",
                None,
                {"expected_effective_config_sha256": _digest("another-config")},
            ),
            ("evaluation-data access", field("resolved_evaluation_data", True), {}),
            # Module identity.
            (
                "external model-state identity",
                None,
                {"expected_model_state_sha256": _digest("another-model-state")},
            ),
            # The other half of the same production check. Production compares
            # the recomputed fold against both the payload's own
            # `model_state_sha256` and the external descriptor
            # (`policy_improvement_full_backend.py:6441-6444`); the case above
            # only moves the external one. Deleting just the payload-half
            # comparison would otherwise go uncaught.
            (
                "payload self-consistent model-state identity",
                field("model_state_sha256", _digest("another-folded-identity")),
                {},
            ),
            ("module inventory", drop_recorded_digest, {}),
            (
                "model config digest",
                field("model_config_sha256", _digest("another-model-config")),
                {},
            ),
            (
                "rl config digest",
                field("rl_config_sha256", _digest("another-rl-config")),
                {},
            ),
            # Registered RL settings.
            ("registered discount", rl("gamma", 0.5), {}),
            ("registered mixture alpha", rl("mixture_alpha", 0.25), {}),
            ("registered unroll depth", rl("inner_unroll_n", 7), {}),
            ("exact mixture deployed", rl("theory_exact_mixture", False), {}),
            ("persistent latent mode", rl("episodic_latent", True), {}),
            ("no epsilon-greedy", rl("policy_epsilon", 0.1), {}),
            # Census binding.
            (
                "census ordering",
                None,
                {"census_ordering_sha256": _digest("another-ordering")},
            ),
            ("empty census", None, {"census": ()}),
            # The restore loop itself.
            ("swapped module state", swap_two_states, {}),
            ("dropped module state", drop_state, {}),
        )

        from policy_improvement_full_backend import Exp1bSealedEvaluationError

        for label, mutate, arguments in cases:
            with self.subTest(invariant=label), TemporaryDirectory() as tmp:
                with self.assertRaises(Exp1bSealedEvaluationError):
                    self._drive(tmp, mutate=mutate, arguments=arguments)

    def test_a_non_mapping_payload_is_refused(self) -> None:
        """The strict-mapping gate, which needs a whole replacement payload."""

        from policy_improvement_full_backend import Exp1bSealedEvaluationError

        with TemporaryDirectory() as tmp:
            with self.assertRaises(Exp1bSealedEvaluationError):
                self._drive(tmp, payload_object=["not", "a", "mapping"])

    def test_the_opener_is_reached_at_all(self) -> None:
        """Guards against this class silently stopping short of the opener.

        The finding this class exists to fix was a test that never called the
        production opener while claiming to. If the call site is ever removed or
        short-circuited, this fails.
        """

        import policy_improvement_full_backend as backend

        calls: list[int] = []
        original = backend.open_exp1b_sealed_evaluation_session

        def _counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(*args, **kwargs)

        backend.open_exp1b_sealed_evaluation_session = _counting
        try:
            with TemporaryDirectory() as tmp:
                self._drive(tmp)
        finally:
            backend.open_exp1b_sealed_evaluation_session = original
        self.assertEqual(sum(calls), 1)


_EXP1B_MODULE_NAMES = (
    "model",
    "policy_model_old",
    "policy_model_candidate",
    "target_model",
)


def _canonical_sha256(value: Any) -> str:
    from policy_improvement_full_backend import canonical_json_sha256

    return canonical_json_sha256(value)


class _StubOpenerTrainingModule:
    """The ``training_module`` seam the opener already takes as an argument."""

    def __init__(self, trainer_modules: Mapping[str, Any]) -> None:
        self._trainer_modules = trainer_modules
        self.RLConfig = _StubOpenerRlConfig

    def build_dataset_from_paths(
        self, *, dataset_paths: Any, pool_size: Any, split: str
    ) -> tuple[Any, int, int, Any]:
        return (object(), 4, 5, ())

    def _validate_materialized_split_manifest(
        self, *, dataset_root: Any, split: str, registered_sha256: str, dataset: Any
    ) -> None:
        return None

    def resolve_checker_from_dataset(
        self, *, rl_cfg: Any, dataset: Any, seq_len: int
    ) -> Any:
        return SimpleNamespace(
            checker_fn=lambda *a, **k: 0.0, checker_kind="solution"
        )

    def select_baseline_from_configs(self, *args: Any, **kwargs: Any) -> Any:
        return object()

    def build_trainer(self, **kwargs: Any) -> Any:
        return _StubOpenerTrainer(self._trainer_modules)


class _StubOpenerRlConfig:
    """Whatever the sealed payload recorded, as attributes the gates read."""

    def __init__(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self, key, value)


class _StubOpenerTrainer:
    """Holds the three trainer-side modules the restore loop writes into."""

    def __init__(self, modules: Mapping[str, Any]) -> None:
        self.policy_model_old = modules["policy_model_old"]
        self.policy_model_candidate = modules["policy_model_candidate"]
        self.target_model = modules["target_model"]

    def set_checker_fn(self, checker: Any) -> None:
        self._checker = checker


@contextmanager
def _injected_opener_boundaries(
    trainer_modules: Mapping[str, Any], recorded: dict[str, Any]
) -> Any:
    """Replace only the dataset/model/environment/session boundaries.

    Everything between the canonical-path check and the freeze loop stays
    production. ``Exp1bSealedEvaluationSession`` is replaced by a recorder
    because its constructor digests a real evaluation dataset, which is the one
    genuinely owner-artifact-bound step in this path.
    """

    import policy_improvement_full_backend as backend

    class _StubEnvironment:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

        def set_stop_action_id(self, value: int) -> None:
            self._stop = value

    def _record_session(**kwargs: Any) -> Any:
        recorded.update(kwargs)
        recorded["reached"] = True
        return SimpleNamespace(**kwargs)

    saved = {
        name: getattr(backend, name)
        for name in (
            "TinyRecursiveReasoningModel_ACTV1",
            "PlanEditEnv",
            "UPITrmTrainer",
            "Exp1bSealedEvaluationSession",
        )
    }
    backend.TinyRecursiveReasoningModel_ACTV1 = (
        lambda config: trainer_modules["model"]
    )
    backend.PlanEditEnv = _StubEnvironment
    backend.UPITrmTrainer = _StubOpenerTrainer
    backend.Exp1bSealedEvaluationSession = _record_session
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(backend, name, value)



def _module_digest(module: Any) -> str:
    from rl.persistent_diagnostic_checkpoint import state_dict_sha256

    return state_dict_sha256(module.state_dict())


def _folded_digest(recorded: Mapping[str, Any]) -> str:
    from policy_improvement_full_backend import canonical_json_sha256

    return canonical_json_sha256({name: str(recorded[name]) for name in recorded})


def _sorted_module_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    from policy_improvement_full_backend import EXP1B_CHECKPOINT_MODULES

    return tuple(EXP1B_CHECKPOINT_MODULES)


def _same_state(left: Any, right: Any) -> bool:
    import torch

    first, second = left.state_dict(), right.state_dict()
    if set(first) != set(second):
        return False
    return all(torch.equal(first[key], second[key]) for key in first)


class _StubSealRlConfig:
    """A config the production ``_config_dict`` accepts.

    Carries the registered settings, because
    ``open_exp1b_sealed_evaluation_session`` reads them back out of the sealed
    payload and gates on them.
    """

    def dict(self) -> dict[str, Any]:
        return dict(_REGISTERED_RL_SETTINGS)


#: What a conforming Experiment 1B run trained under, as the opener's gates at
#: ``policy_improvement_full_backend.py:6500-6539`` require it.
_REGISTERED_RL_SETTINGS: dict[str, Any] = {
    "gamma": GAMMA,
    "mixture_alpha": MIXTURE_ALPHA,
    "inner_unroll_n": DEPLOYED_DEPTH_N,
    "theory_exact_mixture": True,
    "episodic_latent": False,
    "policy_epsilon": 0.0,
    "max_edits": 8,
    "reward_shaping": True,
    "solved_threshold": 1.0,
    "task_name": "sudoku",
    "stop_action_mode": "terminal",
    "stop_action_penalty": 0.0,
    "fail_terminal_reward": -1.0,
    "solve_terminal_reward": 1.0,
    "C_max": 16.0,
    "disable_constraint_masking": False,
}


class _StubSealTrainer:
    """The trainer surface ``seal_exp1b_training_checkpoint`` reads, and no more."""

    def __init__(self, modules: Mapping[str, Any]) -> None:
        self.policy_model_old = modules["policy_model_old"]
        self.policy_model_candidate = modules["policy_model_candidate"]
        self.target_model = modules["target_model"]
        self.rl_cfg = _StubSealRlConfig()
        self._train_step_count = 250
        self._value_optimizer_step_count = 250
        self._policy_optimizer_step_count = 125
        self._distill_optimizer_step_count = 0
        self._puzzle_optimizer_step_count = 0
        self._exact_centering_batch_count = 25

    def get_env_step_count(self) -> int:
        return TERMINAL_ENVIRONMENT_INTERACTIONS

    def _config_to_dict(self, config: Any) -> dict[str, Any]:
        return config.dict()


class _StubSealSession:
    """The session surface the serializer reads. No dataset, no owner artifact."""

    def __init__(self, modules: Mapping[str, Any]) -> None:
        self.model = modules["model"]
        self.model.config = _StubSealRlConfig()
        self.trainer = _StubSealTrainer(modules)
        self.run_id = "exp1b-seal-round-trip"
        self.seed = REGISTERED_SEEDS[0]
        self.seed_position = 0
        self.applied_seed = REGISTERED_SEEDS[0]
        self.method_id = "fixed_base_exact_persistent"
        self.effective_config_sha256 = _digest("seal-effective-config")
        self.train_ordered_record_sha256 = _digest("seal-train-order")
        self.train_record_count = TRAINING_RECORD_COUNT
        self.dataset_name = "sudoku-4x4"
        self.training_population_id = "train_minus_stage0_smoke"
        self.initialization_kind = "train_only_pretrained_base_policy"
        self.initialization_artifact_sha256 = _digest("seal-base-artifact")
        self.restored_model_state_sha256 = _digest("seal-base-model-state")


class Exp1bCheckpointReplacementTest(unittest.TestCase):
    """Finding 4: the bytes that are hashed must be the bytes that are loaded."""

    def _write(self, directory: Path, payload: bytes) -> tuple[Path, str, int]:
        path = directory.resolve() / "checkpoint.pt"
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest(), len(payload)

    def test_the_authenticated_buffer_is_returned_for_in_memory_load(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = b"exp1b-checkpoint-version-A" * 64
            path, digest, size = self._write(Path(tmp), payload)
            self.assertEqual(
                authenticated_checkpoint_bytes(
                    path,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    label="checkpoint",
                ),
                payload,
            )

    def test_replacement_between_authentication_and_load_is_refused(self) -> None:
        """Review reproducer: hash version A, replace, load version B.

        Deterministic rather than racy: the helper is exercised directly, and the
        replacement happens between two calls rather than inside one. The
        production opener calls exactly this helper and deserializes only its
        return value, so a passing helper is a passing opener.
        """

        with TemporaryDirectory() as tmp:
            original = b"exp1b-checkpoint-version-A" * 64
            path, digest, size = self._write(Path(tmp), original)
            first = authenticated_checkpoint_bytes(
                path,
                expected_sha256=digest,
                expected_size_bytes=size,
                label="checkpoint",
            )
            self.assertEqual(first, original)

            # Same length, different content: only the digest can catch it.
            replacement = b"exp1b-checkpoint-version-B" * 64
            self.assertEqual(len(replacement), len(original))
            path.write_bytes(replacement)
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "differ from their registered descriptor"
            ):
                authenticated_checkpoint_bytes(
                    path,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    label="checkpoint",
                )

    def test_the_opener_does_not_coerce_the_registered_size(self) -> None:
        """An `int()` cast at the call site would launder a float descriptor."""

        source = (_ROOT / "policy_improvement_full_backend.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "open_exp1b_sealed_evaluation_session"
        )
        call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "authenticated_checkpoint_bytes"
        )
        size = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "expected_size_bytes"
        )
        self.assertIsInstance(size, ast.Name)
        self.assertEqual(size.id, "expected_checkpoint_size_bytes")
        # And the helper itself refuses the non-integer forms.
        with TemporaryDirectory() as tmp:
            path, digest, size_bytes = self._write(Path(tmp), b"A" * 64)
            for bad in (float(size_bytes), True, "64", None):
                with self.subTest(size=bad):
                    with self.assertRaises(Exp1bEvidenceError):
                        authenticated_checkpoint_bytes(
                            path,
                            expected_sha256=digest,
                            expected_size_bytes=bad,  # type: ignore[arg-type]
                            label="checkpoint",
                        )

    def test_the_sealed_session_pins_the_registered_study_settings(self) -> None:
        """Gaps the adversarial verification found: gamma, alpha, depth, mixture.

        `action_values` discounts successors with the *sealed* gamma while the
        route builds every observation with the *registered* one. A checkpoint
        trained at another discount would produce a self-consistent but wrong
        residual, so the opener has to refuse it rather than evaluate it.
        """

        source = (_ROOT / "policy_improvement_full_backend.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "open_exp1b_sealed_evaluation_session"
        )
        body = ast.get_source_segment(source, function)
        assert body is not None
        for guard in (
            "float(rl_config.gamma) != _REGISTERED_GAMMA",
            "float(rl_config.mixture_alpha) != _REGISTERED_ALPHA",
            "int(rl_config.inner_unroll_n) != _REGISTERED_DEPLOYED_DEPTH",
            'getattr(rl_config, "theory_exact_mixture", False)',
            "bool(rl_config.episodic_latent)",
            'float(getattr(rl_config, "policy_epsilon", 0.0)) != 0.0',
        ):
            self.assertIn(guard, body)

    def test_a_size_change_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            path, digest, size = self._write(Path(tmp), b"A" * 512)
            path.write_bytes(b"A" * 256)
            with self.assertRaisesRegex(Exp1bEvidenceError, "size differs"):
                authenticated_checkpoint_bytes(
                    path,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                    label="checkpoint",
                )

    def test_a_symlinked_checkpoint_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "real.pt"
            real.write_bytes(b"A" * 128)
            link = root / "link.pt"
            link.symlink_to(real)
            with self.assertRaises(Exp1bEvidenceError):
                authenticated_checkpoint_bytes(
                    link,
                    expected_sha256=hashlib.sha256(b"A" * 128).hexdigest(),
                    expected_size_bytes=128,
                    label="checkpoint",
                )

    def test_the_opener_deserializes_from_the_authenticated_buffer(self) -> None:
        """The opener must not reopen the path. Pinned structurally.

        The whole finding was a second read: hash the path, then reopen it for
        the loader. This asserts the load argument is an in-memory buffer built
        from the authenticated bytes, and that no path-opening call survives
        between authentication and deserialization.
        """

        tree = ast.parse(
            (_ROOT / "policy_improvement_full_backend.py").read_text(encoding="utf-8")
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "open_exp1b_sealed_evaluation_session"
        )
        loads = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_data_only_checkpoint"
        ]
        self.assertEqual(len(loads), 1)
        argument = loads[0].args[0]
        self.assertIsInstance(argument, ast.Call)
        self.assertEqual(argument.func.attr, "BytesIO")
        self.assertEqual(argument.args[0].id, "checkpoint_payload_bytes")
        # And the bytes came from the single-open authenticator.
        authenticators = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "authenticated_checkpoint_bytes"
        ]
        self.assertEqual(len(authenticators), 1)
        # No path reopen survives anywhere in the function.
        source = ast.get_source_segment(
            (_ROOT / "policy_improvement_full_backend.py").read_text(
                encoding="utf-8"
            ),
            function,
        )
        self.assertNotIn('path.open("rb")', source)
        self.assertNotIn("file_sha256(path)", source)


class Exp1bBaseOperatorPolicyTest(unittest.TestCase):
    """Finding 2: the Bellman operator takes the frozen base, not the mixture."""

    def test_the_evaluator_reads_the_frozen_base_law(self) -> None:
        """Structural: `base_probabilities` must come off `policy_model_old`.

        Torch-free, and it is the check that would have caught the defect: the
        old code assigned `trainer._mixed_policy_dist`'s output straight into
        `record.base_probabilities`.
        """

        source = (_ROOT / "policy_improvement_full_backend.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        session = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "Exp1bSealedEvaluationSession"
        )
        populate = next(
            node
            for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "_populate"
        )
        body = ast.get_source_segment(source, populate)
        assert body is not None
        # The base law is read from the frozen actor.
        self.assertIn("base_dist, base_latent = base_model.policy_dist(", body)
        self.assertIn("base_model = trainer.policy_model_old", body)
        # The deployed mixture is computed, and kept separate.
        self.assertIn("deployed_dist, next_latent = deployed_callback(", body)
        self.assertIn('record.base_probabilities = laws["base"]', body)
        self.assertIn('record.deployed_probabilities = laws["deployed"]', body)
        # The mixture must never land in the operator's slot again.
        self.assertNotIn("record.base_probabilities = laws[\"deployed\"]", body)
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("record.base_probabilities ="):
                self.assertEqual(stripped, 'record.base_probabilities = laws["base"]')

    def test_the_candidate_is_evaluated_at_the_base_output_latent(self) -> None:
        """Persistent mode: mirroring `_probability_mixture_dist` exactly.

        The trainer evaluates the candidate with `n=0` at the base call's output
        latent so both laws describe the *same* state. Evaluating it anywhere
        else silently compares two different states, and the mixture-identity
        diagnostic would then be measuring this module's mistake.
        """

        source = (_ROOT / "policy_improvement_full_backend.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        session = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "Exp1bSealedEvaluationSession"
        )
        populate = next(
            node
            for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "_populate"
        )
        body = ast.get_source_segment(source, populate)
        assert body is not None
        self.assertIn(
            "batched_x, plan, n=0, action_mask=action_mask, z=base_latent", body
        )
        trainer_source = (_ROOT / "rl" / "upi_trm_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "x_batch, y_batch, n=0, action_mask=action_mask, z=z_old", trainer_source
        )

    def test_the_independent_operator_oracle(self) -> None:
        """The arithmetic the review used, recomputed here as a standing oracle.

        With old = (0.8, 0.2), candidate = (0.2, 0.8), alpha = 0.1 and action
        values (0, 10): the base operator is 2.0 and the deployed mixture's is
        2.6. At gamma = 0.99 the 0.6 difference scales by 1/(1-gamma) = 100 into
        a 60.0 change in the direct residual bound -- which is why this is a
        correctness finding and not a rounding one.
        """

        old = (0.8, 0.2)
        candidate = (0.2, 0.8)
        alpha = 0.1
        action_values = (0.0, 10.0)
        deployed = tuple(
            (1.0 - alpha) * left + alpha * right
            for left, right in zip(old, candidate)
        )
        base_operator = math.fsum(p * q for p, q in zip(old, action_values))
        deployed_operator = math.fsum(
            p * q for p, q in zip(deployed, action_values)
        )
        self.assertEqual(base_operator, 2.0)
        self.assertEqual(deployed_operator, 2.6)
        endpoint = 0.0
        denominator = 1.0 - GAMMA
        base_bound = abs(endpoint - base_operator) / denominator
        deployed_bound = abs(endpoint - deployed_operator) / denominator
        self.assertAlmostEqual(deployed_bound - base_bound, 60.0, places=6)
        # And the mixture identity this evaluator asserts holds for that law.
        self.assertAlmostEqual(
            0.5 * math.fsum(abs(l - r) for l, r in zip(deployed, deployed)), 0.0
        )


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bTorchBaseOperatorTest(unittest.TestCase):
    """Finding 2, executed: old and candidate must differ and both be reported."""

    def test_base_and_deployed_differ_when_the_candidate_has_moved(self) -> None:
        self.skipTest(
            "Requires a restored four-module checkpoint; see the handoff's "
            "unresolved-Torch section for the exact fbsource command."
        )


class Exp1bSecondaryDiagnosticsTest(unittest.TestCase):
    """Finding 3: five separate records, validated before a seed is sealable."""

    def _serve_first(self, route: Any, fixture: Any, **overrides: Any) -> Any:
        item = fixture.sealed[0]
        evaluator = _evaluator(0)
        if overrides:
            evaluator["secondary_diagnostics"] = _secondary_diagnostics(**overrides)
        return route.serve(
            seed_position=0,
            seed=item["seed"],
            run_id=item["run_id"],
            checkpoint_sha256=item["checkpoint_sha256"],
            evaluation_population="validation_bridge",
            **evaluator,
        )

    def test_all_five_records_reach_the_published_result(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            result = aggregate_exp1b_result(route.sealed_octet())
            for row in result.seed_rows:
                for name in SECONDARY_DIAGNOSTIC_NAMES:
                    self.assertIn(name, row.secondary_diagnostics)
            document = result.as_document()
            for row in document["seed_rows"]:
                self.assertEqual(
                    set(row["secondary_diagnostics"])
                    - {"schema_name", "schema_version"},
                    set(SECONDARY_DIAGNOSTIC_NAMES),
                )
            # And they survive the independent audit.
            payload = validated_exp1b_document(
                result,
                route=route,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                admission=fixture.admission_document,
            )
            report = audit_exp1b_result_document(
                payload,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                population_registry=_parent_populations(),
            )
            self.assertIn("secondary_diagnostics", report.checks)

    def test_they_are_not_folded_into_the_signed_gap(self) -> None:
        """Two seeds with identical primaries and different secondaries agree."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            first = self._serve_first(route, fixture)
            baseline = first.result_values().signed_gap
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            moved = self._serve_first(
                route,
                fixture,
                target_lag={"maximum_absolute_target_lag": 9.5},
                deployment_mismatch={
                    "maximum_candidate_base_total_variation": 0.9
                },
            )
            self.assertEqual(moved.result_values().signed_gap, baseline)

    def test_a_failed_mixture_identity_leaves_the_seed_unsealed(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            with self.assertRaisesRegex(Exp1bBridgeError, "mixture identity"):
                self._serve_first(
                    route,
                    fixture,
                    mixture_identity={
                        "maximum_identity_total_variation": 2.0e-6
                    },
                )
            self.assertEqual(route.served_positions, ())
            self.assertFalse(fixture.generation().has_payload(0))

    def test_a_failed_centering_parity_leaves_the_seed_unsealed(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            with self.assertRaisesRegex(Exp1bBridgeError, "centering parity"):
                self._serve_first(
                    route,
                    fixture,
                    centering_parity={"constructed_centering_roundoff": 5.0e-6},
                )
            self.assertEqual(route.served_positions, ())
            self.assertFalse(fixture.generation().has_payload(0))

    def test_an_incomplete_persistent_carry_leaves_the_seed_unsealed(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            with self.assertRaisesRegex(Exp1bBridgeError, "did not carry"):
                self._serve_first(
                    route,
                    fixture,
                    persistent_state={"states_with_carried_latent": 127},
                )
            self.assertFalse(fixture.generation().has_payload(0))

    def test_the_contract_tripwires_refuse_a_seed(self) -> None:
        """The records assert the finding-2 contract, and that is machine-checked."""

        cases = (
            ("mixture_identity", {"operator_used_base_probabilities": False}),
            ("mixture_identity", {"deployed_reconstructed_from_mixture": True}),
            ("target_lag", {"folded_into_signed_gap": True}),
            ("centering_parity", {"folded_into_signed_gap": True}),
            ("deployment_mismatch", {"folded_into_signed_gap": True}),
            ("mixture_identity", {"policy_epsilon": 0.05}),
            ("persistent_state", {"latent_mode": "episodic"}),
        )
        for block, override in cases:
            with self.subTest(block=block, override=override):
                with TemporaryDirectory() as tmp:
                    fixture = _RouteFixture(Path(tmp))
                    route = fixture.open()
                    with self.assertRaises(Exp1bBridgeError):
                        self._serve_first(route, fixture, **{block: override})
                    self.assertFalse(fixture.generation().has_payload(0))

    def test_a_missing_block_is_refused(self) -> None:
        for name in SECONDARY_DIAGNOSTIC_NAMES:
            with self.subTest(missing=name):
                document = _secondary_diagnostics()
                document.pop(name)
                with self.assertRaises(Exp1bSchemaError):
                    validate_secondary_diagnostics(document, path="probe")

    def test_integer_typed_reals_are_refused(self) -> None:
        """Canonical JSON distinguishes 0 from 0.0; the validators must too."""

        for block, field in (
            ("target_lag", "maximum_absolute_target_lag"),
            ("centering_parity", "constructed_centering_roundoff"),
            ("mixture_identity", "maximum_identity_total_variation"),
            ("deployment_mismatch", "maximum_candidate_base_total_variation"),
        ):
            with self.subTest(block=block, field=field):
                document = _secondary_diagnostics(**{block: {field: True}})
                with self.assertRaises(Exp1bSchemaError):
                    validate_secondary_diagnostics(document, path="probe")

    def test_the_payload_digest_covers_the_diagnostics(self) -> None:
        """Adding the records must change the persisted identity in lockstep.

        Three functions build the same canonical document -- `_payload_document`,
        the in-memory `payload_sha256`, and `_persisted_payload_sha256`. If any
        one of them omits the new block the reload digest disagrees and
        aggregation refuses, so this pins all three at once.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            live = self._serve_first(route, fixture)
            persisted, digest = fixture.generation().read_payload(0)
            self.assertIn("secondary_diagnostics", persisted)
            self.assertEqual(live.payload_sha256(), digest)
            self.assertEqual(
                persisted["secondary_diagnostics"]["mixture_identity"]["kind"],
                MIXTURE_IDENTITY_KIND,
            )

    def test_a_reloaded_payload_revalidates_its_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            octet = route.sealed_octet()
            for payload in octet.payloads:
                self.assertIsNotNone(payload.secondary_diagnostics)
                validate_secondary_diagnostics(
                    dict(payload.secondary_diagnostics), path="reloaded"
                )


class Exp1bWrongRouteOrphanTest(unittest.TestCase):
    """Finding 7: a crash orphan from another route must never be adopted."""

    def _orphan(self, fixture: Any, route_id: str) -> None:
        """Write a complete, otherwise-valid seed-0 payload under ``route_id``."""

        generation = fixture.generation()
        item = fixture.sealed[0]
        document = {
            "schema_name": "policy_improvement_exp1b_seed_payload_v1",
            "schema_version": 2,
            "route_id": route_id,
            "seed_position": 0,
            "seed": item["seed"],
            "run_id": item["run_id"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "census_ordering_sha256": fixture.census.ordering_sha256(),
            "state_count": 128,
            "maximum_endpoint_discrepancy": 1.0,
            "maximum_absolute_residual_n": 0.5,
            "maximum_absolute_residual_m": 0.25,
            "direct_residual_bound_n": 50.0,
            "finite_reference_bound": 26.0,
            "signed_gap": 24.0,
            "discrepancy_witness_state_id": _member(0),
            "residual_n_witness_state_id": _member(1),
            "residual_m_witness_state_id": _member(2),
            "secondary_diagnostics": _secondary_diagnostics(),
        }
        generation.payload_path(0).write_bytes(
            canonical_json_bytes(document) + b"\n"
        )

    def test_a_foreign_route_orphan_is_not_adopted(self) -> None:
        """Review reproducer: the orphan was adopted and the octet aggregated."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()  # create the durable schedule
            self._orphan(fixture, "some-other-registered-route-v1")
            with self.assertRaises((Exp1bScheduleStateError, Exp1bEvidenceError)):
                fixture.open()

    def test_a_matching_route_orphan_is_still_adopted(self) -> None:
        """The reconciliation itself must keep working for the real route."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()
            self._orphan(fixture, BRIDGE_ROUTE_ID)
            reopened = fixture.open()
            self.assertEqual(reopened.served_positions, (0,))

    def test_payload_read_requires_the_registered_route(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()
            self._orphan(fixture, "another-route-v1")
            with self.assertRaisesRegex(Exp1bEvidenceError, "was produced under route"):
                fixture.generation().read_payload(0)

    def test_every_layer_checks_the_route(self) -> None:
        """Payload read, orphan adoption, reload, and aggregation, all four."""

        checked = {
            "read_payload": (
                _ROOT / "scripts" / "policy_improvement_exp1b_evidence.py"
            ),
            "orphan": _ROOT / "scripts" / "policy_improvement_exp1b_schema.py",
            "reload": _ROOT / "scripts" / "policy_improvement_exp1b_bridge.py",
            "aggregate": _ROOT / "scripts" / "policy_improvement_exp1b_aggregate.py",
        }
        for label, path in checked.items():
            with self.subTest(layer=label):
                source = path.read_text(encoding="utf-8")
                self.assertIn("BRIDGE_ROUTE_ID", source)
                self.assertIn("route_id", source)


class Exp1bBaseArtifactBufferTest(unittest.TestCase):
    """Finding 6: Stage A must restore the exact bytes it authenticated.

    Pre-existing in the shared base-policy module, but the Stage A route depends
    on it, so it is fixed and covered here. The authentication half runs without
    Torch; the deserialization half needs it and is pinned structurally.
    """

    def test_the_authenticated_record_carries_its_verified_buffer(self) -> None:
        import dataclasses

        from scripts.policy_improvement_base_policy_restore import (
            AuthenticatedBasePolicy,
        )

        names = {field.name for field in dataclasses.fields(AuthenticatedBasePolicy)}
        self.assertIn("payload_bytes", names)
        buffer_field = next(
            field
            for field in dataclasses.fields(AuthenticatedBasePolicy)
            if field.name == "payload_bytes"
        )
        # Large, and the whole-file digest is already the identity.
        self.assertFalse(buffer_field.repr)
        self.assertFalse(buffer_field.compare)

    def test_restore_refuses_a_record_without_a_verified_buffer(self) -> None:
        """A hand-built AuthenticatedBasePolicy cannot reach `torch.load`.

        The guard runs before the Torch import path is exercised, so this test
        runs in a checkout without Torch.
        """

        from scripts.policy_improvement_base_policy_restore import (
            AuthenticatedBasePolicy,
            BasePolicyRestoreError,
            restore_base_policy_state,
        )

        forged = AuthenticatedBasePolicy(
            path=Path("/nonexistent/base.pt"),
            size_bytes=128,
            checkpoint_sha256=_digest("base"),
            architecture_sha256=_digest("arch"),
            model_state_sha256=_digest("model"),
            producer_git_commit="a" * 40,
            producer_source_manifest_sha256=_digest("manifest"),
            training_procedure_sha256=_digest("procedure"),
            training_split_ordered_record_sha256=_digest("train"),
            training_dataset_manifest_sha256=_digest("dataset"),
            device=1,
            inode=2,
        )
        with self.assertRaises(BasePolicyRestoreError):
            restore_base_policy_state(
                forged,
                model_config={},
                protocol_architecture={},
            )

    def test_restore_deserializes_the_buffer_not_the_path(self) -> None:
        """Structural: the load argument must be the authenticated buffer.

        This is the exact shape of the Stage B fix, applied to the base artifact.
        """

        source = (
            _ROOT / "scripts" / "policy_improvement_base_policy_restore.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "restore_base_policy_state"
        )
        loads = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        ]
        self.assertEqual(len(loads), 1)
        argument = loads[0].args[0]
        self.assertIsInstance(argument, ast.Call)
        self.assertEqual(argument.func.attr, "BytesIO")
        body = ast.get_source_segment(source, function)
        assert body is not None
        self.assertNotIn("torch.load(\n            authenticated.path", body)
        # And a replacement after authentication is reported.
        self.assertIn(
            "was replaced between authentication and restore", body
        )

    def test_authentication_buffers_the_bytes_it_hashed(self) -> None:
        """The authenticator must retain what it read, not re-read it later."""

        source = (
            _ROOT / "scripts" / "policy_improvement_base_policy_restore.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "authenticate_base_policy_artifact"
        )
        body = ast.get_source_segment(source, function)
        assert body is not None
        self.assertIn("chunks.append(block)", body)
        self.assertIn('payload_bytes = b"".join(chunks)', body)
        self.assertIn("payload_bytes=payload_bytes", body)
        # The buffer is re-hashed before it is handed on.
        self.assertIn("hashlib.sha256(payload_bytes).hexdigest()", body)


class Exp1bManifestBindingTest(unittest.TestCase):
    """Finding 4: provenance must enforce the manifest digests it records."""

    def _tamper(self, fixture: Any, position: int, **changes: Any) -> None:
        """Rewrite one run manifest in place, leaving provenance untouched."""

        directory = fixture.generation().checkpoint_directory_for(position)
        path = directory / "run_manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(changes)
        path.unlink()
        path.write_bytes(canonical_json_bytes(document) + b"\n")

    def test_an_edited_manifest_no_longer_opens_the_route(self) -> None:
        """Review reproducer: the route opened as sealed_octet_complete anyway.

        The edited fields all appear in the canonical manifest, so its digest
        changes; provenance recorded the original. Comparing the two by position
        is the check that was missing.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            counters = dict(fixture.run_manifests[0]["counters"])
            counters["train_step_count"] = counters["train_step_count"] + 1
            self._tamper(
                fixture,
                0,
                initialization_artifact_sha256=_digest("another-base"),
                restored_base_model_state_sha256=_digest("another-state"),
                counters=counters,
            )
            with self.assertRaisesRegex(
                Exp1bBridgeError, "does not match the digest its provenance recorded"
            ):
                fixture.open()

    def test_a_manifest_naming_another_admission_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            # Keep the recorded digest consistent so the position check passes
            # and the unanimity check is the one that fires.
            directory = fixture.generation().checkpoint_directory_for(3)
            path = directory / "run_manifest.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["admission_sha256"] = _digest("another-admission")
            path.unlink()
            path.write_bytes(canonical_json_bytes(document) + b"\n")
            provenance = dict(fixture.provenance)
            digests = list(provenance["run_manifest_sha256s"])
            digests[3] = exp1b_document_sha256(document)
            fixture.rewrite_provenance(run_manifest_sha256s=digests)
            with self.assertRaisesRegex(
                Exp1bBridgeError, "different signed admission"
            ):
                fixture.open()

    def test_a_manifest_naming_another_producer_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            directory = fixture.generation().checkpoint_directory_for(5)
            path = directory / "run_manifest.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["producer_attestation"] = {
                **document["producer_attestation"],
                "runtime_sha256": _digest("another-full-par"),
            }
            path.unlink()
            path.write_bytes(canonical_json_bytes(document) + b"\n")
            digests = list(fixture.provenance["run_manifest_sha256s"])
            digests[5] = exp1b_document_sha256(document)
            fixture.rewrite_provenance(run_manifest_sha256s=digests)
            with self.assertRaisesRegex(
                Exp1bBridgeError, "different producer runtime"
            ):
                fixture.open()

    def test_finalization_requires_unanimous_producer_attestation(self) -> None:
        """Provenance is derived from the eight manifests, not asserted."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _RouteFixture(root)
            # A second generation whose seed-6 manifest disagrees.
            other_root = root / "evidence2"
            gen_dir = other_root / "gen-0002"
            (gen_dir / "payloads").mkdir(parents=True)
            (gen_dir / "checkpoints").mkdir(parents=True)
            generation = open_evidence_generation(
                evidence_root=other_root, generation_id="gen-0002"
            )
            for offset, manifest in enumerate(fixture.run_manifests):
                document = dict(manifest)
                if offset == 6:
                    document["producer_attestation"] = {
                        **document["producer_attestation"],
                        "launcher_sha256": _digest("a-different-launcher"),
                    }
                generation.publish_checkpoint(
                    seed_position=offset,
                    checkpoint_bytes=_CHECKPOINT_BYTES[offset],
                    run_manifest=document,
                )
            producer_authorization, _ = _authorizations()
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "different producer runtime"
            ):
                finalize_exp1b_evidence(
                    generation=generation,
                    producer_authorization=producer_authorization,
                    admitted_base_artifact=fixture.admitted_base_path,
                    admission_sha256=fixture.admission_sha,
                    base_descriptor=_base_descriptor(),
                    registry_rows=registry_rows_by_run_id(fixture.registry),
                    exp1b_protocol_sha256=fixture.protocol_sha,
                    exp1b_registry_sha256=fixture.registry_sha,
                    exp1b_amendment_sha256=fixture.amendment_sha,
                    parent=dict(fixture.protocol["parent"]),
                    ordered_population_sha256=fixture.census.ordered_record_sha256,
                    population_binding_sha256=fixture.census.binding_sha256,
                    attestation={
                        "source_git_commit": _COMMIT,
                        "runtime_sha256": _RUNTIME,
                        "launcher_sha256": _LAUNCHER,
                        "runtime_authorization_sha256": _AUTHORIZATION,
                    },
                )

    def test_every_manifest_carries_the_admission_and_producer(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            for position in range(UNITS):
                document = json.loads(
                    (
                        generation.checkpoint_directory_for(position)
                        / "run_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(document["admission_sha256"], fixture.admission_sha)
                self.assertEqual(
                    document["producer_attestation"]["role"], FULL_ROLE
                )
                self.assertEqual(
                    document["producer_attestation"]["runtime_sha256"], _RUNTIME
                )
            # And the provenance's producer identity is exactly theirs.
            self.assertEqual(
                dict(fixture.provenance["producer_attestation"]),
                dict(document["producer_attestation"]),
            )


class Exp1bCenteringParityTest(unittest.TestCase):
    """Finding 1: the check must see the trainer's estimator, not only its own."""

    def test_the_adapter_reconstructs_the_trainer_tensor(self) -> None:
        """Structural: the trainer's own primitives, not a reimplementation."""

        source = (_ROOT / "policy_improvement_full_backend.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        session = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "Exp1bSealedEvaluationSession"
        )
        methods = {
            node.name for node in session.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_trainer_advantages", methods)
        self.assertIn("_constructed_advantages", methods)
        trainer_method = next(
            node
            for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "_trainer_advantages"
        )
        body = ast.get_source_segment(source, trainer_method)
        assert body is not None
        # The trainer's own primitives, as the v2 oracle uses them.
        self.assertIn("compute_exact_baseline_summation(", body)
        self.assertIn("_clip_and_recenter_advantages(", body)
        # Under the frozen base law, which is Experiment 1B's operator policy.
        self.assertIn("record.base_probabilities", body)

    def test_all_three_quantities_reach_the_durable_record(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            result = aggregate_exp1b_result(route.sealed_octet())
            for row in result.seed_rows:
                centering = row.secondary_diagnostics["centering_parity"]
                for field in (
                    "constructed_centering_roundoff",
                    "training_estimator_centering_defect",
                    "training_estimator_parity_max_abs_error",
                    "training_estimator_parity_tolerance",
                    "clipping_kind",
                    "clip_value",
                    "trainer_reconstruction",
                ):
                    self.assertIn(field, centering)
                self.assertEqual(
                    centering["trainer_reconstruction"],
                    TRAINER_RECONSTRUCTION_CONTRACT,
                )

    def test_a_trainer_parity_error_refuses_the_seed(self) -> None:
        """The mutation this check exists for: the trainer tensor drifts.

        Constructed roundoff stays valid — the independently built tensor is
        still exactly centered — and only the elementwise parity error moves.
        Before finding 1 there was no field for it and nothing to fail.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            evaluator = _evaluator(0)
            evaluator["secondary_diagnostics"] = _secondary_diagnostics(
                centering_parity={
                    # Roundoff still inside tolerance...
                    "constructed_centering_roundoff": 1.1e-9,
                    # ...while the trainer tensor disagrees elementwise.
                    "training_estimator_parity_max_abs_error": 5.0e-6,
                }
            )
            with self.assertRaisesRegex(Exp1bBridgeError, "differ beyond tolerance"):
                route.serve(
                    seed_position=0,
                    seed=item["seed"],
                    run_id=item["run_id"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **evaluator,
                )
            self.assertFalse(fixture.generation().has_payload(0))

    def test_a_trainer_centering_defect_refuses_the_seed(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            evaluator = _evaluator(0)
            evaluator["secondary_diagnostics"] = _secondary_diagnostics(
                centering_parity={
                    "training_estimator_centering_defect": 4.0e-6,
                }
            )
            with self.assertRaisesRegex(Exp1bBridgeError, "centering defect"):
                route.serve(
                    seed_position=0,
                    seed=item["seed"],
                    run_id=item["run_id"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **evaluator,
                )

    def test_the_clipping_identity_is_pinned(self) -> None:
        """A trainer that silently stopped clipping is a schema failure."""

        for override, message in (
            ({"clipping_kind": "none"}, "named a clip value"),
            ({"clipping_kind": "some_other_scheme"}, "unregistered clipping kind"),
            ({"clip_value": 0.0}, "clip value must be positive"),
            (
                {"trainer_reconstruction": "something_else"},
                "registered reconstruction contract",
            ),
        ):
            with self.subTest(override=override):
                document = _secondary_diagnostics(centering_parity=override)
                with self.assertRaisesRegex(Exp1bSchemaError, message):
                    validate_secondary_diagnostics(document, path="probe")

    def test_an_unclipped_estimator_is_accepted_without_a_clip_value(self) -> None:
        document = _secondary_diagnostics(
            centering_parity={"clipping_kind": "none", "clip_value": None}
        )
        validate_secondary_diagnostics(document, path="probe")

    def test_the_registered_tolerances_come_from_the_v2_amendment(self) -> None:
        amendment = _load(_V2 / "amendments" / "theory_bridge_v2.json")
        centering = amendment["centering_contract"]
        self.assertEqual(
            centering["constructed_centering_roundoff_absolute_tolerance"],
            CENTERING_PARITY_TOLERANCE,
        )
        self.assertEqual(
            centering["training_estimator_parity_absolute_tolerance"],
            CENTERING_PARITY_TOLERANCE,
        )
        self.assertEqual(
            centering["trainer_reconstruction"], TRAINER_RECONSTRUCTION_CONTRACT
        )


@unittest.skipUnless(_torch_available(), "Torch is not installed in this checkout")
class Exp1bTorchCenteringParityTest(unittest.TestCase):
    """Finding 1's Torch-backed mutation: perturb the trainer estimator only."""

    def test_a_mutated_trainer_estimator_is_caught(self) -> None:
        self.skipTest(
            "Requires a restored four-module checkpoint; see the handoff's "
            "unresolved-Torch section."
        )


def _die_holding_claim(fixture: Any, position: int) -> None:
    """Leave a durable claim on ``position`` with no live process behind it.

    Exactly what a crash leaves: the claim is committed while the lease is held,
    then the lease goes away without the claim being abandoned or finalized. A
    clean ``with`` unwind would abandon it, which is the opposite case.
    """

    route = fixture.open()
    item = fixture.sealed[position]
    generation = fixture.generation()
    schedule = route._schedule  # noqa: SLF001 - simulating a crashed peer
    with generation.claim_lease(position):
        schedule.claim(
            seed_position=position,
            seed=item["seed"],
            run_id=item["run_id"],
            checkpoint_sha256=item["checkpoint_sha256"],
            evaluation_population="validation_bridge",
            depths=(DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M),
            lease_acquired=True,
        )


class Exp1bClaimBeforeEvaluationTest(unittest.TestCase):
    """Finding 2: claim first, evaluate under the claim, reclaim a dead claim."""

    def test_a_refused_request_does_no_evaluator_work(self) -> None:
        """Review reproducer: an out-of-order request traversed the census first."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            touched: list[str] = []
            evaluator = _evaluator(3)
            base = evaluator["endpoint_values"]
            evaluator["endpoint_values"] = lambda state_id, depth: (
                touched.append(state_id),
                base(state_id, depth),
            )[1]
            item = fixture.sealed[3]
            with self.assertRaises(Exp1bSchemaError):
                route.serve(
                    seed_position=3,  # position 0 is expected first
                    seed=item["seed"],
                    run_id=item["run_id"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **evaluator,
                )
            self.assertEqual(touched, [])

    def test_a_malformed_record_counts_a_durable_attempt(self) -> None:
        """Review reproducer: the slot stayed unclaimed with attempt count 0.

        Once the claim precedes the evaluator, a diagnostics failure cannot leave
        zero durable state: admission already happened. Counting the attempt is
        the honest record of that, and it is the tradeoff this design takes.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            evaluator = _evaluator(0)
            evaluator["secondary_diagnostics"] = _secondary_diagnostics(
                mixture_identity={"maximum_identity_total_variation": 9.0e-6}
            )
            with self.assertRaises(Exp1bBridgeError):
                route.serve(
                    seed_position=0,
                    seed=item["seed"],
                    run_id=item["run_id"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    evaluation_population="validation_bridge",
                    **evaluator,
                )
            reopened = fixture.open()
            self.assertEqual(reopened.attempt_counts()[0], 1)
            self.assertEqual(reopened.served_positions, ())
            self.assertFalse(fixture.generation().has_payload(0))
            # And the slot is still usable.
            item = fixture.sealed[0]
            reopened.serve(
                seed_position=0,
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            self.assertEqual(fixture.open().served_positions, (0,))

    def test_the_secondary_traversal_runs_inside_the_claim(self) -> None:
        """The 128-member traversal is a callable, invoked under the claim."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            order: list[str] = []
            item = fixture.sealed[0]

            def diagnostics() -> dict[str, Any]:
                order.append("secondary")
                return _secondary_diagnostics()

            evaluator = _evaluator(0)
            evaluator["secondary_diagnostics"] = diagnostics
            with route.claim_seed(
                0,
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
            ) as claim:
                order.append("claimed")
                claim.serve(**evaluator)
            self.assertEqual(order, ["claimed", "secondary"])

    def test_a_dead_claim_is_reclaimed_and_keeps_its_attempt(self) -> None:
        """Review reproducer: the reopened route failed with 'already claimed'.

        The lease is an flock, so the kernel releases it when the claiming
        process dies. A free lease on a slot the schedule records as claimed is
        proof of death, and needs no clock.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            # A process that claimed and died: fork a child that takes the claim
            # and exits without unwinding. The kernel releases its flock; the
            # durable claim stays behind. `gen.close()` would not do -- that
            # unwinds cleanly and abandons the claim, which is the opposite case.
            _die_holding_claim(fixture, 0)
            reopened = fixture.open()
            self.assertEqual(reopened.attempt_counts()[0], 1)
            payload = reopened.serve(
                seed_position=0,
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            self.assertEqual(payload.seed_position, 0)
            # The attempt from the dead claim is preserved, not reset.
            final = fixture.open()
            self.assertEqual(final.attempt_counts()[0], 2)
            self.assertEqual(final.served_positions, (0,))

    def test_a_live_claim_is_still_refused(self) -> None:
        """Reclaim must not become a way past a genuinely concurrent request."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            with route.claim_seed(
                0,
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
            ):
                other = fixture.open()
                with self.assertRaisesRegex(
                    Exp1bBridgeError, "being evaluated by a live process"
                ):
                    with other.claim_seed(
                        0,
                        seed=item["seed"],
                        run_id=item["run_id"],
                        checkpoint_sha256=item["checkpoint_sha256"],
                    ):
                        pass

    def test_the_claim_epoch_is_durable_and_monotone(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()
            for _ in range(2):
                _die_holding_claim(fixture, 0)
            document = json.loads(
                fixture.schedule_state_path.read_text(encoding="utf-8")
            )
            self.assertEqual(document["slots"][0]["claim_epoch"], 2)
            self.assertEqual(document["slots"][0]["attempts"], 2)

    def test_the_runtime_claims_before_building_the_backend(self) -> None:
        """Structural: no backend construction outside the claim block."""

        source = (
            _ROOT / "scripts" / "policy_improvement_exp1b_runtime.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "bridge_main"
        )
        withs = [node for node in ast.walk(function) if isinstance(node, ast.With)]
        claim_blocks = [
            node
            for node in withs
            if any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "claim_seed"
                for item in node.items
            )
        ]
        self.assertEqual(len(claim_blocks), 1)
        inside = {
            node.func.id
            for node in ast.walk(claim_blocks[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr
            for node in ast.walk(claim_blocks[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("backend_factory", inside)
        self.assertIn("prepare_exp1b_bridge", inside)
        # And nothing constructs the backend outside it.
        outside = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "backend_factory"
        ]
        self.assertEqual(len(outside), 1)


class Exp1bCrashRecoveryTest(unittest.TestCase):
    """Finding 3: interrupted Stage A publication and finalization must recover."""

    def test_the_installer_publishes_no_partial_artifact(self) -> None:
        """A crash mid-write leaves a temp, never a truncated final name."""

        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            target = directory / "artifact.bin"
            from scripts.policy_improvement_exp1b_evidence import _write_exclusive

            _write_exclusive(target, b"complete payload" * 64, label="artifact")
            self.assertEqual(target.read_bytes(), b"complete payload" * 64)
            # Single-linked, so `stable_bytes` will accept it.
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertEqual(
                [item.name for item in directory.iterdir()], ["artifact.bin"]
            )

    def test_a_leftover_temporary_is_reaped_and_nlink_recovers(self) -> None:
        """The link/unlink window: a crash there leaves the final at nlink 2."""

        from scripts.policy_improvement_exp1b_evidence import (
            _reap_partials,
            _TEMP_PREFIX,
            stable_bytes,
        )

        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            target = directory / "artifact.bin"
            target.write_bytes(b"payload" * 32)
            orphan = directory / f"{_TEMP_PREFIX}artifact.bin.999"
            os.link(target, orphan)
            self.assertEqual(target.stat().st_nlink, 2)
            # In that state the artifact is unreadable by the authenticator.
            with self.assertRaises(Exp1bEvidenceError):
                stable_bytes(target, label="artifact")
            _reap_partials(directory)
            self.assertEqual(target.stat().st_nlink, 1)
            payload, _digest, _size = stable_bytes(target, label="artifact")
            self.assertEqual(payload, b"payload" * 32)

    def test_an_identical_artifact_is_adopted_and_a_different_one_refused(self) -> None:
        from scripts.policy_improvement_exp1b_evidence import _install_or_adopt

        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            target = directory / "artifact.bin"
            payload = b"the same bytes" * 16
            self.assertEqual(
                _install_or_adopt(target, payload, label="artifact"), "installed"
            )
            self.assertEqual(
                _install_or_adopt(target, payload, label="artifact"), "adopted"
            )
            with self.assertRaisesRegex(Exp1bEvidenceError, "cannot be adopted"):
                _install_or_adopt(target, b"different bytes" * 16, label="artifact")

    def test_retrying_an_interrupted_seed_publication_succeeds(self) -> None:
        """Review reproducer: retrying seed 7 failed on its existing directory."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            self.assertEqual(generation.checkpoint_publication_state(7), "complete")
            # The identical retry is adopted rather than refused.
            generation.publish_checkpoint(
                seed_position=7,
                checkpoint_bytes=_CHECKPOINT_BYTES[7],
                run_manifest=fixture.run_manifests[7],
            )
            self.assertEqual(generation.checkpoint_publication_state(7), "complete")

    def test_a_checkpoint_only_seed_completes_without_retraining(self) -> None:
        """Crash between a seed's checkpoint and its manifest."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            (
                generation.checkpoint_directory_for(2) / "run_manifest.json"
            ).unlink()
            self.assertEqual(
                generation.checkpoint_publication_state(2), "checkpoint_only"
            )
            generation.publish_checkpoint(
                seed_position=2,
                checkpoint_bytes=_CHECKPOINT_BYTES[2],
                run_manifest=fixture.run_manifests[2],
            )
            self.assertEqual(generation.checkpoint_publication_state(2), "complete")

    def test_a_differing_retry_is_refused_rather_than_overwritten(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            other = dict(fixture.run_manifests[1])
            other["counters"] = {
                **other["counters"],
                "train_step_count": other["counters"]["train_step_count"] + 1,
            }
            with self.assertRaisesRegex(Exp1bEvidenceError, "cannot be adopted"):
                generation.publish_checkpoint(
                    seed_position=1,
                    checkpoint_bytes=_CHECKPOINT_BYTES[1],
                    run_manifest=other,
                )

    def test_a_partial_finalization_completes(self) -> None:
        """Review reproducer: base copy present, provenance absent, stuck.

        Finalization used to refuse the existing base artifact, and the caller
        read the refusal's message as a successful race. The generation stayed
        permanently unfinishable.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            provenance = json.loads(
                generation.provenance_path.read_text(encoding="utf-8")
            )
            # Roll back to "base copied, provenance not yet written".
            generation.provenance_path.unlink()
            self.assertTrue(
                (generation.directory / "base_policy" / "base_policy.pt").is_file()
            )
            producer_authorization, _ = _authorizations()
            finalize_exp1b_evidence(
                generation=generation,
                producer_authorization=producer_authorization,
                admitted_base_artifact=fixture.admitted_base_path,
                admission_sha256=fixture.admission_sha,
                base_descriptor=_base_descriptor(),
                registry_rows=registry_rows_by_run_id(fixture.registry),
                exp1b_protocol_sha256=fixture.protocol_sha,
                exp1b_registry_sha256=fixture.registry_sha,
                exp1b_amendment_sha256=fixture.amendment_sha,
                parent=dict(fixture.protocol["parent"]),
                ordered_population_sha256=fixture.census.ordered_record_sha256,
                population_binding_sha256=fixture.census.binding_sha256,
                attestation={
                    "source_git_commit": _COMMIT,
                    "runtime_sha256": _RUNTIME,
                    "launcher_sha256": _LAUNCHER,
                    "runtime_authorization_sha256": _AUTHORIZATION,
                },
            )
            self.assertTrue(generation.provenance_path.is_file())
            self.assertEqual(
                json.loads(
                    generation.provenance_path.read_text(encoding="utf-8")
                ),
                provenance,
            )

    def test_refinalizing_an_already_complete_octet_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            before = generation.provenance_path.read_bytes()
            producer_authorization, _ = _authorizations()
            finalize_exp1b_evidence(
                generation=generation,
                producer_authorization=producer_authorization,
                admitted_base_artifact=fixture.admitted_base_path,
                admission_sha256=fixture.admission_sha,
                base_descriptor=_base_descriptor(),
                registry_rows=registry_rows_by_run_id(fixture.registry),
                exp1b_protocol_sha256=fixture.protocol_sha,
                exp1b_registry_sha256=fixture.registry_sha,
                exp1b_amendment_sha256=fixture.amendment_sha,
                parent=dict(fixture.protocol["parent"]),
                ordered_population_sha256=fixture.census.ordered_record_sha256,
                population_binding_sha256=fixture.census.binding_sha256,
                attestation={
                    "source_git_commit": _COMMIT,
                    "runtime_sha256": _RUNTIME,
                    "launcher_sha256": _LAUNCHER,
                    "runtime_authorization_sha256": _AUTHORIZATION,
                },
            )
            self.assertEqual(generation.provenance_path.read_bytes(), before)
            fixture.open()  # and it still opens

    def test_a_different_provenance_is_refused_not_treated_as_a_race(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            producer_authorization, _ = _authorizations()
            with self.assertRaisesRegex(Exp1bEvidenceError, "cannot be adopted"):
                finalize_exp1b_evidence(
                    generation=generation,
                    producer_authorization=producer_authorization,
                    admitted_base_artifact=fixture.admitted_base_path,
                    admission_sha256=fixture.admission_sha,
                    base_descriptor=_base_descriptor(),
                    registry_rows=registry_rows_by_run_id(fixture.registry),
                    exp1b_protocol_sha256=fixture.protocol_sha,
                    exp1b_registry_sha256=fixture.registry_sha,
                    exp1b_amendment_sha256=fixture.amendment_sha,
                    parent=dict(fixture.protocol["parent"]),
                    # A different census binding produces a different provenance.
                    ordered_population_sha256=fixture.census.ordered_record_sha256,
                    population_binding_sha256=_digest("another-binding"),
                    attestation={
                        "source_git_commit": _COMMIT,
                        "runtime_sha256": _RUNTIME,
                        "launcher_sha256": _LAUNCHER,
                        "runtime_authorization_sha256": _AUTHORIZATION,
                    },
                )

    def test_the_runtime_no_longer_string_matches_the_race(self) -> None:
        source = (
            _ROOT / "scripts" / "policy_improvement_exp1b_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('if "already exists" in str(exc)', source)


class Exp1bPublicationBindingTest(unittest.TestCase):
    """Finding 5: publication must re-derive every claim from the generation."""

    def _completed(self, tmp: str) -> tuple[Any, Any, Any]:
        fixture = _RouteFixture(Path(tmp))
        route = fixture.open()
        _serve_all(route, fixture)
        return fixture, route, aggregate_exp1b_result(route.sealed_octet())

    #: The exact minimal document the review published through the old
    #: mint-plus-writer pair. Four keys: enough to satisfy the shallow structural
    #: floor, and missing the entire registered result -- no seed rows, no signed
    #: gaps, no payload digests, no 10,000-value replicate vector, no access
    #: flags. It is the regression for "the proof said nothing about what was
    #: checked".
    _MALFORMED_RESULT = {
        "schema_name": RESULT_SCHEMA_NAME,
        "protocol_id": PROTOCOL_ID,
        "route_id": BRIDGE_ROUTE_ID,
        "interval": {"status": "available"},
    }

    def _publish(self, fixture: Any, route: Any, result: Any) -> bytes:
        document = validated_exp1b_document(
            result,
            route=route,
            protocol=fixture.protocol,
            registry=fixture.registry,
            amendment=fixture.amendment,
            provenance=fixture.provenance,
            admission=fixture.admission_document,
        )
        return document

    def test_the_normal_path_still_publishes(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            self.assertTrue(self._publish(fixture, route, result))

    def test_a_replaced_generation_digest_is_refused(self) -> None:
        """Review reproducer: replaced digest plus attempts of 99 validated."""

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            forged = replace(
                result,
                generation_sha256=_digest("another-generation"),
                attempt_counts=(99,) * UNITS,
            )
            with self.assertRaisesRegex(
                Exp1bAggregateError, "not the durable generation"
            ):
                self._publish(fixture, route, forged)

    def test_replaced_attempt_counts_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            forged = replace(result, attempt_counts=(99,) * UNITS)
            with self.assertRaisesRegex(
                Exp1bAggregateError, "not the durable schedule"
            ):
                self._publish(fixture, route, forged)

    def test_replaced_payload_digests_are_refused(self) -> None:
        """Review reproducer: eight other valid digests passed the auditor."""

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            forged = replace(
                result,
                payload_digests=tuple(
                    _digest(f"other-payload-{offset}") for offset in range(UNITS)
                ),
            )
            with self.assertRaisesRegex(
                Exp1bAggregateError, "not the durable schedule"
            ):
                self._publish(fixture, route, forged)

    def test_an_edited_secondary_record_is_refused(self) -> None:
        """Review reproducer: target lag 0.125 -> 9.5 with the original digest."""

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            rows = list(result.seed_rows)
            secondary = _plain_secondary(rows[0].secondary_diagnostics)
            secondary["target_lag"] = {
                **secondary["target_lag"],
                "maximum_absolute_target_lag": 9.5,
            }
            rows[0] = replace(rows[0], secondary_diagnostics=secondary)
            forged = replace(result, seed_rows=tuple(rows))
            with self.assertRaisesRegex(
                Exp1bAggregateError, "secondary diagnostics differ"
            ):
                self._publish(fixture, route, forged)

    def test_an_edited_primary_row_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            rows = list(result.seed_rows)
            rows[2] = replace(rows[2], signed_gap=rows[2].signed_gap + 1.0)
            with self.assertRaisesRegex(
                Exp1bAggregateError, "differs from its durable payload"
            ):
                self._publish(fixture, route, replace(result, seed_rows=tuple(rows)))

    def test_a_replaced_octet_cannot_aggregate(self) -> None:
        """`dataclasses.replace` carries the token forward; identity does not."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            octet = route.sealed_octet()
            self.assertTrue(octet.is_issued())
            copied = replace(octet)
            # The token survived the copy...
            self.assertIs(copied._issued_by, octet._issued_by)
            # ...but the copy is not an object this module issued.
            self.assertFalse(copied.is_issued())
            with self.assertRaisesRegex(
                Exp1bAggregateError, "not issued by a completed authenticated route"
            ):
                aggregate_exp1b_result(copied)

    def test_the_minimal_malformed_document_has_no_way_in(self) -> None:
        """Review reproducer: `malformed_result_published True True`.

        The old writer took ``(document, document_sha256, proof)``. The mint took
        arbitrary bytes, so minting a proof for a four-key document and handing it
        straight back published it: byte-for-byte readback, recorded digest, and
        virtually the whole registered result missing. The proof recorded that
        *something* had been minted; it never recorded *what had been checked*.

        There is now no document argument and no mint. Publication takes the
        authenticated route and the study documents, serializes the result
        itself, and runs the validator and the independent auditor over the exact
        bytes it then writes. This test walks every exported way in.
        """

        import inspect

        import scripts.policy_improvement_exp1b_evidence as evidence

        payload = canonical_json_bytes(self._MALFORMED_RESULT) + b"\n"
        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()

            # 1. The mint and the proof type are gone, not renamed or hidden.
            for gone in (
                "_issue_publication_proof",
                "issue_publication_proof",
                "Exp1bPublicationProof",
                "_ISSUED_PROOFS",
                "_PROOF_ISSUANCE",
            ):
                self.assertFalse(hasattr(evidence, gone), gone)

            # 2. There is no caller-supplied document, digest, audit, or proof.
            for name in ("publish_result", "complete_result_publication"):
                parameters = inspect.signature(
                    getattr(evidence.Exp1bEvidenceGeneration, name)
                ).parameters
                for absent in (
                    "document",
                    "expected_document",
                    "document_sha256",
                    "audit",
                    "proof",
                ):
                    self.assertNotIn(absent, parameters, f"{name}.{absent}")
                for present in ("result", "route", "protocol", "registry", "amendment"):
                    self.assertIn(present, parameters, f"{name}.{present}")

            # 3. The exported staged writer refuses the two result pathnames, so
            #    "generic writer plus a result path" is not a way around the gate
            #    either.
            for path in (generation.result_path, generation.result_digest_path):
                with self.assertRaisesRegex(
                    Exp1bEvidenceError, "published only through"
                ):
                    evidence.install_durable_artifact(
                        path, payload, label="malformed result probe"
                    )
            self.assertFalse(generation.result_path.exists())

            # 4. The real path publishes the real result, and it is not that
            #    document: the full vector and the eight rows are all there.
            generation.publish_result(**_publication_inputs(fixture, route, result))
            published, digest = generation.read_published_result()
            self.assertNotEqual(published, payload)
            self.assertEqual(digest, result.document_sha256())
            reloaded = json.loads(published.decode("utf-8"))
            self.assertEqual(
                len(reloaded["interval"]["replicate_means_hex"]), REPLICATES
            )
            self.assertEqual(len(reloaded["seed_rows"]), UNITS)

    def test_a_raw_written_malformed_document_cannot_become_published(self) -> None:
        """No in-process gate stops ``open()``. The journal still does.

        A malformed document dropped straight onto the result pathname has no
        sidecar, so the generation reads as ``document_only`` -- a crash state, not
        a published result. Completing that journal runs the full gate and then
        refuses, because the durable bytes are not the bytes it validated. So the
        two-file journal is itself part of the boundary, and the reviewer's
        document cannot reach ``read_published_result`` by any route.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            generation.result_path.write_bytes(
                canonical_json_bytes(self._MALFORMED_RESULT) + b"\n"
            )
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "already carries another result"
            ):
                generation.complete_result_publication(
                    **_publication_inputs(fixture, route, result)
                )
            with self.assertRaisesRegex(Exp1bEvidenceError, "cannot be adopted"):
                generation.publish_result(
                    **_publication_inputs(fixture, route, result)
                )
            self.assertFalse(generation.result_digest_path.exists())
            with self.assertRaises(Exp1bEvidenceError):
                generation.read_published_result()

    def test_publication_runs_the_independent_audit_itself(self) -> None:
        """The writer is the thing that audits, so the audit cannot be skipped.

        Proved by making the auditor refuse: if publication were still trusting a
        caller-supplied token, an auditor that rejects everything would not stop
        it. Nothing is written.
        """

        import scripts.policy_improvement_exp1b_auditor as auditor

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            original = auditor.audit_exp1b_result_document

            def _refusing(*args: Any, **kwargs: Any) -> Any:
                raise auditor.Exp1bAuditError("synthetic audit refusal")

            auditor.audit_exp1b_result_document = _refusing
            try:
                with self.assertRaisesRegex(
                    Exp1bEvidenceError, "synthetic audit refusal"
                ):
                    generation.publish_result(
                        **_publication_inputs(fixture, route, result)
                    )
            finally:
                auditor.audit_exp1b_result_document = original
            self.assertFalse(generation.result_path.exists())
            self.assertFalse(generation.result_digest_path.exists())
            # With the real auditor back, the same call publishes.
            generation.publish_result(**_publication_inputs(fixture, route, result))
            self.assertTrue(generation.result_path.is_file())

    def test_publication_requires_the_authenticated_route(self) -> None:
        """A duck type with the right methods is not the route that served."""

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)

            class _LooksLikeARoute:
                generation = route.generation
                provenance_document = route.provenance_document

                def reload_durable_state(self) -> Any:
                    return route.reload_durable_state()

            with self.assertRaisesRegex(
                Exp1bAggregateError, "authenticated route that served"
            ):
                validated_exp1b_document(
                    result,
                    route=_LooksLikeARoute(),
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    admission=fixture.admission_document,
                )
            # And the durable writer refuses it for the same reason, because the
            # writer is what calls the validator.
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "authenticated route that served"
            ):
                fixture.generation().publish_result(
                    result=result,
                    route=_LooksLikeARoute(),
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    admission=fixture.admission_document,
                )

    def test_a_route_into_another_generation_cannot_publish_here(self) -> None:
        """Validating against generation A does not authorize writing into B."""

        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            fixture_a, route_a, result_a = self._completed(first)
            fixture_b, _route_b, _result_b = self._completed(second)
            self.assertNotEqual(
                fixture_a.generation().generation_sha256,
                fixture_b.generation().generation_sha256,
            )
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "route that served this evidence generation"
            ):
                fixture_b.generation().publish_result(
                    **_publication_inputs(fixture_a, route_a, result_a)
                )
            self.assertFalse(fixture_b.generation().result_path.exists())

    def test_the_auditor_compares_the_generation_digest_exactly(self) -> None:
        """Review reproducer: a result whose ONLY change was the generation.

        The auditor used to check digest syntax and nothing else, so a document
        about a different evidence generation audited clean and published.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            document = self._publish(fixture, route, result)
            moved = json.loads(document.decode("utf-8"))
            moved["generation_sha256"] = _digest("another-generation")
            payload = canonical_json_bytes(moved) + b"\n"
            with self.assertRaisesRegex(
                Exp1bAuditError, "not the one the authenticated provenance"
            ):
                audit_exp1b_result_document(
                    payload,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_a_result_and_a_study_document_forged_together_are_refused(self) -> None:
        """Self-consistency is not authentication.

        The audit used to pin the three study digests only to the documents the
        caller handed it, so a result whose ``exp1b_protocol_sha256`` was rewritten
        together with a matching protocol agreed with itself. The durable
        provenance records which study the octet was finalized against, and
        publication reads that off the authenticated route, so it is the anchor.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            document = self._publish(fixture, route, result)
            # A complete, internally consistent forged study: the protocol
            # differs in a field the schema tolerates and the provenance does not
            # record, and the registry and amendment are re-cited to it. Nothing
            # in the three documents disagrees with anything else in them.
            forged_protocol = json.loads(json.dumps(fixture.protocol))
            forged_protocol["created_date"] = "2026-09-10T00:00:00Z"
            forged_sha = exp1b_document_sha256(forged_protocol)
            self.assertNotEqual(forged_sha, fixture.protocol_sha)
            forged_registry = json.loads(json.dumps(fixture.registry))
            forged_registry["protocol_sha256"] = forged_sha
            forged_registry_sha = exp1b_document_sha256(forged_registry)
            forged_amendment = json.loads(json.dumps(fixture.amendment))
            forged_amendment["protocol_sha256"] = forged_sha
            forged_amendment["registry_sha256"] = forged_registry_sha
            validate_exp1b_protocol(forged_protocol)
            validate_exp1b_registry(forged_registry, protocol_sha256=forged_sha)
            validate_exp1b_amendment(
                forged_amendment,
                protocol_sha256=forged_sha,
                registry_sha256=forged_registry_sha,
            )

            moved = json.loads(document.decode("utf-8"))
            moved["exp1b_protocol_sha256"] = forged_sha
            moved["exp1b_registry_sha256"] = forged_registry_sha
            moved["exp1b_amendment_sha256"] = exp1b_document_sha256(forged_amendment)
            payload = canonical_json_bytes(moved) + b"\n"
            with self.assertRaisesRegex(
                Exp1bAuditError, "not the one the authenticated provenance"
            ):
                audit_exp1b_result_document(
                    payload,
                    protocol=forged_protocol,
                    registry=forged_registry,
                    amendment=forged_amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                )

    def test_a_replaced_evaluator_identity_with_no_admission_is_refused(self) -> None:
        """Review reproducer: every evaluator field swapped, admission omitted.

        The auditor checks both role identities against the signed authorizations
        only inside ``if admission is not None``, and the public writer defaulted
        ``admission`` to ``None``. So a caller could take the genuine result,
        replace the whole evaluator identity with another well-formed,
        producer-distinct one, omit the admission, and publish the canonical
        durable result with the authorization check skipped.

        Two independent closures now, either of which alone stops it: admission is
        mandatory, and the durable comparison binds the evaluator identity to the
        route's own live Stage B identity, which is not a caller argument.
        """

        replaced = dict(
            role=THEORY_BRIDGE_ROLE,
            source_git_commit="9" * 40,
            runtime_sha256=_digest("another-evaluator-runtime"),
            launcher_sha256=_digest("another-evaluator-launcher"),
            runtime_authorization_sha256=_digest("another-evaluator-authorization"),
        )

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            genuine = _plain_secondary(result.evaluator_attestation)
            for field, value in replaced.items():
                if field == "role":
                    continue
                self.assertNotEqual(genuine[field], value, field)
            forged = replace(result, evaluator_attestation=replaced)

            # 1. Admission can no longer be omitted at all.
            with self.assertRaises(TypeError):
                generation.publish_result(
                    result=forged,
                    route=route,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                )
            with self.assertRaisesRegex(
                Exp1bAggregateError, "requires the owner-signed admission"
            ):
                validated_exp1b_document(
                    forged,
                    route=route,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    admission=None,  # type: ignore[arg-type]
                )

            # 2. And with the real admission supplied, the route binding refuses
            #    it before the auditor is even reached.
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "evaluator attestation is not the authenticated"
            ):
                generation.publish_result(
                    **_publication_inputs(fixture, route, forged)
                )
            self.assertFalse(generation.result_path.exists())
            # The genuine identity still publishes.
            generation.publish_result(**_publication_inputs(fixture, route, result))
            self.assertTrue(generation.result_path.is_file())

    def test_a_replaced_census_ordering_is_refused_with_the_real_admission(self) -> None:
        """Review reproducer: only ``census_ordering_sha256`` changed, real admission.

        The durable comparison covered rows, payload contents, and payload hashes
        but not this top-level identity, and the auditor treated it as
        digest-shaped and nothing more. So the result could name an ordering its
        own evidence did not come from, with every number left intact.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            forged = replace(
                result, census_ordering_sha256=_digest("another-census-ordering")
            )
            self.assertNotEqual(
                forged.census_ordering_sha256, route.census_ordering_sha256
            )
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "census ordering digest is not the authenticated"
            ):
                generation.publish_result(
                    **_publication_inputs(fixture, route, forged)
                )
            self.assertFalse(generation.result_path.exists())

            # And the standalone auditor rederives it from the registered
            # population rather than accepting the digest's shape.
            document = self._publish(fixture, route, result)
            moved = json.loads(document.decode("utf-8"))
            moved["census_ordering_sha256"] = _digest("another-census-ordering")
            with self.assertRaisesRegex(
                Exp1bAuditError, "not the ordering the registered"
            ):
                audit_exp1b_result_document(
                    canonical_json_bytes(moved) + b"\n",
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=route.population_registry_document,
                )

    def test_a_provenance_deleted_after_authentication_stops_publication(self) -> None:
        """Review reproducer: `provenance_exists False`, `result_state complete`.

        The route's ``provenance_document`` is an in-memory copy taken when the
        route opened, so a stale route could finalize a result claiming a
        provenance that is no longer on disk. Publication now re-reads the anchor.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            generation.provenance_path.unlink()
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "cannot read the durable provenance"
            ):
                generation.publish_result(
                    **_publication_inputs(fixture, route, result)
                )
            self.assertFalse(generation.result_path.exists())
            self.assertFalse(generation.result_digest_path.exists())
            self.assertEqual(generation.result_publication_state(), "absent")

    def test_a_provenance_replaced_after_authentication_stops_publication(self) -> None:
        """Review reproducer: provenance swapped for one naming another study."""

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            swapped = json.loads(
                generation.provenance_path.read_text(encoding="utf-8")
            )
            swapped["exp1b_protocol_sha256"] = _digest("another-protocol")
            generation.provenance_path.unlink()
            generation.provenance_path.write_bytes(
                canonical_json_bytes(swapped) + b"\n"
            )
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "not the one this route authenticated"
            ):
                generation.publish_result(
                    **_publication_inputs(fixture, route, result)
                )
            self.assertFalse(generation.result_path.exists())
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "not the one this route authenticated"
            ):
                generation.complete_result_publication(
                    **_publication_inputs(fixture, route, result)
                )

    def test_the_anchor_is_rechecked_under_the_writer_lock(self) -> None:
        """The window between validation and installation is closed too.

        Validation releases the generation lock before the writer takes it, so a
        provenance replaced in between would otherwise reach a durable result.
        The window is driven deterministically here by removing the anchor the
        moment validation returns.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            klass = type(generation)
            original = klass._validated_result_bytes
            calls: list[str] = []

            def _swap(inner: Any, **kwargs: Any) -> Any:
                out = original(inner, **kwargs)
                calls.append("validated")
                # Exactly the window: audited, not yet installed.
                generation.provenance_path.unlink()
                return out

            klass._validated_result_bytes = _swap  # type: ignore[method-assign]
            try:
                with self.assertRaisesRegex(
                    Exp1bEvidenceError, "cannot read the durable provenance"
                ):
                    generation.publish_result(
                        **_publication_inputs(fixture, route, result)
                    )
            finally:
                klass._validated_result_bytes = original  # type: ignore[method-assign]
            self.assertEqual(calls, ["validated"])
            self.assertFalse(generation.result_path.exists())
            self.assertFalse(generation.result_digest_path.exists())

    def test_validation_itself_reads_the_anchor_off_disk(self) -> None:
        """Isolates the validation-stage read from the pre-install recheck.

        Both layers refuse a missing anchor, so a publication test cannot tell
        which one fired. This calls the validation stage on its own: with the
        anchor gone it must refuse there, before the writer's lock is ever taken
        and before the auditor is handed a document that is no longer on disk.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            inputs = _publication_inputs(fixture, route, result)
            # Sanity: it validates while the anchor is there.
            payload, digest = generation._validated_result_bytes(**inputs)
            self.assertTrue(payload and digest)
            generation.provenance_path.unlink()
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "cannot read the durable provenance"
            ):
                generation._validated_result_bytes(**inputs)

    def test_a_durable_payload_from_another_ordering_is_refused(self) -> None:
        """The per-payload half of the census binding, exercised on its own.

        The route's own reload already refuses a payload whose census differs, so
        this comparison is not reachable through a live route -- it is the layer
        that holds if that one is ever loosened. Driven directly here, because an
        unreachable check that no test can fail is indistinguishable from one that
        was deleted.
        """

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            genuine = route.reload_durable_state

            def _mismatched() -> Any:
                # No lock here: `_require_durable_agreement` calls this while
                # already holding `generation.exclusive()`, and flock does not
                # recurse within one process.
                state = dict(genuine())
                payloads = list(state["payloads"])
                payloads[3] = replace(
                    payloads[3],
                    census_ordering_sha256=_digest("another-census-ordering"),
                )
                state["payloads"] = tuple(payloads)
                return state

            route.reload_durable_state = _mismatched  # type: ignore[method-assign]
            try:
                with self.assertRaisesRegex(
                    Exp1bAggregateError, "different census ordering"
                ):
                    validated_exp1b_document(
                        result,
                        route=route,
                        protocol=fixture.protocol,
                        registry=fixture.registry,
                        amendment=fixture.amendment,
                        provenance=fixture.provenance,
                        admission=fixture.admission_document,
                    )
            finally:
                route.reload_durable_state = genuine  # type: ignore[method-assign]

    def test_completion_rechecks_the_anchor_under_the_writer_lock(self) -> None:
        """The same window, on the document-only recovery path."""

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            generation.publish_result(**_publication_inputs(fixture, route, result))
            generation.result_digest_path.unlink()
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )

            klass = type(generation)
            original = klass._validated_result_bytes

            def _swap(inner: Any, **kwargs: Any) -> Any:
                out = original(inner, **kwargs)
                generation.provenance_path.unlink()
                return out

            klass._validated_result_bytes = _swap  # type: ignore[method-assign]
            try:
                with self.assertRaisesRegex(
                    Exp1bEvidenceError, "cannot read the durable provenance"
                ):
                    generation.complete_result_publication(
                        **_publication_inputs(fixture, route, result)
                    )
            finally:
                klass._validated_result_bytes = original  # type: ignore[method-assign]
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )

    def test_document_only_recovery_runs_the_same_full_gate(self) -> None:
        """Completing a crashed journal is a publication, not a cheaper write."""

        import scripts.policy_improvement_exp1b_auditor as auditor

        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            generation.publish_result(**_publication_inputs(fixture, route, result))
            generation.result_digest_path.unlink()
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )

            original = auditor.audit_exp1b_result_document

            def _refusing(*args: Any, **kwargs: Any) -> Any:
                raise auditor.Exp1bAuditError("synthetic audit refusal")

            auditor.audit_exp1b_result_document = _refusing
            try:
                with self.assertRaisesRegex(
                    Exp1bEvidenceError, "synthetic audit refusal"
                ):
                    generation.complete_result_publication(
                        **_publication_inputs(fixture, route, result)
                    )
            finally:
                auditor.audit_exp1b_result_document = original
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )
            generation.complete_result_publication(
                **_publication_inputs(fixture, route, result)
            )
            self.assertEqual(generation.result_publication_state(), "complete")

    def test_document_only_recovery_refuses_a_different_durable_result(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, route, result = self._completed(tmp)
            generation = fixture.generation()
            document = self._publish(fixture, route, result)
            edited = document.replace(b'"seed_count":8', b'"seed_count":9')
            self.assertNotEqual(edited, document)
            generation.result_path.write_bytes(edited)
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "already carries another result"
            ):
                generation.complete_result_publication(
                    **_publication_inputs(fixture, route, result)
                )


class Exp1bCensusDerivationTest(unittest.TestCase):
    """Finding 1: the route builds its own census with the registered derivation."""

    def test_the_factory_no_longer_takes_a_state_id_callback(self) -> None:
        """The parameter is gone, so an alternate derivation cannot be supplied.

        It used to be accepted, and the digests the factory checks -- the
        ordered-record and population-binding digests -- do not cover the
        identifiers a callback produces. So an alternate derivation opened the
        route, served all eight requests, and spent the exact-once state machine
        before anything noticed.
        """

        import inspect

        parameters = inspect.signature(open_authenticated_exp1b_route).parameters
        self.assertNotIn("state_id_for", parameters)

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            producer, evaluator = _authorizations()
            with self.assertRaises(TypeError):
                open_authenticated_exp1b_route(
                    producer_authorization=producer,
                    evaluator_authorization=evaluator,
                    exp1b_protocol_path=fixture.protocol_path,
                    exp1b_registry_path=fixture.registry_path,
                    exp1b_amendment_path=fixture.amendment_path,
                    parent_protocol_path=fixture.parent_protocol_path,
                    parent_registry_path=fixture.parent_registry_path,
                    parent_population_path=fixture.parent_population_path,
                    parent_theory_amendment_path=fixture.parent_theory_path,
                    evidence_root=fixture.evidence_root,
                    generation_id=fixture.generation_id,
                    attestation=_theory_attestation(),
                    state_id_for=lambda index, digest: f"alternate-{index}-{digest}",
                )
            # Nothing durable was touched: no schedule, no payload, no claim.
            self.assertFalse(generation.schedule_path.exists())
            for position in range(UNITS):
                self.assertFalse(generation.has_payload(position))

    def test_a_nonregistered_census_is_refused_before_the_schedule(self) -> None:
        """The internal guard, driven directly.

        The factory now passes the registered derivation itself, so this check
        cannot fail through that path. It is what holds if the census is ever
        built another way, and it runs before the request schedule is
        constructed -- so the refusal costs no durable state.
        """

        import scripts.policy_improvement_exp1b_bridge as bridge

        population = _load(_V2 / "populations.json")["populations"][
            "validation_bridge"
        ]
        alternate = census_from_population(
            population, state_id_for=lambda index, digest: f"alternate-{index}"
        )
        self.assertNotEqual(
            alternate.ordering_sha256(), _canonical_census().ordering_sha256()
        )

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            self.assertFalse(generation.schedule_path.exists())
            genuine = bridge.census_from_population
            bridge.census_from_population = lambda *a, **k: alternate
            try:
                with self.assertRaisesRegex(
                    Exp1bBridgeError, "not the registered derivation"
                ):
                    fixture.open()
            finally:
                bridge.census_from_population = genuine
            self.assertFalse(generation.schedule_path.exists())
            for position in range(UNITS):
                self.assertFalse(generation.has_payload(position))
            # And the genuine derivation still opens it.
            self.assertEqual(fixture.open().access_state, "sealed_octet_complete")


class Exp1bWitnessMembershipTest(unittest.TestCase):
    """Finding 3: every witness must name a state the seed actually evaluated."""

    def test_the_two_witness_field_lists_agree(self) -> None:
        """The route gate and the auditor must not check different subsets.

        The auditor covered four of the six secondary witnesses and omitted both
        centering sub-witnesses. The list is duplicated rather than imported --
        the auditor imports no producer module -- so it is pinned here instead.
        """

        import scripts.policy_improvement_exp1b_auditor as auditor
        import scripts.policy_improvement_exp1b_bridge as bridge

        self.assertEqual(
            bridge.SECONDARY_WITNESS_FIELDS, auditor.SECONDARY_WITNESS_FIELDS
        )
        self.assertEqual(len(bridge.SECONDARY_WITNESS_FIELDS), 6)
        self.assertEqual(
            {record for record, _field in bridge.SECONDARY_WITNESS_FIELDS},
            set(SECONDARY_DIAGNOSTIC_NAMES) - {"persistent_state"},
        )

    def test_the_canonical_fixture_uses_real_census_members(self) -> None:
        members = {member.state_id for member in _canonical_census().members}
        secondary = _secondary_diagnostics()
        import scripts.policy_improvement_exp1b_bridge as bridge

        for record, field in bridge.SECONDARY_WITNESS_FIELDS:
            with self.subTest(witness=f"{record}.{field}"):
                self.assertIn(secondary[record][field], members)

    def test_the_route_refuses_each_nonmember_secondary_witness(self) -> None:
        """One negative case per witness field, through the real serve path."""

        import scripts.policy_improvement_exp1b_bridge as bridge

        for record, field in bridge.SECONDARY_WITNESS_FIELDS:
            with self.subTest(witness=f"{record}.{field}"), TemporaryDirectory() as tmp:
                fixture = _RouteFixture(Path(tmp))
                route = fixture.open()
                item = fixture.sealed[0]
                evaluator = _evaluator(0)
                evaluator["secondary_diagnostics"] = _secondary_diagnostics(
                    **{record: {field: "exp1b-state-" + ("c" * 64)}}
                )
                with self.assertRaisesRegex(
                    Exp1bBridgeError, "not a member of the registered"
                ):
                    route.serve(
                        seed_position=item["seed_position"],
                        seed=item["seed"],
                        run_id=item["run_id"],
                        checkpoint_sha256=item["checkpoint_sha256"],
                        evaluation_population="validation_bridge",
                        **evaluator,
                    )
                # Refused before anything durable: the slot is still unserved.
                self.assertEqual(route.served_positions, ())
                self.assertFalse(fixture.generation().has_payload(0))

    def test_the_auditor_refuses_each_nonmember_witness(self) -> None:
        """Both the three primary and the six secondary witnesses."""

        import scripts.policy_improvement_exp1b_bridge as bridge

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            result = aggregate_exp1b_result(route.sealed_octet())
            document = validated_exp1b_document(
                result,
                route=route,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                admission=fixture.admission_document,
            )
            outsider = "exp1b-state-" + ("d" * 64)

            def audited(edit: Any) -> None:
                moved = json.loads(document.decode("utf-8"))
                edit(moved["seed_rows"][0])
                audit_exp1b_result_document(
                    canonical_json_bytes(moved) + b"\n",
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    population_registry=_parent_populations(),
                    admission=fixture.admission_document,
                )

            for name in (
                "discrepancy_witness_state_id",
                "residual_n_witness_state_id",
                "residual_m_witness_state_id",
            ):
                with self.subTest(witness=name):
                    with self.assertRaisesRegex(
                        Exp1bAuditError, "not a member of the registered"
                    ):
                        audited(lambda row, name=name: row.__setitem__(name, outsider))
            for record, field in bridge.SECONDARY_WITNESS_FIELDS:
                with self.subTest(witness=f"{record}.{field}"):
                    with self.assertRaisesRegex(
                        Exp1bAuditError, "not a member of the registered"
                    ):
                        audited(
                            lambda row, record=record, field=field: row[
                                "secondary_diagnostics"
                            ][record].__setitem__(field, outsider)
                        )


class Exp1bPackagedAuditorTest(unittest.TestCase):
    """Finding 2: the shipped binary must be able to run the check it advertises."""

    def _published(self, tmp: Path) -> tuple[Any, bytes]:
        fixture = _RouteFixture(tmp)
        route = fixture.open()
        _serve_all(route, fixture)
        result = aggregate_exp1b_result(route.sealed_octet())
        generation = fixture.generation()
        generation.publish_result(
            result=result,
            route=route,
            protocol=fixture.protocol,
            registry=fixture.registry,
            amendment=fixture.amendment,
            admission=fixture.admission_document,
        )
        return fixture, generation.result_path.read_bytes()

    @staticmethod
    def _quiet(call: Any) -> int:
        """Run the CLI without its report landing in the suite's output."""

        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return call()

    def _argv(self, fixture: Any, result_path: Path, **overrides: str) -> list[str]:
        values = {
            "--result": str(result_path),
            "--reduced-study-protocol": str(fixture.protocol_path),
            "--reduced-study-registry": str(fixture.registry_path),
            "--reduced-study-amendment": str(fixture.amendment_path),
            "--substitute-provenance": str(fixture.provenance_path),
            "--parent-populations": str(fixture.parent_population_path),
            "--execution-admission": str(fixture.admission_path),
        }
        values.update(overrides)
        return [item for pair in values.items() for item in pair]

    def test_the_cli_requires_the_parent_population_registry(self) -> None:
        import scripts.policy_improvement_exp1b_auditor as auditor

        with TemporaryDirectory() as tmp:
            fixture, document = self._published(Path(tmp))
            path = Path(tmp) / "published.json"
            path.write_bytes(document)
            argv = self._argv(fixture, path)
            without = [
                item
                for index, item in enumerate(argv)
                if argv[index] != "--parent-populations"
                and (index == 0 or argv[index - 1] != "--parent-populations")
            ]
            with self.assertRaises(SystemExit):
                self._quiet(lambda: auditor.main(without))

    def test_the_cli_audits_a_genuine_result(self) -> None:
        import scripts.policy_improvement_exp1b_auditor as auditor

        with TemporaryDirectory() as tmp:
            fixture, document = self._published(Path(tmp))
            path = Path(tmp) / "published.json"
            path.write_bytes(document)
            self.assertEqual(
                self._quiet(lambda: auditor.main(self._argv(fixture, path))), 0
            )

    def test_the_cli_refuses_a_substituted_census_ordering(self) -> None:
        """Review reproducer: the CLI printed a successful audit and returned 0.

        Only ``census_ordering_sha256`` changes. Every other supported argument
        is supplied, including the real admission.
        """

        import scripts.policy_improvement_exp1b_auditor as auditor

        with TemporaryDirectory() as tmp:
            fixture, document = self._published(Path(tmp))
            moved = json.loads(document.decode("utf-8"))
            moved["census_ordering_sha256"] = _digest("another-census-ordering")
            path = Path(tmp) / "substituted.json"
            path.write_bytes(canonical_json_bytes(moved) + b"\n")
            self.assertEqual(
                self._quiet(lambda: auditor.main(self._argv(fixture, path))), 1
            )

    def test_the_audit_api_cannot_skip_the_census_rederivation(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture, document = self._published(Path(tmp))
            with self.assertRaises(TypeError):
                audit_exp1b_result_document(
                    document,
                    protocol=fixture.protocol,
                    registry=fixture.registry,
                    amendment=fixture.amendment,
                    provenance=fixture.provenance,
                    admission=fixture.admission_document,
                )

    def test_the_packaged_binary_ships_the_population_registry(self) -> None:
        """The argument is useless if the PAR does not carry the document."""

        import phase4_runtime_profile

        buck = (_ROOT / "BUCK").read_text(encoding="utf-8")
        start = buck.index('name = "policy_improvement_exp1b_auditor_bin"')
        target = buck[start : buck.index("python_library", start)]
        self.assertIn("configs/policy_improvement_v2/populations.json", target)
        self.assertIn(
            "configs/policy_improvement_v2/populations.json",
            phase4_runtime_profile.POLICY_IMPROVEMENT_EXP1B_AUDIT_PROFILE_PATHS,
        )



def _plain_secondary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {key: _plain_secondary(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_secondary(item) for item in value]  # type: ignore[return-value]
    return value


class Exp1bStageARetryTest(unittest.TestCase):
    """Finding 3: a retry of a published seed must not call the trainer."""

    class _RefusingBackend:
        """Fails loudly if anything Torch-bearing is constructed or run."""

        def __init__(self, marker: list[str]) -> None:
            self.marker = marker

        def prepare_exp1b_training(self, **kwargs: Any) -> Any:
            self.marker.append("prepare")
            raise AssertionError(
                "Stage A constructed a training session on a completed retry."
            )

        def run_exp1b_training(self, *, session: Any) -> Any:
            self.marker.append("train")
            raise AssertionError("Stage A retrained a completed seed.")

    def _publish_all(self, fixture: Any) -> None:
        """Run all eight seeds through the real production `main`."""

        harness = Exp1bStageAEndToEndTest()
        for offset in range(UNITS):
            exp1b_runtime.main(
                fixture.argv(seed_position=offset),
                backend=harness._backend(fixture, {}, seed_position=offset),
                runtime_attestation=fixture.attestation,
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )

    def test_a_retry_of_a_published_seed_never_trains(self) -> None:
        """Review reproducer: retrying seed 7 called run_exp1b_training again."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._publish_all(fixture)
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            self.assertTrue(generation.provenance_path.is_file())
            marker: list[str] = []
            code = exp1b_runtime.main(
                fixture.argv(seed_position=7),
                backend=self._RefusingBackend(marker),
                runtime_attestation=fixture.attestation,
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )
            self.assertEqual(code, 0)
            self.assertEqual(marker, [])

    def test_an_interrupted_finalization_completes_without_training(self) -> None:
        """All eight published, provenance removed: finalize, do not retrain."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._publish_all(fixture)
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            generation.provenance_path.unlink()
            marker: list[str] = []
            code = exp1b_runtime.main(
                fixture.argv(seed_position=7),
                backend=self._RefusingBackend(marker),
                runtime_attestation=fixture.attestation,
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )
            self.assertEqual(code, 0)
            self.assertEqual(marker, [])
            self.assertTrue(generation.provenance_path.is_file())

    def test_a_retry_producing_different_output_never_reaches_the_writer(self) -> None:
        """Review reproducer: a differing retry refused and left no provenance.

        Recovery must not depend on a 10,000-interaction run repeating bit for
        bit. The published pair is adopted on its own authenticated identity, so
        a backend that would produce different bytes is never consulted.
        """

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._publish_all(fixture)
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            generation.provenance_path.unlink()

            class _DifferentBackend(Exp1bStageARetryTest._RefusingBackend):
                pass

            marker: list[str] = []
            code = exp1b_runtime.main(
                fixture.argv(seed_position=3),
                backend=_DifferentBackend(marker),
                runtime_attestation=fixture.attestation,
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )
            self.assertEqual(code, 0)
            self.assertEqual(marker, [])
            self.assertTrue(generation.provenance_path.is_file())

    def test_an_unpublished_seed_still_trains(self) -> None:
        """Adoption must not become a way to skip work that never happened."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            marker: list[str] = []
            with self.assertRaises(AssertionError):
                exp1b_runtime.main(
                    fixture.argv(seed_position=0),
                    backend=self._RefusingBackend(marker),
                    runtime_attestation=fixture.attestation,
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )
            self.assertEqual(marker, ["prepare"])

    def _crashed_between_the_pair(self, fixture: Any, position: int) -> Any:
        """The reachable ``checkpoint_only`` state, exactly as a crash leaves it.

        A crash between a seed's checkpoint and its manifest happens *during* that
        seed's publication, so the generation cannot yet be finalized: provenance
        is written only once all eight pairs are complete. Removing the manifest
        from a finalized generation, as the review's probe did, is a state a crash
        cannot produce -- and there the checkpoint is referenced by provenance,
        which is the separate refusal below.
        """

        self._publish_all(fixture)
        generation = open_evidence_generation(
            evidence_root=fixture.evidence_root,
            generation_id=fixture.generation_id,
        )
        generation.provenance_path.unlink()
        (
            generation.checkpoint_directory_for(position) / "run_manifest.json"
        ).unlink()
        self.assertEqual(
            generation.checkpoint_publication_state(position), "checkpoint_only"
        )
        return generation

    def test_a_checkpoint_only_seed_still_trains(self) -> None:
        """A crash between the pair's two files is not a published seed.

        The manifest carries the counters and identities only the run itself can
        produce, so there is nothing to adopt and the seed must run again.
        """

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._crashed_between_the_pair(fixture, 4)
            marker: list[str] = []
            with self.assertRaises(AssertionError):
                exp1b_runtime.main(
                    fixture.argv(seed_position=4),
                    backend=self._RefusingBackend(marker),
                    runtime_attestation=fixture.attestation,
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )
            self.assertEqual(marker, ["prepare"])

    def test_a_checkpoint_only_seed_recovers_with_differing_retry_bytes(self) -> None:
        """Review reproducer: prepare and training ran, then `cannot be adopted`.

        The rerun is a fresh 10,000-interaction run, so its serialization differs
        from the checkpoint the crash left behind and ``_install_or_adopt``
        refused it -- after the full run. The leftover is now retired under the
        generation lock first, while it is provably referenced by nothing, and the
        rerun installs one new complete pair.
        """

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            generation = self._crashed_between_the_pair(fixture, 4)
            directory = generation.checkpoint_directory_for(4)
            stranded = (directory / "checkpoint.pt").read_bytes()
            stranded_sha256 = hashlib.sha256(stranded).hexdigest()

            recorder: dict[str, Any] = {}
            code = exp1b_runtime.main(
                fixture.argv(seed_position=4),
                backend=Exp1bStageAEndToEndTest()._backend(
                    fixture, recorder, seed_position=4, variant="-retry"
                ),
                runtime_attestation=fixture.attestation,
                environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
            )
            self.assertEqual(code, 0)
            # It really did retrain, with different output.
            self.assertIn("session", recorder)
            republished = (directory / "checkpoint.pt").read_bytes()
            self.assertNotEqual(republished, stranded)

            # One new complete pair, and the manifest names the new bytes.
            self.assertEqual(generation.checkpoint_publication_state(4), "complete")
            manifest = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["checkpoint_sha256"],
                hashlib.sha256(republished).hexdigest(),
            )
            self.assertEqual(manifest["seed_position"], 4)

            # Provenance completes over the new octet.
            self.assertTrue(generation.provenance_path.is_file())
            provenance = json.loads(
                generation.provenance_path.read_text(encoding="utf-8")
            )
            sealed = provenance["sealed_checkpoints"][4]
            self.assertEqual(
                sealed["checkpoint_sha256"], hashlib.sha256(republished).hexdigest()
            )

            # The retired checkpoint is kept, content-named, outside the pair.
            retired = directory / "retired" / f"checkpoint-{stranded_sha256}.pt"
            self.assertEqual(retired.read_bytes(), stranded)
            record = json.loads(
                (directory / "retired" / f"checkpoint-{stranded_sha256}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["checkpoint_sha256"], stranded_sha256)
            self.assertEqual(record["seed_position"], 4)
            self.assertEqual(
                record["reason"], "crash_between_checkpoint_and_run_manifest"
            )

            # And all eight checkpoints still authenticate against the new
            # provenance: the `retired/` subdirectory is invisible to the pair.
            sealed = authenticate_sealed_octet_checkpoints(
                generation=generation,
                provenance=provenance,
                registry_rows=registry_rows_by_run_id(fixture.registry),
                effective_config_sha256=manifest["effective_config_sha256"],
                train_ordered_record_sha256=(
                    manifest["train_split_ordered_record_sha256"]
                ),
            )
            self.assertEqual(len(sealed), UNITS)
            self.assertEqual(
                sealed[4].checkpoint_sha256,
                hashlib.sha256(republished).hexdigest(),
            )

    def test_a_missing_manifest_in_a_finalized_generation_is_refused(self) -> None:
        """Provenance names that checkpoint, so it is not ours to retire.

        This is the review probe's actual state: a manifest deleted from an
        already-finalized generation. A crash cannot produce it, the checkpoint is
        referenced, and the sealed generation stays immutable. Refusing before the
        backend is built beats refusing after a 10,000-interaction rerun.
        """

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._publish_all(fixture)
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            self.assertTrue(generation.provenance_path.is_file())
            (
                generation.checkpoint_directory_for(4) / "run_manifest.json"
            ).unlink()
            marker: list[str] = []
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "finalized provenance"
            ):
                exp1b_runtime.main(
                    fixture.argv(seed_position=4),
                    backend=self._RefusingBackend(marker),
                    runtime_attestation=fixture.attestation,
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )
            self.assertEqual(marker, [])
            self.assertTrue(
                (
                    generation.checkpoint_directory_for(4) / "checkpoint.pt"
                ).is_file()
            )

    def test_retirement_never_touches_a_complete_pair_or_an_absent_one(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            generation = self._crashed_between_the_pair(fixture, 4)
            # Seed 6 is a complete pair: immutable, reported, untouched.
            before = (
                generation.checkpoint_directory_for(6) / "checkpoint.pt"
            ).read_bytes()
            self.assertEqual(
                generation.retire_unreferenced_checkpoint(6), "complete"
            )
            self.assertEqual(
                (generation.checkpoint_directory_for(6) / "checkpoint.pt").read_bytes(),
                before,
            )
            self.assertFalse(
                (generation.checkpoint_directory_for(6) / "retired").exists()
            )
            # Seed 4 retires once, then reports `absent`.
            self.assertEqual(
                generation.retire_unreferenced_checkpoint(4), "retired"
            )
            self.assertEqual(
                generation.checkpoint_publication_state(4), "absent"
            )
            self.assertEqual(generation.retire_unreferenced_checkpoint(4), "absent")

    def test_retirement_refuses_a_checkpoint_a_payload_could_name(self) -> None:
        """A sealed Stage B payload means the manifest existed; do not retire."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            generation = fixture.generation()
            (
                generation.checkpoint_directory_for(2) / "run_manifest.json"
            ).unlink()
            self.assertEqual(
                generation.checkpoint_publication_state(2), "checkpoint_only"
            )
            with self.assertRaisesRegex(
                Exp1bEvidenceError, "not a reachable crash state"
            ):
                generation.retire_unreferenced_checkpoint(2)
            self.assertTrue(
                (
                    generation.checkpoint_directory_for(2) / "checkpoint.pt"
                ).is_file()
            )

    def test_a_pair_published_under_other_inputs_is_refused(self) -> None:
        """Adoption authenticates; it does not merely observe two files."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._publish_all(fixture)
            generation = open_evidence_generation(
                evidence_root=fixture.evidence_root,
                generation_id=fixture.generation_id,
            )
            path = generation.checkpoint_directory_for(5) / "run_manifest.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["admission_sha256"] = _digest("another-admission")
            path.unlink()
            path.write_bytes(canonical_json_bytes(document) + b"\n")
            marker: list[str] = []
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "different admission_sha256"
            ):
                exp1b_runtime.main(
                    fixture.argv(seed_position=5),
                    backend=self._RefusingBackend(marker),
                    runtime_attestation=fixture.attestation,
                    environment={EXECUTION_ENVIRONMENT_VARIABLE: "1"},
                )
            self.assertEqual(marker, [])

    def test_adoption_runs_after_authentication_not_before(self) -> None:
        """An unauthenticated process must not be able to skip work."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "producer"
            fixture = _StageAFixture(root)
            self._publish_all(fixture)
            marker: list[str] = []
            with self.assertRaisesRegex(
                exp1b_runtime.Exp1bRuntimeError, "execution requires"
            ):
                exp1b_runtime.main(
                    fixture.argv(seed_position=7),
                    backend=self._RefusingBackend(marker),
                    runtime_attestation=fixture.attestation,
                    environment={},
                )
            self.assertEqual(marker, [])


def _sever_link(path: Path) -> Path:
    """Recreate the exact state a crash between `os.link` and the temp unlink leaves.

    The final artifact is complete but at ``st_nlink == 2``, which
    ``stable_bytes`` refuses outright.
    """

    from scripts.policy_improvement_exp1b_evidence import _TEMP_PREFIX

    orphan = path.with_name(f"{_TEMP_PREFIX}{path.name}.99999")
    os.link(path, orphan)
    assert path.stat().st_nlink == 2
    return orphan


class Exp1bRestartRecoveryTest(unittest.TestCase):
    """Finding 2: initial-write and link/unlink crash states must recover."""

    # --- initial schedule creation -----------------------------------------

    def test_a_zero_byte_initial_schedule_is_repaired(self) -> None:
        """Review reproducer: reopened with 'not JSON', permanently.

        The old `_create` opened the FINAL pathname and did one unchecked write,
        so a crash there left a name that every restart tried and failed to
        parse. The staged install means the name only ever appears complete; a
        zero-byte file at that path is therefore not something this module wrote,
        and the recovery is to reap and recreate.
        """

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()
            state = fixture.schedule_state_path
            state.unlink()
            state.write_bytes(b"")
            with self.assertRaises(Exp1bScheduleStateError):
                fixture.open()
            # The operator repair is removing the empty file; the next open
            # recreates it, which the old code could not do either.
            state.unlink()
            reopened = fixture.open()
            self.assertEqual(reopened.access_state, "sealed_octet_complete")
            self.assertGreater(state.stat().st_size, 0)

    def test_a_short_initial_schedule_never_appears(self) -> None:
        """The staged install publishes the name only with all its bytes."""

        source = (
            _ROOT / "scripts" / "policy_improvement_exp1b_schema.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        create = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_create"
        )
        body = ast.get_source_segment(source, create)
        assert body is not None
        self.assertIn("install_durable_artifact(", body)
        # No raw write onto the final pathname survives.
        self.assertNotIn("os.O_EXCL", body)
        self.assertNotIn("os.write(", body)

    def test_the_initial_schedule_is_installed_atomically(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            fixture.open()
            state = fixture.schedule_state_path
            self.assertEqual(state.stat().st_nlink, 1)
            document = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(len(document["slots"]), UNITS)

    # --- link/unlink window, per artifact ----------------------------------

    def test_a_severed_payload_link_recovers(self) -> None:
        """Review reproducer: 'is a link alias'; nlink stayed 2."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            route.serve(
                seed_position=0,
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            payload_path = fixture.generation().payload_path(0)
            _sever_link(payload_path)
            # The reader reaps first, so the state is transient.
            reopened = fixture.open()
            self.assertEqual(reopened.served_positions, (0,))
            self.assertEqual(payload_path.stat().st_nlink, 1)

    def test_a_severed_orphan_payload_link_recovers(self) -> None:
        """Orphan reconciliation reads first, so it must reap first."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            item = fixture.sealed[0]
            route.serve(
                seed_position=0,
                seed=item["seed"],
                run_id=item["run_id"],
                checkpoint_sha256=item["checkpoint_sha256"],
                evaluation_population="validation_bridge",
                **_evaluator(0),
            )
            # Roll the marker back to make it an orphan, and sever the link.
            document = json.loads(
                fixture.schedule_state_path.read_text(encoding="utf-8")
            )
            document["slots"][0]["state"] = "pending"
            document["slots"][0]["payload_sha256"] = None
            fixture.schedule_state_path.write_bytes(
                canonical_json_bytes(document) + b"\n"
            )
            payload_path = fixture.generation().payload_path(0)
            _sever_link(payload_path)
            reopened = fixture.open()
            self.assertEqual(reopened.served_positions, (0,))
            self.assertEqual(payload_path.stat().st_nlink, 1)

    def test_a_severed_result_document_link_recovers(self) -> None:
        """Review reproducer: complete_result_publication refused, no sidecar."""

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            result = aggregate_exp1b_result(route.sealed_octet())
            document = validated_exp1b_document(
                result,
                route=route,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                admission=fixture.admission_document,
            )
            generation = fixture.generation()
            generation.publish_result(**_publication_inputs(fixture, route, result))
            generation.result_digest_path.unlink()
            _sever_link(generation.result_path)
            # The recovery entry point reaps under the lock; the reader itself
            # deliberately does not, so an unlocked reader cannot delete a
            # concurrent writer's staging temporary.
            generation.reap_under_lock()
            self.assertEqual(
                generation.result_publication_state(), "document_only"
            )
            self.assertEqual(generation.result_path.stat().st_nlink, 1)
            generation.complete_result_publication(
                **_publication_inputs(fixture, route, result)
            )
            self.assertEqual(generation.result_publication_state(), "complete")

    def test_a_severed_result_sidecar_link_recovers(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            route = fixture.open()
            _serve_all(route, fixture)
            result = aggregate_exp1b_result(route.sealed_octet())
            document = validated_exp1b_document(
                result,
                route=route,
                protocol=fixture.protocol,
                registry=fixture.registry,
                amendment=fixture.amendment,
                provenance=fixture.provenance,
                admission=fixture.admission_document,
            )
            generation = fixture.generation()
            generation.publish_result(**_publication_inputs(fixture, route, result))
            _sever_link(generation.result_digest_path)
            generation.reap_under_lock()
            reloaded, digest = generation.read_published_result()
            self.assertEqual(reloaded, document)
            self.assertEqual(digest, result.document_sha256())
            self.assertEqual(generation.result_digest_path.stat().st_nlink, 1)

    def test_a_severed_checkpoint_link_recovers(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            checkpoint = (
                generation.checkpoint_directory_for(3) / "checkpoint.pt"
            )
            _sever_link(checkpoint)
            generation.reap_under_lock()
            self.assertEqual(
                generation.checkpoint_publication_state(3), "complete"
            )
            self.assertEqual(checkpoint.stat().st_nlink, 1)
            # And the route still opens over the whole octet.
            self.assertEqual(fixture.open().access_state, "sealed_octet_complete")

    def test_a_severed_manifest_link_recovers(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            manifest = (
                generation.checkpoint_directory_for(6) / "run_manifest.json"
            )
            _sever_link(manifest)
            generation.reap_under_lock()
            self.assertEqual(
                generation.checkpoint_publication_state(6), "complete"
            )
            self.assertEqual(fixture.open().access_state, "sealed_octet_complete")

    def test_a_severed_provenance_link_recovers(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            _sever_link(generation.provenance_path)
            # Route opening reads provenance through the generation, which reaps
            # when the schedule is constructed.
            self.assertEqual(fixture.open().access_state, "sealed_octet_complete")
            self.assertEqual(generation.provenance_path.stat().st_nlink, 1)

    def test_readers_do_not_reap_and_recovery_callers_do(self) -> None:
        """The reap belongs to lock holders, not to readers.

        Reaping inside an unlocked reader closes the crash window and opens a
        worse one: the reader deletes a lock-holding writer's staging temporary
        in another process and the write is lost. Adversarial verification
        reproduced exactly that, so the reap moved to `reap_under_lock`, which
        every recovery entry point calls before it reads.
        """

        evidence = (
            _ROOT / "scripts" / "policy_improvement_exp1b_evidence.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(evidence)
        generation = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "Exp1bEvidenceGeneration"
        )
        for name in (
            "read_payload",
            "has_payload",
            "checkpoint_publication_state",
            "result_publication_state",
            "read_published_result",
        ):
            with self.subTest(reader=name):
                method = next(
                    node
                    for node in generation.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                )
                body = ast.get_source_segment(evidence, method)
                assert body is not None
                self.assertNotIn("_reap_partials(", body)
        # And the reaper itself takes the lock.
        reaper = next(
            node
            for node in generation.body
            if isinstance(node, ast.FunctionDef) and node.name == "reap_under_lock"
        )
        reaper_body = ast.get_source_segment(evidence, reaper)
        assert reaper_body is not None
        self.assertIn("with self.exclusive():", reaper_body)

        # Every production recovery entry point calls it.
        runtime = (
            _ROOT / "scripts" / "policy_improvement_exp1b_runtime.py"
        ).read_text(encoding="utf-8")
        runtime_tree = ast.parse(runtime)
        for name in (
            "_authenticated_publication",
            "_finalize_when_complete",
            "_publish_exp1b_result",
        ):
            with self.subTest(caller=name):
                function = next(
                    node
                    for node in runtime_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                )
                body = ast.get_source_segment(runtime, function)
                assert body is not None
                self.assertIn("reap_under_lock()", body)
        bridge = (
            _ROOT / "scripts" / "policy_improvement_exp1b_bridge.py"
        ).read_text(encoding="utf-8")
        self.assertIn("generation.reap_under_lock()", bridge)

    def test_an_unlocked_reader_cannot_delete_a_writers_staging_file(self) -> None:
        """The regression the reap-in-reader version introduced."""

        from scripts.policy_improvement_exp1b_evidence import _TEMP_PREFIX

        with TemporaryDirectory() as tmp:
            fixture = _RouteFixture(Path(tmp))
            generation = fixture.generation()
            # A peer's in-flight staging temporary.
            staging = generation.directory / f"{_TEMP_PREFIX}provenance.json.4242"
            staging.write_bytes(b"a writer is mid-install")
            # Readers must leave it alone.
            generation.result_publication_state()
            generation.checkpoint_publication_state(0)
            generation.has_payload(0)
            self.assertTrue(staging.is_file())
            # Only the locked reaper clears it, and a writer holds that lock for
            # the whole install.
            generation.reap_under_lock()
            self.assertFalse(staging.is_file())


if __name__ == "__main__":
    unittest.main()
