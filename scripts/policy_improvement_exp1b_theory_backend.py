#!/usr/bin/env fbpython
"""Experiment 1B Stage B production backend.

The theory-bridge PAR reaches this module only on the launcher-owned Experiment
1B branch. It is deliberately separate from
``scripts/policy_improvement_theory_backend_v2.py``: the v2 backend's standing
``validation_bridge`` refusal is untouched, and nothing here can be reached from
a v2 theory request.

What it provides
----------------
The four things ``scripts/policy_improvement_exp1b_runtime.bridge_main`` calls:

``exp1b_state_id_for``
    Re-exported from the schema module, where it lives so the Stage A runtime can
    bind the census ordering without importing this Torch-restoring module.
``prepare_exp1b_bridge``
    Authenticates one sealed checkpoint out of the evidence generation, restores
    it read-only, and returns the four evaluator callables the route drives over
    the census.
``publish_exp1b_result``
    Records that the durable result the runtime already wrote is the one this
    backend's runtime produced. Publication itself is the evidence module's job;
    this is the backend-side acknowledgement and identity check.

Torch is imported lazily
------------------------
Only :meth:`Exp1bTheoryBackend.prepare_exp1b_bridge` needs Torch, and it imports
the full backend at call time. Everything else in this module -- the state-ID
derivation, the identity checks, the refusals -- runs in a Torch-free
environment, which is where the test suite exercises it.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.policy_improvement_exp1b_evidence import (
    AuthenticatedCheckpoint,
    Exp1bEvidenceError,
)
from scripts.policy_improvement_exp1b_schema import (
    DEPLOYED_DEPTH_N,
    exp1b_state_id_for,
    STATE_ID_NAMESPACE,
    EVALUATION_POPULATION_ID,
    REFERENCE_DEPTH_M,
    REGISTERED_SEEDS,
    UNITS,
)

__all__ = [
    "Exp1bTheoryBackendError",
    "Exp1bTheoryBackend",
    "Exp1bSeedEvaluator",
    "create_exp1b_theory_backend",
    "exp1b_state_id_for",
    "STATE_ID_NAMESPACE",
]


class Exp1bTheoryBackendError(RuntimeError):
    """Raised when the Stage B backend is not authenticated as required."""


@dataclass(frozen=True)
class Exp1bSeedEvaluator:
    """One restored sealed checkpoint, exposed as four read-only callables."""

    seed_position: int
    seed: int
    run_id: str
    checkpoint_sha256: str
    endpoint_values: Any
    action_values: Any
    action_mask: Any
    #: The **frozen base** law: the Bellman operator's policy. Never the mixture.
    base_probabilities: Any
    #: The five required secondary checks. A **callable**, not a materialized
    #: mapping: computing them traverses all 128 census members, which is
    #: evaluator work and must happen inside the durable request claim rather
    #: than during backend preparation.
    secondary_diagnostics: Any


class Exp1bTheoryBackend:
    """Stage B backend bound to one authenticated PAR, authorization, and octet."""

    def __init__(
        self,
        *,
        runtime_attestation: Mapping[str, str],
        authenticated_checkpoints: Sequence[AuthenticatedCheckpoint],
        effective_config_sha256: str,
        census_ordering_sha256: str,
        census: Sequence[Any],
        dataset_root: Any,
        evaluation_split: str,
        evaluation_split_manifest_sha256: str,
        full_backend_module: Any | None = None,
    ) -> None:
        required = {
            "source_git_commit",
            "runtime_sha256",
            "launcher_sha256",
            "runtime_authorization_sha256",
        }
        if not isinstance(runtime_attestation, Mapping) or set(
            runtime_attestation
        ) != required:
            raise Exp1bTheoryBackendError(
                "Stage B backend requires the packaged runtime attestation."
            )
        checkpoints = tuple(authenticated_checkpoints)
        if len(checkpoints) != UNITS or any(
            not isinstance(item, AuthenticatedCheckpoint) for item in checkpoints
        ):
            raise Exp1bTheoryBackendError(
                "Stage B backend requires eight authenticated checkpoints."
            )
        for offset, item in enumerate(checkpoints):
            if item.seed_position != offset or item.seed != REGISTERED_SEEDS[offset]:
                raise Exp1bTheoryBackendError(
                    "Stage B checkpoints are not the registered seeds in order."
                )
            if item.effective_config_sha256 != effective_config_sha256:
                raise Exp1bTheoryBackendError(
                    "Stage B checkpoint was produced under another configuration."
                )
        self._attestation = {str(k): str(v) for k, v in runtime_attestation.items()}
        self._checkpoints = checkpoints
        self._effective_config_sha256 = str(effective_config_sha256)
        self._census_ordering_sha256 = str(census_ordering_sha256)
        self._census = tuple(census)
        if not self._census:
            raise Exp1bTheoryBackendError("Stage B backend requires the census.")
        self._dataset_root = dataset_root
        self._evaluation_split = str(evaluation_split)
        self._evaluation_split_manifest_sha256 = str(
            evaluation_split_manifest_sha256
        )
        self._full_backend = full_backend_module
        self._sessions: dict[int, Any] = {}
        self._published = False

    # -- route inputs --------------------------------------------------------

    @staticmethod
    def exp1b_state_id_for(record_index: int, dataset_record_sha256: str) -> str:
        return exp1b_state_id_for(record_index, dataset_record_sha256)

    def prepare_exp1b_bridge(
        self,
        *,
        seed_position: int,
        runtime_attestation: Any,
        effective_config_sha256: str,
    ) -> Exp1bSeedEvaluator:
        """Restore one sealed checkpoint read-only and return its evaluators."""

        if (
            isinstance(seed_position, bool)
            or not isinstance(seed_position, int)
            or not 0 <= seed_position < UNITS
        ):
            raise Exp1bTheoryBackendError(
                "Stage B seed position is outside the registered octet."
            )
        for name in (
            "source_git_commit",
            "runtime_sha256",
            "launcher_sha256",
            "runtime_authorization_sha256",
        ):
            if getattr(runtime_attestation, name, None) != self._attestation[name]:
                raise Exp1bTheoryBackendError(
                    f"Stage B evaluation attestation {name} differs from the runtime."
                )
        if effective_config_sha256 != self._effective_config_sha256:
            raise Exp1bTheoryBackendError(
                "Stage B evaluation requested another effective configuration."
            )
        checkpoint = self._checkpoints[seed_position]

        module = self._full_backend
        if module is None:
            module = importlib.import_module("policy_improvement_full_backend")
        opener = getattr(module, "open_exp1b_sealed_evaluation_session", None)
        if not callable(opener):
            raise Exp1bTheoryBackendError(
                "Full backend does not expose open_exp1b_sealed_evaluation_session; "
                "Experiment 1B Stage B cannot restore a sealed checkpoint."
            )
        census_type = getattr(module, "Exp1bCensusMember", None)
        if census_type is None:
            raise Exp1bTheoryBackendError(
                "Full backend does not expose Exp1bCensusMember."
            )
        members = tuple(
            census_type(
                state_id=item.state_id,
                record_index=item.record_index,
                dataset_record_sha256=item.dataset_record_sha256,
            )
            for item in self._census
        )
        training_module = importlib.import_module("upi_trm_train")
        try:
            session = opener(
                checkpoint_path=Path(checkpoint.checkpoint_path),
                expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                expected_checkpoint_size_bytes=checkpoint.checkpoint_size_bytes,
                expected_model_state_sha256=checkpoint.model_state_sha256,
                expected_effective_config_sha256=checkpoint.effective_config_sha256,
                expected_environment_interactions=(checkpoint.environment_interactions),
                run_id=checkpoint.run_id,
                seed=checkpoint.seed,
                seed_position=checkpoint.seed_position,
                census_ordering_sha256=self._census_ordering_sha256,
                census=members,
                depths=(DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M),
                evaluation_population=EVALUATION_POPULATION_ID,
                dataset_root=self._dataset_root,
                evaluation_split=self._evaluation_split,
                evaluation_split_manifest_sha256=(
                    self._evaluation_split_manifest_sha256
                ),
                training_module=training_module,
            )
        except Exp1bEvidenceError as exc:
            raise Exp1bTheoryBackendError(str(exc)) from exc
        except Exception as exc:  # the opener raises FullBackendError subclasses
            raise Exp1bTheoryBackendError(
                f"Stage B sealed restore refused: {exc}"
            ) from exc
        self._sessions[seed_position] = session
        for name in (
            "endpoint_values",
            "action_values",
            "action_mask",
            "base_probabilities",
            "deployed_probabilities",
            "candidate_probabilities",
            "secondary_diagnostics",
        ):
            if not callable(getattr(session, name, None)):
                raise Exp1bTheoryBackendError(
                    f"Stage B evaluation session exposes no {name} callable."
                )
        if session.checkpoint_sha256 != checkpoint.checkpoint_sha256:
            raise Exp1bTheoryBackendError(
                "Stage B evaluation session restored another checkpoint."
            )
        return Exp1bSeedEvaluator(
            seed_position=checkpoint.seed_position,
            seed=checkpoint.seed,
            run_id=checkpoint.run_id,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            endpoint_values=session.endpoint_values,
            action_values=session.action_values,
            action_mask=session.action_mask,
            base_probabilities=session.base_probabilities,
            secondary_diagnostics=session.secondary_diagnostics,
        )

    # -- publication acknowledgement ----------------------------------------

    def publish_exp1b_result(
        self,
        *,
        protocol_id: str,
        result_path: Path,
        document_sha256: str,
        runtime_attestation: Any,
    ) -> None:
        """Acknowledge one durable publication, exactly once per backend.

        The bytes are already on disk and already read back by the runtime. What
        this adds is a same-process second check that the file the runtime named
        is the one it says it is, and a hard refusal of a second publication --
        a retry after emission is permanent, not idempotent.
        """

        if self._published:
            raise Exp1bTheoryBackendError(
                "Experiment 1B result was already published by this runtime; a "
                "post-emission retry is permanently refused."
            )
        if getattr(runtime_attestation, "runtime_authorization_sha256", None) != (
            self._attestation["runtime_authorization_sha256"]
        ):
            raise Exp1bTheoryBackendError(
                "Publication attestation differs from the sealed runtime."
            )
        path = Path(result_path)
        if not path.is_absolute():
            raise Exp1bTheoryBackendError("Published result path is not absolute.")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise Exp1bTheoryBackendError("Published result cannot be read.") from exc
        body = payload[:-1] if payload.endswith(b"\n") else payload
        from scripts.policy_improvement_exp1b_schema import exp1b_document_sha256
        from scripts.policy_improvement_schema import (
            load_strict_json_bytes,
            PolicyImprovementSchemaError,
        )

        try:
            parsed = load_strict_json_bytes(body)
        except PolicyImprovementSchemaError as exc:
            raise Exp1bTheoryBackendError(
                "Published result is not strict JSON."
            ) from exc
        if not isinstance(parsed, Mapping) or parsed.get("protocol_id") != protocol_id:
            raise Exp1bTheoryBackendError(
                "Published result is not the registered Experiment 1B protocol."
            )
        if exp1b_document_sha256(parsed) != document_sha256:
            raise Exp1bTheoryBackendError(
                "Published result digest differs from the document on disk."
            )
        self._published = True


def create_exp1b_theory_backend(
    *,
    runtime_attestation: Mapping[str, str],
    authenticated_checkpoints: Sequence[AuthenticatedCheckpoint],
    effective_config_sha256: str,
    census_ordering_sha256: str,
    census: Sequence[Any],
    dataset_root: Any,
    evaluation_split: str,
    evaluation_split_manifest_sha256: str,
) -> Exp1bTheoryBackend:
    """The single Stage B backend factory."""

    return Exp1bTheoryBackend(
        runtime_attestation=runtime_attestation,
        authenticated_checkpoints=authenticated_checkpoints,
        effective_config_sha256=effective_config_sha256,
        census_ordering_sha256=census_ordering_sha256,
        census=census,
        dataset_root=dataset_root,
        evaluation_split=evaluation_split,
        evaluation_split_manifest_sha256=evaluation_split_manifest_sha256,
    )
