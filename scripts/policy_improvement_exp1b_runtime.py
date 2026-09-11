#!/usr/bin/env fbpython
"""Experiment 1B authenticated runtime handlers.

Two entry points, both reached only through the packaged launcher:

* :func:`main` -- the Stage A train-only handler, imported by
  ``policy_improvement_full_entrypoint`` when the launcher owns and injects
  ``--policy-improvement-exp1b-entrypoint``.
* :func:`bridge_main` -- the Stage B handler, imported by
  ``policy_improvement_theory_bridge_entrypoint`` when the launcher owns and
  injects ``--policy-improvement-exp1b-bridge-entrypoint``.

Both refuse to run unless invoked through that path: each requires an
authenticated backend factory and a runtime attestation that only the entrypoint
can supply from its preflight. Calling either directly, or with the
authorization arguments merely *present* rather than *consumed*, fails closed.

The four gates, in both stages
------------------------------
Neither stage does any work until all four of these pass, in this order:

1. **Document transition.** Protocol, registry, and reduced-study amendment are
   read as stable files, validated, and cross-bound by canonical digest. The
   owner-signed *admission* amendment is then validated against those exact
   digests. It is the document that moves ``base_policy_artifact`` and
   ``runtime_authorization`` from unavailable to available; without it, both
   stages stop here.
2. **Purpose and authorization.** The admission names one purpose,
   ``policy_improvement_exp1b_reduced_study``, and one authorized runtime. The
   live attestation -- commit, runtime digest, launcher digest, authorization
   digest -- has to be the one it admitted.
3. **Execution environment.** ``RUN_UPITRM_FULL_EXPERIMENTS=1``, read from the
   real process environment. Experiment 1B neither widens this gate nor
   supplies a fallback.
4. **Producer root.** Every artifact path is resolved under the one producer
   root the launcher passed, and the evidence generation is separately proved to
   be owner-controlled.

Torch stays behind the seam
---------------------------
The Torch-bearing model, restoration, trainer construction, and checkpoint
serialisation come from ``policy_improvement_full_backend`` through the
``backend`` argument. This module imports nothing from Torch, so its argument
grammar, gating, and ordering contract are exercised by the test suite in a
Torch-free environment. The real backend supplies the real callables; the
contract is the same object either way.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.policy_improvement_exp1b_aggregate import (
    aggregate_exp1b_result,
    Exp1bAggregateError,
)
from scripts.policy_improvement_exp1b_bridge import (
    Exp1bBridgeError,
    Exp1bRuntimeAttestation,
    open_authenticated_exp1b_route,
)
from scripts.policy_improvement_exp1b_evidence import (
    Exp1bEvidenceError,
    finalize_exp1b_evidence,
    open_evidence_generation,
    stable_bytes,
)
from scripts.policy_improvement_exp1b_schema import (
    ADMISSION_ROLES,
    apply_exp1b_admission,
    FULL_ROLE,
    role_attestation_document,
    THEORY_BRIDGE_ROLE,
    EVALUATION_POPULATION_ID,
    Exp1bExecutionAdmission,
    Exp1bSchemaError,
    exp1b_document_sha256,
    EXECUTION_PURPOSE,
    PROTOCOL_ID,
    registry_rows_by_run_id,
    TERMINAL_ENVIRONMENT_INTERACTIONS,
    TRAINING_RECORD_COUNT,
    UNITS,
    validate_exp1b_admission,
    validate_exp1b_amendment,
    validate_exp1b_protocol,
    validate_exp1b_registry,
)
from scripts.policy_improvement_exp1b_session import (
    build_exp1b_training_session,
    Exp1bSessionError,
    register_exp1b_training,
    TrainOnlyDatasetGuard,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
)

__all__ = [
    "Exp1bRuntimeError",
    "EXP1B_BRIDGE_ENTRYPOINT_MARKER",
    "EXP1B_TRAINING_ENTRYPOINT_MARKER",
    "Exp1bAdmittedStudy",
    "admit_exp1b_study",
    "bridge_main",
    "main",
]

EXP1B_TRAINING_ENTRYPOINT_MARKER = "--policy-improvement-exp1b-entrypoint"
EXP1B_BRIDGE_ENTRYPOINT_MARKER = "--policy-improvement-exp1b-bridge-entrypoint"

RUN_MANIFEST_SCHEMA_NAME = "policy_improvement_exp1b_run_manifest_v1"
RUN_MANIFEST_SCHEMA_VERSION = 1


class Exp1bRuntimeError(RuntimeError):
    """Raised when the Experiment 1B runtime is not authenticated as required."""


def _absolute_file(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise Exp1bRuntimeError(f"{label} must be an absolute canonical file.")
    return path


def _absolute_directory(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise Exp1bRuntimeError(f"{label} must be an absolute canonical directory.")
    return path


def _under_producer_root(path: Path, *, root: Path, label: str) -> Path:
    """Refuse any artifact path outside the producer root the launcher passed."""

    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise Exp1bRuntimeError(f"{label} is outside the producer root.") from exc
    return path


def _outside_producer_root(path: Path, *, root: Path, label: str) -> Path:
    """Refuse a signed owner artifact that sits *inside* the source checkout.

    Source authorization requires an exact clean commit and rejects any
    untracked or modified file. A signed admission placed in the checkout makes
    the tree dirty and is refused before launch; committing one that must name
    the very commit it lives in is circular. Owner-signed artifacts therefore
    live outside the checkout, and the launcher passes an absolute path to one.
    """

    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return path
    raise Exp1bRuntimeError(
        f"{label} must live outside the immutable source checkout; "
        "a signed artifact inside it cannot be authorized."
    )


def _stable_document(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    """Read one document as stable bytes and return it with its canonical digest."""

    try:
        payload, _file_sha, _size = stable_bytes(path, label=label)
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    try:
        parsed = load_strict_json_bytes(payload)
    except PolicyImprovementSchemaError as exc:
        raise Exp1bRuntimeError(f"{label} is not strict JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise Exp1bRuntimeError(f"{label} is not an object.")
    return parsed, exp1b_document_sha256(parsed)


def _attestation(value: Mapping[str, str] | None) -> Exp1bRuntimeAttestation:
    """Build the runtime attestation from the entrypoint's preflight values."""

    if not isinstance(value, Mapping):
        raise Exp1bRuntimeError(
            "Experiment 1B requires its packaged entrypoint's runtime attestation."
        )
    required = {
        "source_git_commit",
        "runtime_sha256",
        "launcher_sha256",
        "runtime_authorization_sha256",
    }
    if set(value) != required:
        raise Exp1bRuntimeError("Experiment 1B runtime attestation inventory differs.")
    return Exp1bRuntimeAttestation(
        source_git_commit=str(value["source_git_commit"]),
        runtime_sha256=str(value["runtime_sha256"]),
        launcher_sha256=str(value["launcher_sha256"]),
        runtime_authorization_sha256=str(value["runtime_authorization_sha256"]),
    )


@dataclass(frozen=True)
class Exp1bAdmittedStudy:
    """The four study documents, validated, cross-bound, and admitted."""

    protocol: Mapping[str, Any]
    registry: Mapping[str, Any]
    amendment: Mapping[str, Any]
    admission: Exp1bExecutionAdmission
    admission_document: Mapping[str, Any]
    role: str
    effective_protocol: Mapping[str, Any]
    protocol_sha256: str
    registry_sha256: str
    amendment_sha256: str
    admission_sha256: str


def admit_exp1b_study(
    *,
    protocol_path: Path,
    registry_path: Path,
    amendment_path: Path,
    admission_path: Path,
    producer_root: Path,
    role: str,
    attestation: Exp1bRuntimeAttestation,
    environment: Mapping[str, str] | None = None,
) -> Exp1bAdmittedStudy:
    """Run gates 1 through 4 and return the admitted study, or refuse.

    Shared by both stages on purpose. A gate that only Stage B enforced would let
    Stage A produce evidence that Stage B then has to trust or discard; the
    review's finding was exactly that asymmetry.
    """

    if role not in ADMISSION_ROLES:
        raise Exp1bRuntimeError(
            f"Experiment 1B stage role {role!r} is not a registered runtime role."
        )
    for path, label in (
        (protocol_path, "reduced-study protocol"),
        (registry_path, "reduced-study registry"),
        (amendment_path, "reduced-study amendment"),
    ):
        _under_producer_root(path, root=producer_root, label=label)
    _outside_producer_root(
        admission_path, root=producer_root, label="execution admission amendment"
    )

    protocol, protocol_sha256 = _stable_document(
        protocol_path, label="reduced-study protocol"
    )
    registry, registry_sha256 = _stable_document(
        registry_path, label="reduced-study registry"
    )
    amendment, amendment_sha256 = _stable_document(
        amendment_path, label="reduced-study amendment"
    )
    admission_document, _admission_file_sha = _stable_document(
        admission_path, label="execution admission amendment"
    )

    try:
        validate_exp1b_protocol(protocol)
        validate_exp1b_registry(registry, protocol_sha256=protocol_sha256)
        validate_exp1b_amendment(
            amendment,
            protocol_sha256=protocol_sha256,
            registry_sha256=registry_sha256,
        )
        admission = validate_exp1b_admission(
            admission_document,
            protocol_sha256=protocol_sha256,
            registry_sha256=registry_sha256,
            prior_amendment_sha256=amendment_sha256,
        )
        if admission.purpose != EXECUTION_PURPOSE:
            raise Exp1bSchemaError("Experiment 1B admission opens another purpose.")
        admission.require_attestation(
            role=role,
            source_git_commit=attestation.source_git_commit,
            runtime_sha256=attestation.runtime_sha256,
            launcher_sha256=attestation.launcher_sha256,
            runtime_authorization_sha256=attestation.runtime_authorization_sha256,
        )
        admission.require_execution_environment(environment)
        effective = apply_exp1b_admission(protocol, admission)
    except Exp1bSchemaError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc

    return Exp1bAdmittedStudy(
        protocol=protocol,
        registry=registry,
        amendment=amendment,
        admission=admission,
        admission_document=admission_document,
        role=role,
        effective_protocol=effective,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        amendment_sha256=amendment_sha256,
        admission_sha256=admission.admission_sha256,
    )


def _training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--policy-improvement-exp1b-training", action="store_true")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--reduced-study-protocol", required=True)
    parser.add_argument("--reduced-study-registry", required=True)
    parser.add_argument("--reduced-study-amendment", required=True)
    parser.add_argument("--execution-admission", required=True)
    parser.add_argument("--base-policy-artifact", required=True)
    parser.add_argument("--base-policy-amendment", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--evidence-generation", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: Any = None,
    runtime_attestation: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Stage A: build, run, and durably publish one registered train-only run.

    ``backend`` is the authenticated ``SealedFullRunBackend`` the packaged
    entrypoint constructs after its preflight. Without it, and without the
    runtime attestation, this refuses to start: the presence of authorization
    arguments is not authentication.

    The dataset root is *not* an argument. It is derived from the registered
    training population, so a caller cannot point the run at a directory the
    protocol never registered.
    """

    arguments = _training_parser().parse_args(argv)
    if not arguments.policy_improvement_exp1b_training or backend is None:
        raise Exp1bRuntimeError(
            "Experiment 1B training requires its authenticated packaged entrypoint."
        )
    attestation = _attestation(runtime_attestation)
    producer_root = _absolute_directory(arguments.project_root, label="producer root")

    study = admit_exp1b_study(
        protocol_path=_absolute_file(
            arguments.reduced_study_protocol, label="reduced-study protocol"
        ),
        registry_path=_absolute_file(
            arguments.reduced_study_registry, label="reduced-study registry"
        ),
        amendment_path=_absolute_file(
            arguments.reduced_study_amendment, label="reduced-study amendment"
        ),
        admission_path=_absolute_file(
            arguments.execution_admission, label="execution admission amendment"
        ),
        producer_root=producer_root,
        role=FULL_ROLE,
        attestation=attestation,
        environment=environment,
    )

    base_artifact = _outside_producer_root(
        _absolute_file(arguments.base_policy_artifact, label="base-policy artifact"),
        root=producer_root,
        label="base-policy artifact",
    )
    base_amendment = _outside_producer_root(
        _absolute_file(arguments.base_policy_amendment, label="base-policy amendment"),
        root=producer_root,
        label="base-policy amendment",
    )
    # Authenticate the admitted base artifact against the signed admission
    # before it is handed to anything that would deserialize it.
    try:
        _bytes, base_sha256, _size = stable_bytes(
            base_artifact, label="Experiment 1B base-policy artifact"
        )
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    if base_sha256 != study.admission.base_policy_checkpoint_sha256:
        raise Exp1bRuntimeError(
            "Base-policy artifact bytes differ from the admitted checkpoint digest."
        )
    _amendment_document, base_amendment_sha256 = _stable_document(
        base_amendment, label="base-policy admission amendment"
    )
    if base_amendment_sha256 != study.admission.base_policy_amendment_sha256:
        raise Exp1bRuntimeError(
            "Base-policy amendment differs from the admitted signed amendment."
        )

    try:
        registration = register_exp1b_training(
            protocol=study.protocol,
            registry=study.registry,
            project_root=producer_root,
            seed=int(arguments.seed),
        )
    except (Exp1bSessionError, Exp1bSchemaError) as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    if registration.run_id != arguments.row_id:
        raise Exp1bRuntimeError(
            "Experiment 1B row ID differs from the registered row for that seed."
        )

    try:
        generation = open_evidence_generation(
            evidence_root=_absolute_directory(
                arguments.evidence_root, label="evidence root"
            ),
            generation_id=str(arguments.evidence_generation),
        )
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc

    # Finding 3: a retry of a seed that already published must not retrain.
    #
    # Training is not bit-reproducible, so re-running a completed
    # 10,000-interaction seed produces different checkpoint bytes and different
    # counters, `_install_or_adopt` then refuses the existing artifact, and the
    # generation can never be finalized. Recovery previously worked only when the
    # repeat happened to be byte-identical, which a real run is not.
    #
    # The check runs *after* study and base authentication -- an unauthenticated
    # process must not be able to skip work -- and before anything Torch-bearing
    # is constructed.
    already_published = _authenticated_publication(
        generation=generation,
        registration=registration,
        study=study,
        attestation=attestation,
        base_policy_checkpoint_sha256=base_sha256,
    )
    if already_published is not None:
        _finalize_when_complete(
            generation=generation,
            study=study,
            base_policy=_base_descriptor_from_amendment(
                amendment=_amendment_document,
                checkpoint_sha256=base_sha256,
                model_state_sha256=(
                    study.admission.base_policy_model_state_sha256
                ),
            ),
            admitted_base_artifact=base_artifact,
            attestation=attestation,
        )
        return 0

    # Finding 2: this seed is not published, so it runs -- but if a crash landed
    # between its checkpoint and its manifest, an unreferenced checkpoint is
    # still sitting on the final pathname. The rerun cannot produce those bytes
    # again (training is not bit reproducible), so `_install_or_adopt` would
    # refuse the new one after a full 10,000-interaction run. Retire the leftover
    # under the generation lock first, while it is provably referenced by
    # nothing. A complete pair is never touched: it was adopted above.
    try:
        generation.retire_unreferenced_checkpoint(registration.seed_position)
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc

    # Everything Torch-bearing arrives through the authenticated backend.
    prepared = backend.prepare_exp1b_training(
        producer_root=producer_root,
        dataset_root=registration.dataset_root,
        registration=registration,
        protocol=study.protocol,
        base_policy_artifact=base_artifact,
        base_policy_amendment=base_amendment,
        base_policy_checkpoint_sha256=base_sha256,
        base_policy_model_state_sha256=(
            study.admission.base_policy_model_state_sha256
        ),
        effective_config_sha256=registration.effective_config_sha256,
        runtime_attestation=attestation,
    )
    guard = TrainOnlyDatasetGuard(registration=registration, loader=prepared.load_split)
    try:
        session = build_exp1b_training_session(
            registration=registration,
            dataset_guard=guard,
            base_policy=prepared.base_policy,
            apply_seed=prepared.apply_seed,
            model_factory=prepared.model_factory,
            restore_base_policy=prepared.restore_base_policy,
            trainer_factory=prepared.trainer_factory,
            model_state_digest=prepared.model_state_digest,
        )
    except Exp1bSessionError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc

    outcome = backend.run_exp1b_training(session=session)
    code = publish_exp1b_training_outcome(
        generation=generation,
        registration=registration,
        session=session,
        outcome=outcome,
        admission_sha256=study.admission_sha256,
        attestation=attestation,
    )
    # The eighth run finalizes the generation. Doing it here rather than in a
    # separate launcher purpose keeps the step inside the authenticated Stage A
    # path that already holds the admitted base policy and the validated study;
    # `O_EXCL` on the provenance makes it exactly-once regardless of which run
    # observes the last manifest.
    _finalize_when_complete(
        generation=generation,
        study=study,
        base_policy=prepared.base_policy,
        admitted_base_artifact=base_artifact,
        attestation=attestation,
    )
    return code


def _finalize_when_complete(
    *,
    generation: Any,
    study: Exp1bAdmittedStudy,
    base_policy: Any,
    admitted_base_artifact: Path,
    attestation: Exp1bRuntimeAttestation,
) -> None:
    """Publish provenance once all eight budget-final runs are durable.

    A no-op until the eighth. Losing the race is not an error: the provenance is
    written ``O_EXCL``, so a second finalizer sees the artifact and stops.
    """

    # Recovery read: reap under the lock before classifying.
    try:
        generation.reap_under_lock()
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    # Complete means all eight, both files each. `checkpoint_only` is a crash
    # between a seed's two artifacts: the seed has to be republished, and
    # returning here lets that happen without an error.
    for position in range(UNITS):
        if generation.checkpoint_publication_state(position) != "complete":
            return
    if generation.provenance_path.exists():
        # Finding 3: prove it, do not assume it. A partial finalization used to
        # read as a completed one because the base artifact existed and the
        # error message contained "already exists"; the generation then stayed
        # permanently unfinishable. `finalize_exp1b_evidence` now adopts a
        # byte-identical provenance and refuses a different one, so re-running it
        # is the check.
        pass

    protocol = study.protocol
    population = protocol["evaluation_population"]
    descriptor_fields = (
        "architecture_sha256",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "training_procedure_sha256",
    )
    descriptor: dict[str, Any] = {
        "initialization_kind": "train_only_pretrained_base_policy",
        "amendment_sha256": study.admission.base_policy_amendment_sha256,
        "model_state_sha256": study.admission.base_policy_model_state_sha256,
        "training_data_sha256": str(
            getattr(base_policy, "training_split_ordered_record_sha256", "")
        ),
        "shared_across_seeds": True,
        "not_selected_by_validation_or_test": True,
    }
    for name in descriptor_fields:
        value = getattr(base_policy, name, None)
        if not isinstance(value, str) or not value:
            raise Exp1bRuntimeError(
                f"Admitted base policy exposes no {name} for evidence finalization."
            )
        descriptor[name] = value
    if not descriptor["training_data_sha256"]:
        raise Exp1bRuntimeError(
            "Admitted base policy exposes no training-split identity."
        )
    try:
        finalize_exp1b_evidence(
            generation=generation,
            admitted_base_artifact=admitted_base_artifact,
            admission_sha256=study.admission_sha256,
            base_descriptor=descriptor,
            registry_rows=registry_rows_by_run_id(study.registry),
            exp1b_protocol_sha256=study.protocol_sha256,
            exp1b_registry_sha256=study.registry_sha256,
            exp1b_amendment_sha256=study.amendment_sha256,
            parent=dict(protocol["parent"]),
            ordered_population_sha256=str(population["ordered_record_sha256"]),
            population_binding_sha256=str(population["binding_sha256"]),
            attestation={
                "source_git_commit": attestation.source_git_commit,
                "runtime_sha256": attestation.runtime_sha256,
                "launcher_sha256": attestation.launcher_sha256,
                "runtime_authorization_sha256": (
                    attestation.runtime_authorization_sha256
                ),
            },
            producer_authorization=study.admission.authorization_for(FULL_ROLE),
        )
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc


def _base_descriptor_from_amendment(
    *, amendment: Mapping[str, Any], checkpoint_sha256: str, model_state_sha256: str
) -> Any:
    """The base-artifact descriptor fields, without constructing a Torch session.

    The finalizer needs the admitted base policy's producer identity. On the
    normal path that comes off the authenticated ``AuthenticatedBasePolicy`` the
    backend built; on the adopt-only retry path there is no backend, so the same
    fields are read from the signed amendment the runtime already authenticated.
    Both sources are the same document -- the amendment is what
    ``authenticate_base_policy_artifact`` validates against.
    """

    registered = amendment.get("base_policy_artifact")
    if not isinstance(registered, Mapping):
        raise Exp1bRuntimeError(
            "Base-policy amendment registers no artifact to finalize from."
        )
    return SimpleNamespace(
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
        architecture_sha256=str(registered["architecture_sha256"]),
        producer_git_commit=str(registered["producer_git_commit"]),
        producer_source_manifest_sha256=str(
            registered["producer_source_manifest_sha256"]
        ),
        training_procedure_sha256=str(registered["training_procedure_sha256"]),
        training_split_ordered_record_sha256=str(
            registered["training_split_ordered_record_sha256"]
        ),
    )


def _authenticated_publication(
    *,
    generation: Any,
    registration: Any,
    study: Exp1bAdmittedStudy,
    attestation: Exp1bRuntimeAttestation,
    base_policy_checkpoint_sha256: str,
) -> Mapping[str, Any] | None:
    """Return this seed's already-published manifest, or ``None`` to run it.

    Adoption is not "a file exists". The checkpoint bytes are re-hashed, the
    manifest is re-read and re-validated, and every identity it asserts -- run,
    seed, position, applied seed, budget, effective config, train split, record
    count, no-evaluation-access, admission, producer runtime, and the base
    artifact it initialized from -- has to match what *this* invocation was
    authenticated for. Anything short of a complete, matching pair means the seed
    has not been published and training proceeds.
    """

    position = registration.seed_position
    # Recovery read: clear this module's crash leftovers under the generation
    # lock before classifying, so a link/unlink crash window is transient. The
    # readers themselves do not reap -- an unlocked reader would delete a
    # concurrent writer's staging temporary.
    try:
        generation.reap_under_lock()
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    try:
        state = generation.checkpoint_publication_state(position)
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    if state != "complete":
        # `absent` has nothing to adopt. `checkpoint_only` is a crash between the
        # pair's two files: the manifest carries the counters and identities that
        # only the run itself can produce, so the seed is genuinely unpublished
        # and must run again.
        return None

    directory = generation.checkpoint_directory_for(position)
    try:
        payload, checkpoint_sha256, size = stable_bytes(
            directory / "checkpoint.pt",
            label=f"Experiment 1B checkpoint {position}",
        )
        del payload
        manifest, _digest = _stable_document(
            directory / "run_manifest.json",
            label=f"Experiment 1B run manifest {position}",
        )
    except (Exp1bEvidenceError, Exp1bRuntimeError) as exc:
        raise Exp1bRuntimeError(
            f"Experiment 1B seed {position} has an unreadable published pair: {exc}"
        ) from exc

    expected = {
        "run_id": registration.run_id,
        "seed": registration.seed,
        "seed_position": position,
        "applied_seed": registration.seed,
        "environment_interactions": TERMINAL_ENVIRONMENT_INTERACTIONS,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": size,
        "effective_config_sha256": registration.effective_config_sha256,
        "train_split_ordered_record_sha256": registration.ordered_record_sha256,
        "train_record_count": registration.train_record_count,
        "resolved_evaluation_data": False,
        "admission_sha256": study.admission_sha256,
        "initialization_artifact_sha256": base_policy_checkpoint_sha256,
        "restored_base_model_state_sha256": (
            study.admission.base_policy_model_state_sha256
        ),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise Exp1bRuntimeError(
                f"Experiment 1B seed {position} is already published with a "
                f"different {name}; it was not produced by this invocation's "
                "authenticated inputs."
            )
    observed_producer = manifest.get("producer_attestation")
    try:
        expected_producer = role_attestation_document(
            role=FULL_ROLE,
            attestation={
                "source_git_commit": attestation.source_git_commit,
                "runtime_sha256": attestation.runtime_sha256,
                "launcher_sha256": attestation.launcher_sha256,
                "runtime_authorization_sha256": (
                    attestation.runtime_authorization_sha256
                ),
            },
        )
    except Exp1bSchemaError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    if dict(observed_producer or {}) != expected_producer:
        raise Exp1bRuntimeError(
            f"Experiment 1B seed {position} is already published under a "
            "different producer runtime."
        )
    return manifest


def publish_exp1b_training_outcome(
    *,
    generation: Any,
    registration: Any,
    session: Any,
    outcome: Any,
    admission_sha256: str,
    attestation: Exp1bRuntimeAttestation,
) -> int:
    """Write one seed's budget-final checkpoint and run manifest, durably.

    Separated from :func:`main` so the publication contract -- what the manifest
    has to assert, and that it has to agree with the bytes -- is testable without
    a backend. The manifest fields are exactly what
    ``authenticate_sealed_octet_checkpoints`` will demand when Stage B opens.
    """

    checkpoint_bytes = getattr(outcome, "checkpoint_bytes", None)
    if not isinstance(checkpoint_bytes, (bytes, bytearray)) or not checkpoint_bytes:
        raise Exp1bRuntimeError("Experiment 1B training produced no checkpoint bytes.")
    interactions = getattr(outcome, "environment_interactions", None)
    if interactions != TERMINAL_ENVIRONMENT_INTERACTIONS:
        raise Exp1bRuntimeError(
            "Experiment 1B run did not reach the registered terminal interaction "
            "budget."
        )
    model_state_sha256 = getattr(outcome, "model_state_sha256", None)
    if not isinstance(model_state_sha256, str) or len(model_state_sha256) != 64:
        raise Exp1bRuntimeError("Experiment 1B run reported no model-state digest.")
    if session.dataset_guard.touched_evaluation_data:
        raise Exp1bRuntimeError("Experiment 1B run resolved evaluation data.")
    if session.train_record_count != TRAINING_RECORD_COUNT:
        raise Exp1bRuntimeError(
            "Experiment 1B run did not train on the registered record count."
        )

    payload = bytes(checkpoint_bytes)
    counters = getattr(outcome, "counters", None)
    if not isinstance(counters, Mapping) or not counters:
        raise Exp1bRuntimeError("Experiment 1B run reported no trainer counters.")
    manifest = {
        "schema_name": RUN_MANIFEST_SCHEMA_NAME,
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": session.run_id,
        "seed": session.seed,
        "seed_position": session.seed_position,
        "applied_seed": session.applied_seed,
        "environment_interactions": TERMINAL_ENVIRONMENT_INTERACTIONS,
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_size_bytes": len(payload),
        "model_state_sha256": model_state_sha256,
        "effective_config_sha256": registration.effective_config_sha256,
        "train_split_ordered_record_sha256": session.train_ordered_record_sha256,
        "train_record_count": session.train_record_count,
        "resolved_evaluation_data": False,
        # Finding 6: the manifest records what the run initialized from, so
        # evidence finalization can prove all eight descend from the one
        # admitted base artifact instead of taking it on the provenance's word.
        "initialization_kind": session.initialization_kind,
        "initialization_artifact_sha256": session.initialization_artifact_sha256,
        "restored_base_model_state_sha256": session.restored_model_state_sha256,
        "counters": {str(k): int(v) for k, v in sorted(counters.items())},
        # Finding 4: every run records which signed admission opened it and
        # which producer runtime produced it, so evidence finalization can
        # derive the provenance's producer identity from eight unanimous
        # attestations instead of taking one caller's word alongside them.
        "admission_sha256": admission_sha256,
        "producer_attestation": role_attestation_document(
            role=FULL_ROLE,
            attestation={
                "source_git_commit": attestation.source_git_commit,
                "runtime_sha256": attestation.runtime_sha256,
                "launcher_sha256": attestation.launcher_sha256,
                "runtime_authorization_sha256": (
                    attestation.runtime_authorization_sha256
                ),
            },
        ),
    }
    if session.applied_seed != session.seed:
        raise Exp1bRuntimeError(
            "Experiment 1B run applied a seed other than its registered one."
        )
    try:
        generation.publish_checkpoint(
            seed_position=session.seed_position,
            checkpoint_bytes=payload,
            run_manifest=manifest,
        )
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    return 0


def _bridge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--policy-improvement-exp1b-bridge", action="store_true")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--reduced-study-protocol", required=True)
    parser.add_argument("--reduced-study-registry", required=True)
    parser.add_argument("--reduced-study-amendment", required=True)
    parser.add_argument("--execution-admission", required=True)
    parser.add_argument("--parent-protocol", required=True)
    parser.add_argument("--parent-registry", required=True)
    parser.add_argument("--parent-populations", required=True)
    parser.add_argument("--parent-theory-amendment", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--evidence-generation", required=True)
    parser.add_argument("--seed-position", required=True, type=int)
    return parser


def bridge_main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: Any = None,
    runtime_attestation: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Stage B: serve one seed of the authenticated sealed-octet bridge route.

    Opens the route through :func:`open_authenticated_exp1b_route`, which
    computes every trust anchor from stable files and the supplied attestation
    and authenticates all eight sealed checkpoints plus the admitted base
    artifact. When the eighth seed completes, the sealed octet is aggregated, the
    result is revalidated immediately before publication, and it is written
    durably into the same evidence generation the payloads came from.
    """

    arguments = _bridge_parser().parse_args(argv)
    if not arguments.policy_improvement_exp1b_bridge or not callable(backend_factory):
        raise Exp1bRuntimeError(
            "Experiment 1B bridge requires its authenticated packaged entrypoint."
        )
    attestation = _attestation(runtime_attestation)
    producer_root = _absolute_directory(arguments.project_root, label="producer root")

    protocol_path = _absolute_file(
        arguments.reduced_study_protocol, label="reduced-study protocol"
    )
    registry_path = _absolute_file(
        arguments.reduced_study_registry, label="reduced-study registry"
    )
    amendment_path = _absolute_file(
        arguments.reduced_study_amendment, label="reduced-study amendment"
    )
    study = admit_exp1b_study(
        protocol_path=protocol_path,
        registry_path=registry_path,
        amendment_path=amendment_path,
        admission_path=_absolute_file(
            arguments.execution_admission, label="execution admission amendment"
        ),
        producer_root=producer_root,
        role=THEORY_BRIDGE_ROLE,
        attestation=attestation,
        environment=environment,
    )

    parent_paths = {
        "parent_protocol_path": _absolute_file(
            arguments.parent_protocol, label="parent protocol"
        ),
        "parent_registry_path": _absolute_file(
            arguments.parent_registry, label="parent registry"
        ),
        "parent_population_path": _absolute_file(
            arguments.parent_populations, label="parent population registry"
        ),
        "parent_theory_amendment_path": _absolute_file(
            arguments.parent_theory_amendment, label="parent theory amendment"
        ),
    }
    for name, path in parent_paths.items():
        _under_producer_root(path, root=producer_root, label=name)

    try:
        route = open_authenticated_exp1b_route(
            exp1b_protocol_path=protocol_path,
            exp1b_registry_path=registry_path,
            exp1b_amendment_path=amendment_path,
            evidence_root=_absolute_directory(
                arguments.evidence_root, label="evidence root"
            ),
            generation_id=str(arguments.evidence_generation),
            attestation=attestation,
            producer_authorization=study.admission.authorization_for(FULL_ROLE),
            evaluator_authorization=study.admission.authorization_for(
                THEORY_BRIDGE_ROLE
            ),
            **parent_paths,
        )
    except (Exp1bBridgeError, Exp1bSchemaError) as exc:
        raise Exp1bRuntimeError(str(exc)) from exc

    # The route recomputed the study digests itself. If they disagree with the
    # admitted ones, two different revisions are in play and neither is served.
    if (
        route.exp1b_protocol_sha256 != study.protocol_sha256
        or route.exp1b_registry_sha256 != study.registry_sha256
        or route.exp1b_amendment_sha256 != study.amendment_sha256
    ):
        raise Exp1bRuntimeError(
            "Bridge route documents differ from the admitted Experiment 1B study."
        )

    effective_config_sha256 = str(
        study.effective_protocol["effective_config"]["effective_config_sha256"]
    )
    # Bind the admitted base identity to the provenance the route authenticated.
    # Without this the base artifact is only self-consistent with the document
    # that describes it, and a consistently edited pair opens the route.
    route_base = route.provenance_base_policy_artifact
    if (
        route_base["checkpoint_sha256"] != study.admission.base_policy_checkpoint_sha256
        or route_base["model_state_sha256"]
        != study.admission.base_policy_model_state_sha256
        or route_base["amendment_sha256"]
        != study.admission.base_policy_amendment_sha256
    ):
        raise Exp1bRuntimeError(
            "Evidence base-policy identity differs from the signed admission."
        )
    if route.provenance_admission_sha256 != study.admission_sha256:
        raise Exp1bRuntimeError(
            "Evidence generation was finalized under a different signed admission."
        )
    # The backend is built from the route, not before it: its eight
    # authenticated checkpoints and the census ordering are outputs of route
    # authentication, so a backend constructed earlier would be trusting claims
    # the route had not yet proved.
    if route.access_state == "payload_emitted":
        # Restart after the eighth payload landed. There is nothing left to
        # serve; resume at aggregation instead of rejecting an already served
        # seed and stranding a complete generation.
        return _publish_exp1b_result(
            route=route, study=study, backend_factory=backend_factory,
            attestation=attestation, runtime_attestation=runtime_attestation,
            effective_config_sha256=effective_config_sha256,
        )

    registered_training = study.effective_protocol["training_population"]
    evaluation_population = study.effective_protocol["evaluation_population"]
    position = int(arguments.seed_position)
    if not 0 <= position < UNITS:
        raise Exp1bRuntimeError("Seed position is outside the registered octet.")
    descriptor = route.authenticated_checkpoints[position]
    served_backend: Any = None

    # Finding 2: the durable claim comes first. Backend construction, the
    # evaluation-split load, the sealed-checkpoint restore, every model call and
    # the 128-member secondary traversal all happen inside it. Previously all of
    # that ran before the schedule saw the request, so an out-of-order or
    # concurrent request could complete a full traversal before being refused,
    # and a malformed record left the slot unclaimed with attempt count 0.
    try:
        with route.claim_seed(
            position,
            seed=descriptor.seed,
            run_id=descriptor.run_id,
            checkpoint_sha256=descriptor.checkpoint_sha256,
            evaluation_population=EVALUATION_POPULATION_ID,
        ) as claim:
            backend = served_backend = backend_factory(
                runtime_attestation=dict(runtime_attestation or {}),
                authenticated_checkpoints=route.authenticated_checkpoints,
                effective_config_sha256=effective_config_sha256,
                census_ordering_sha256=route.census_ordering_sha256,
                census=route.census_members,
                dataset_root=(
                    producer_root / str(registered_training["dataset_root"])
                ).resolve(),
                evaluation_split=str(evaluation_population["split"]),
                evaluation_split_manifest_sha256=str(
                    evaluation_population["split_manifest_sha256"]
                ),
            )
            evaluator = backend.prepare_exp1b_bridge(
                seed_position=position,
                runtime_attestation=attestation,
                effective_config_sha256=effective_config_sha256,
            )
            if (
                evaluator.seed_position != position
                or evaluator.seed != descriptor.seed
                or evaluator.run_id != descriptor.run_id
                or evaluator.checkpoint_sha256 != descriptor.checkpoint_sha256
            ):
                raise Exp1bRuntimeError(
                    "Stage B evaluator identity differs from the claimed slot."
                )
            claim.serve(
                endpoint_values=evaluator.endpoint_values,
                action_values=evaluator.action_values,
                action_mask=evaluator.action_mask,
                base_probabilities=evaluator.base_probabilities,
                secondary_diagnostics=evaluator.secondary_diagnostics,
            )
    except Exp1bBridgeError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc

    if route.access_state != "payload_emitted":
        return 0
    return _publish_exp1b_result(
        route=route,
        study=study,
        backend_factory=backend_factory,
        attestation=attestation,
        runtime_attestation=runtime_attestation,
        effective_config_sha256=effective_config_sha256,
        backend=served_backend,
    )


def _publish_exp1b_result(
    *,
    route: Any,
    study: Exp1bAdmittedStudy,
    backend_factory: Any,
    attestation: Exp1bRuntimeAttestation,
    runtime_attestation: Mapping[str, str] | None,
    effective_config_sha256: str,
    backend: Any = None,
) -> int:
    """Aggregate, audit, and durably publish exactly once.

    Reached both by the eighth serving process and by a restart that finds a
    complete generation. Publication is idempotent-safe rather than idempotent:
    a second attempt is refused by the no-replace artifact, and a crash between
    the document and its digest sidecar is repaired here rather than left as an
    unreconcilable state.
    """

    try:
        result = aggregate_exp1b_result(route.sealed_octet())
    except (Exp1bAggregateError, Exp1bBridgeError) as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    # What this process expects the durable result to be, for the readback check
    # below. Deliberately *not* an input to publication: the writer serializes,
    # validates, and audits the result itself, so there is no document argument
    # to disagree with what was checked.
    document = canonical_json_bytes(result.as_document()) + b"\n"

    generation = route.generation
    publication = {
        "result": result,
        "route": route,
        "protocol": study.protocol,
        "registry": study.registry,
        "amendment": study.amendment,
        "admission": study.admission_document,
    }
    try:
        # Reap under the lock before classifying, then publish outside it:
        # publish_result and complete_result_publication take the lock
        # themselves, and flock does not recurse within one process.
        generation.reap_under_lock()
        state = generation.result_publication_state()
        if state == "absent":
            generation.publish_result(**publication)
        elif state == "document_only":
            # Crash between the document and its sidecar. The document is
            # durable and byte-identical to what we would write, so completing
            # the journal is a repair, not a replacement -- and it runs the same
            # validation and audit that the first write did.
            generation.complete_result_publication(**publication)
        republished, republished_sha256 = generation.read_published_result()
    except Exp1bEvidenceError as exc:
        raise Exp1bRuntimeError(str(exc)) from exc
    if republished != document or republished_sha256 != result.document_sha256():
        raise Exp1bRuntimeError(
            "Published Experiment 1B result does not read back as it was written."
        )
    if backend is not None:
        backend.publish_exp1b_result(
            protocol_id=PROTOCOL_ID,
            result_path=generation.result_path,
            document_sha256=result.document_sha256(),
            runtime_attestation=attestation,
        )
    return 0
