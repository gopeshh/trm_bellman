#!/usr/bin/env fbpython
"""Experiment 1B authenticated validation-bridge route.

This is the D1 route. It is a **separate** route, not a modification of the
policy-improvement-v2 one. The standing v2 refusal in
``scripts/policy_improvement_theory_backend_v2.py`` — every ``validation_bridge``
request rejected pending "the canonical 30-request schedule and its
independently re-audited V_select selection provenance" — is left exactly as it
is, and a v2 request still terminates there. Nothing in this module imports,
patches, subclasses, or reaches into that path.

Trust anchors are computed, not accepted
----------------------------------------
:func:`open_authenticated_exp1b_route` is the only production entry point. It
takes **file paths and a runtime attestation**, never digests. It reads each
document through :func:`stable_file_digest`, which authenticates the bytes while
they are still opaque — canonical absolute non-symlink regular path, single
link, stable inode metadata across the read — then computes every digest itself
and compares:

* the Experiment 1B protocol, registry, and amendment against the digests the
  substitute provenance claims;
* the v2 parent digests against the committed v2 documents;
* the census ordering and binding digests against values **recomputed from the
  registered population document**, not against the declared ones;
* ``source_git_commit``, ``runtime_sha256``, ``launcher_sha256``, and
  ``runtime_authorization_sha256`` against the launcher-supplied runtime
  attestation;
* every sealed-checkpoint run ID against an exact committed registry row.

Before the route exists, the factory also authenticates **all eight budget-final
checkpoint artifacts and the admitted base artifact** against their descriptors,
by size and SHA-256 over opaque bytes plus a digest-bound run manifest. A
provenance that merely claims ``sealed: true`` eight times no longer opens
anything. The substitute provenance itself comes from an owner-controlled
evidence generation, not a caller-named path, and the schedule artifact is
derived from that generation so there is exactly one state machine.

``Exp1bBridgeRoute.__init__`` requires a module-private token object, so no value
a caller can pass will satisfy it.

Aggregation barrier
-------------------
A completed route issues an :class:`Exp1bSealedOctet`, which carries the
provenance, schedule, census, attestation, and all eight payloads. That object
is the *only* thing the aggregator accepts, and only this module can construct
one, so there is no path from loose payload objects to a registered result.

Torch-free by construction
--------------------------
The per-state endpoint values and enumerated action values arrive through a
caller-supplied evaluator seam. The route's authorisation, ordering, and census
checks therefore run in the test suite in environments without Torch, which is
where the security-relevant logic lives.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scripts.policy_improvement_exp1_diagnostics import (
    paired_population_summary,
    paired_state_diagnostic,
    PairedEndpointObservation,
    PairedPopulationSummary,
    PopulationMember,
)
from scripts.policy_improvement_exp1b_evidence import (
    AuthenticatedCheckpoint,
    authenticate_base_artifact,
    authenticate_sealed_octet_checkpoints,
    Exp1bEvidenceError,
    Exp1bEvidenceGeneration,
    open_evidence_generation,
)
from scripts.policy_improvement_exp1b_schema import (
    attestation_matches_authorization,
    validate_secondary_diagnostics,
    BRIDGE_ROUTE_ID,
    Exp1bRoleAuthorization,
    role_attestation_document,
    THEORY_BRIDGE_ROLE,
    DEPLOYED_DEPTH_N,
    EVALUATION_POPULATION_ID,
    EVALUATION_RECORD_COUNT,
    Exp1bRequestSchedule,
    Exp1bSchemaError,
    exp1b_document_sha256,
    exp1b_state_id_for,
    GAMMA,
    REFERENCE_DEPTH_M,
    REGISTERED_SEEDS,
    registry_rows_by_run_id,
    UNITS,
    validate_exp1b_amendment,
    validate_exp1b_protocol,
    validate_exp1b_registry,
    validate_substitute_provenance,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
)

__all__ = [
    "Exp1bBridgeError",
    "Exp1bBridgeRoute",
    "Exp1bCensus",
    "Exp1bRuntimeAttestation",
    "Exp1bSealedOctet",
    "Exp1bSeedClaim",
    "Exp1bSeedPayload",
    "census_from_population",
    "open_authenticated_exp1b_route",
    "stable_file_digest",
]

def _plain_mapping(value: Any) -> Any:
    """Deep-convert proxies and tuples back to plain JSON containers."""

    if isinstance(value, Mapping):
        return {key: _plain_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_mapping(item) for item in value]
    return value


PAYLOAD_SCHEMA_NAME = "policy_improvement_exp1b_seed_payload_v1"
#: Version 2: seed payloads carry the five secondary diagnostic records.
PAYLOAD_SCHEMA_VERSION = 2

_FORBIDDEN_POPULATIONS = frozenset(
    {"validation_select", "confirmatory_test", "test", "stage0_smoke"}
)


class Exp1bBridgeError(RuntimeError):
    """Raised when the Experiment 1B bridge route would open incorrectly."""


@dataclass(frozen=True)
class _StableFile:
    path: Path
    sha256: str
    size_bytes: int
    payload: bytes


def stable_file_digest(path: str | os.PathLike[str], *, label: str) -> _StableFile:
    """Read a file and authenticate its bytes while they are still opaque.

    Mirrors the v2 theory bridge's stable-file discipline: the path must be
    absolute, canonical, and non-symlinked; the object must be a singly-linked
    regular file; and its inode metadata must be unchanged across the read, so
    a file swapped mid-read is rejected rather than half-trusted.
    """

    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise Exp1bBridgeError(f"{label} path must be absolute and canonical.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Exp1bBridgeError(f"{label} is unavailable.") from exc
    if resolved != candidate:
        raise Exp1bBridgeError(f"{label} path must not traverse an alias.")

    descriptor = -1
    try:
        before_path = candidate.lstat()
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before_path.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise Exp1bBridgeError(f"{label} is a link alias or the wrong type.")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        after_path = candidate.lstat()
    except OSError as exc:
        raise Exp1bBridgeError(f"{label} cannot be authenticated.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        identity(before_path) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(after_path)
    ):
        raise Exp1bBridgeError(f"{label} changed while it was read.")
    payload = b"".join(chunks)
    return _StableFile(
        path=candidate,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=before.st_size,
        payload=payload,
    )


def _stable_json(path: str | os.PathLike[str], *, label: str) -> tuple[Any, str]:
    """Authenticate a document's bytes, then parse it and digest it canonically."""

    stable = stable_file_digest(path, label=label)
    try:
        document = load_strict_json_bytes(stable.payload)
    except PolicyImprovementSchemaError as exc:
        raise Exp1bBridgeError(f"{label} is not strict JSON.") from exc
    return document, hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True)
class Exp1bRuntimeAttestation:
    """The launcher-supplied identity the route must match, not assert.

    Populated by the authenticated entrypoint from its preflight attestation.
    Every field here is something the launcher proved; the substitute provenance
    only *claims* the same values, and the route rejects any disagreement.
    """

    source_git_commit: str
    runtime_sha256: str
    launcher_sha256: str
    runtime_authorization_sha256: str

    def __post_init__(self) -> None:
        for field, value in (
            ("source_git_commit", self.source_git_commit),
            ("runtime_sha256", self.runtime_sha256),
            ("launcher_sha256", self.launcher_sha256),
            ("runtime_authorization_sha256", self.runtime_authorization_sha256),
        ):
            if not isinstance(value, str) or not value:
                raise Exp1bBridgeError(f"Runtime attestation {field} is missing.")


@dataclass(frozen=True)
class Exp1bCensus:
    """The fixed ordered 128-record validation_bridge census.

    ``ordered_record_sha256`` and ``binding_sha256`` are the values carried by
    the registered population document; ``ordering_sha256`` is recomputed from
    the members this census actually holds. :func:`census_from_population` is
    the only sanctioned constructor, and it derives the members from validated
    registration metadata rather than accepting them.
    """

    population_id: str
    split: str
    members: tuple[PopulationMember, ...]
    ordered_record_sha256: str
    binding_sha256: str
    member_ordering_sha256: str

    def __post_init__(self) -> None:
        if self.population_id != EVALUATION_POPULATION_ID or self.split != "validation":
            raise Exp1bBridgeError("Experiment 1B census is not validation_bridge.")
        if len(self.members) != EVALUATION_RECORD_COUNT:
            raise Exp1bBridgeError(
                f"Experiment 1B census must hold {EVALUATION_RECORD_COUNT} records, "
                f"holds {len(self.members)}."
            )
        if self.member_ordering_sha256 != _member_ordering_sha256(self.members):
            raise Exp1bBridgeError(
                "Experiment 1B census ordering digest does not match its members."
            )

    def ordering_sha256(self) -> str:
        return self.member_ordering_sha256


def _member_ordering_sha256(members: Sequence[PopulationMember]) -> str:
    return exp1b_document_sha256(
        [
            {
                "state_id": member.state_id,
                "record_index": member.record_index,
                "dataset_record_sha256": member.dataset_record_sha256,
            }
            for member in members
        ]
    )


#: The six secondary witness fields, by the record that carries each. Named once
#: so the route gate and the independent auditor cannot check different subsets:
#: the auditor previously covered four of the six and omitted both centering
#: sub-witnesses.
SECONDARY_WITNESS_FIELDS: tuple[tuple[str, str], ...] = (
    ("target_lag", "target_lag_witness_state_id"),
    ("centering_parity", "centering_witness_state_id"),
    ("centering_parity", "parity_witness_state_id"),
    ("centering_parity", "centering_defect_witness_state_id"),
    ("mixture_identity", "identity_witness_state_id"),
    ("deployment_mismatch", "mismatch_witness_state_id"),
)


def _require_census_witnesses(
    secondary: Mapping[str, Any], *, census: Exp1bCensus, label: str
) -> None:
    """Every secondary witness must name a member of this census."""

    members = {member.state_id for member in census.members}
    for record, field in SECONDARY_WITNESS_FIELDS:
        block = secondary.get(record)
        if not isinstance(block, Mapping):
            raise Exp1bBridgeError(f"{label} secondary diagnostics omit {record}.")
        witness = block.get(field)
        if witness not in members:
            raise Exp1bBridgeError(
                f"{label} {record}.{field} names {witness!r}, which is not a "
                "member of the registered validation_bridge census."
            )


def _require_registered_state_ids(census: Exp1bCensus) -> None:
    """Every census identifier must be the one the registered derivation gives.

    Belt to :func:`open_authenticated_exp1b_route`'s braces. The factory now
    passes :func:`exp1b_state_id_for` itself, so this cannot fail through that
    path -- which is the point: it is what holds if the constructor is ever
    reached another way, and it runs *before* the request schedule is built or
    touched, so a refusal costs no durable state.

    :func:`census_from_population` keeps its ``state_id_for`` argument. It is a
    pure constructor with no durable effect, and tests need to build a census
    under another derivation in order to prove this check bites.
    """

    for member in census.members:
        expected = exp1b_state_id_for(
            member.record_index, member.dataset_record_sha256
        )
        if member.state_id != expected:
            raise Exp1bBridgeError(
                "Experiment 1B census state identifiers are not the registered "
                f"derivation's: record {member.record_index} is "
                f"{member.state_id!r}."
            )


def census_from_population(
    population: Mapping[str, object],
    *,
    state_id_for: Callable[[int, str], str],
) -> Exp1bCensus:
    """Build the census from the registered population document.

    ``indices`` and ``record_sha256s`` are read in their registered order; this
    function never sorts, filters, or reorders them, and it recomputes the
    ordering digest from what it built. It reads registration metadata only and
    opens no record.
    """

    indices = population.get("indices")
    digests = population.get("record_sha256s")
    if not isinstance(indices, Sequence) or not isinstance(digests, Sequence):
        raise Exp1bBridgeError("Registered population lacks an ordered census.")
    if len(indices) != len(digests):
        raise Exp1bBridgeError("Registered population census is inconsistent.")
    members = tuple(
        PopulationMember(
            state_id=state_id_for(int(index), str(digest)),
            record_index=int(index),
            dataset_record_sha256=str(digest),
        )
        for index, digest in zip(indices, digests)
    )
    return Exp1bCensus(
        population_id=str(population["population_id"]),
        split=str(population["split"]),
        members=members,
        ordered_record_sha256=str(population["ordered_record_sha256"]),
        binding_sha256=str(population["binding_sha256"]),
        member_ordering_sha256=_member_ordering_sha256(members),
    )


@dataclass(frozen=True)
class _PersistedSeedResult:
    """A served seed's aggregate values, reloaded from its durable artifact."""

    state_count: int
    maximum_endpoint_discrepancy: float
    maximum_absolute_residual_n: float
    maximum_absolute_residual_m: float
    direct_residual_bound_n: float
    finite_reference_bound: float
    signed_gap: float
    discrepancy_witness_state_id: str
    residual_n_witness_state_id: str
    residual_m_witness_state_id: str
    route_id: str
    secondary_diagnostics: Mapping[str, Any]
    payload_sha256: str


def _persisted_payload_sha256(
    *,
    seed_position: int,
    seed: int,
    run_id: str,
    checkpoint_sha256: str,
    census_ordering_sha256: str,
    persisted: "_PersistedSeedResult",
) -> str:
    """The payload digest, rebuilt from the persisted scalars themselves.

    Reconstructs the exact canonical document ``write_payload`` digested, from
    the fields this record actually holds. Returning the stored digest instead
    would make it a second copy of the record rather than a check on it: an
    edited ``signed_gap`` would leave the digest unchanged and aggregation would
    compare one edited value against another.
    """

    return exp1b_document_sha256(
        {
            "schema_name": PAYLOAD_SCHEMA_NAME,
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "route_id": persisted.route_id,
            "seed_position": seed_position,
            "seed": seed,
            "run_id": run_id,
            "checkpoint_sha256": checkpoint_sha256,
            "census_ordering_sha256": census_ordering_sha256,
            "state_count": persisted.state_count,
            "maximum_endpoint_discrepancy": persisted.maximum_endpoint_discrepancy,
            "maximum_absolute_residual_n": persisted.maximum_absolute_residual_n,
            "maximum_absolute_residual_m": persisted.maximum_absolute_residual_m,
            "direct_residual_bound_n": persisted.direct_residual_bound_n,
            "finite_reference_bound": persisted.finite_reference_bound,
            "signed_gap": persisted.signed_gap,
            "discrepancy_witness_state_id": persisted.discrepancy_witness_state_id,
            "residual_n_witness_state_id": persisted.residual_n_witness_state_id,
            "residual_m_witness_state_id": persisted.residual_m_witness_state_id,
            "secondary_diagnostics": _plain_mapping(
                persisted.secondary_diagnostics
            ),
        }
    )


@dataclass(frozen=True)
class Exp1bSeedPayload:
    """One seed's sealed bridge payload: both depths, one census traversal.

    ``summary`` is present in the serving process. ``persisted`` is present when
    the payload was rebuilt from its durable artifact in a later process, which
    is how the documented one-process-per-seed flow assembles the octet.
    """

    seed_position: int
    seed: int
    run_id: str
    checkpoint_sha256: str
    summary: PairedPopulationSummary | None
    census_ordering_sha256: str
    secondary_diagnostics: Mapping[str, Any] | None = None
    persisted: _PersistedSeedResult | None = None
    #: Set by the route when it builds an in-memory payload; the reloaded path
    #: takes it from the persisted document instead.
    _route_id: str = BRIDGE_ROUTE_ID

    @property
    def route_id(self) -> str:
        """The route this payload was produced under, from either source."""

        if self.persisted is not None:
            return self.persisted.route_id
        return self._route_id

    def result_values(self) -> _PersistedSeedResult:
        """The seed's aggregate values, from memory or from its artifact."""

        if self.persisted is not None:
            return self.persisted
        summary = self.summary
        if summary is None:
            raise Exp1bBridgeError("Seed payload carries neither summary nor artifact.")
        return _PersistedSeedResult(
            state_count=summary.state_count,
            maximum_endpoint_discrepancy=summary.maximum_endpoint_discrepancy.value,
            maximum_absolute_residual_n=summary.maximum_absolute_residual_n.value,
            maximum_absolute_residual_m=summary.maximum_absolute_residual_m.value,
            direct_residual_bound_n=summary.direct_residual_bound_n,
            finite_reference_bound=summary.finite_reference_bound,
            signed_gap=summary.signed_gap,
            discrepancy_witness_state_id=(
                summary.maximum_endpoint_discrepancy.state_id
            ),
            residual_n_witness_state_id=summary.maximum_absolute_residual_n.state_id,
            residual_m_witness_state_id=summary.maximum_absolute_residual_m.state_id,
            route_id=self._route_id,
            secondary_diagnostics=_plain_mapping(self.secondary_diagnostics),
            payload_sha256=self.payload_sha256(),
        )

    def payload_sha256(self) -> str:
        """Digest binding this payload's identity to its computed aggregate.

        Always recomputed from the payload's own fields, including on the
        reloaded path. Returning the stored ``persisted.payload_sha256`` would
        make the digest a second copy of the same record rather than a check on
        it: editing a persisted ``signed_gap`` would leave the digest unchanged
        and aggregation would compare one edited value against another.
        """

        if self.persisted is not None:
            return _persisted_payload_sha256(
                seed_position=self.seed_position,
                seed=self.seed,
                run_id=self.run_id,
                checkpoint_sha256=self.checkpoint_sha256,
                census_ordering_sha256=self.census_ordering_sha256,
                persisted=self.persisted,
            )
        summary = self.summary
        if summary is None:
            raise Exp1bBridgeError("Seed payload carries neither summary nor artifact.")
        return exp1b_document_sha256(
            {
                "schema_name": PAYLOAD_SCHEMA_NAME,
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "route_id": self._route_id,
                "seed_position": self.seed_position,
                "seed": self.seed,
                "run_id": self.run_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "census_ordering_sha256": self.census_ordering_sha256,
                "state_count": summary.state_count,
                "maximum_endpoint_discrepancy": (
                    summary.maximum_endpoint_discrepancy.value
                ),
                "maximum_absolute_residual_n": (
                    summary.maximum_absolute_residual_n.value
                ),
                "maximum_absolute_residual_m": (
                    summary.maximum_absolute_residual_m.value
                ),
                "direct_residual_bound_n": summary.direct_residual_bound_n,
                "finite_reference_bound": summary.finite_reference_bound,
                "signed_gap": summary.signed_gap,
                "discrepancy_witness_state_id": (
                    summary.maximum_endpoint_discrepancy.state_id
                ),
                "residual_n_witness_state_id": (
                    summary.maximum_absolute_residual_n.state_id
                ),
                "residual_m_witness_state_id": (
                    summary.maximum_absolute_residual_m.state_id
                ),
                "secondary_diagnostics": _plain_mapping(self.secondary_diagnostics),
            }
        )


#: Module-private token for the sealed octet. Aggregation is the last barrier
#: before a number is published, so the object it consumes must be one this
#: module issued from a completed durable route, not a public data record a
#: caller can assemble or `dataclasses.replace` into a different shape.
_OCTET_ISSUANCE = object()

#: Octets this module actually issued, by identity. A ``dataclasses.replace``
#: copy passes the token check -- ``replace`` copies the field -- but is a
#: different object and is absent here. Weak, so an entry cannot outlive its
#: octet and be confused with a later one.
#: Keyed by ``id`` with weak values, because the octet is a frozen dataclass
#: holding mappings and is therefore unhashable. The weak value is what makes
#: the ``id`` key safe: the entry disappears with its octet, so a later object
#: that happens to reuse the address does not inherit its issuance.
_ISSUED_OCTETS: "weakref.WeakValueDictionary[int, Exp1bSealedOctet]" = (
    weakref.WeakValueDictionary()
)


@dataclass(frozen=True)
class Exp1bSealedOctet:
    """The completed route's output. Only the factory-issued route builds one.

    Every field is a digest or an immutable record. The aggregator revalidates
    all of them against the canonical documents and the durable schedule, so
    holding an instance of this class is not by itself authority.
    """

    _issued_by: Any
    route_id: str
    #: The Stage A identity the provenance was finalized under, and the live
    #: Stage B identity that served the octet. Distinct PARs, carried
    #: separately all the way to the published result.
    producer_attestation: Mapping[str, Any]
    evaluator_attestation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    provenance_sha256: str
    schedule_sha256: str
    generation_sha256: str
    attempt_counts: tuple[int, ...]
    payload_digests: tuple[str, ...]
    census_ordering_sha256: str
    census_binding_sha256: str
    exp1b_protocol_sha256: str
    exp1b_registry_sha256: str
    exp1b_amendment_sha256: str
    effective_config_sha256: str
    authenticated_checkpoints: tuple[AuthenticatedCheckpoint, ...]
    runtime_attestation: Exp1bRuntimeAttestation
    payloads: tuple[Exp1bSeedPayload, ...]

    def __post_init__(self) -> None:
        if self._issued_by is not _OCTET_ISSUANCE:
            raise Exp1bBridgeError(
                "Experiment 1B sealed octets are issued by a completed "
                "authenticated route; direct construction and replacement are "
                "refused."
            )
    def is_issued(self) -> bool:
        """Whether this exact object was minted by a completed route.

        The token check in ``__post_init__`` cannot answer this on its own:
        ``dataclasses.replace`` copies the token field, so a replaced octet
        satisfies it. Membership is by object identity in a registry that only
        :meth:`Exp1bBridgeRoute.sealed_octet` writes to, and a replacement is a
        different object. The registry is weak, so an entry disappears with its
        octet and no identity can be reused while it is still live.
        """

        return _ISSUED_OCTETS.get(id(self)) is self


#: Module-private issuance token. The route constructor requires the object
#: itself, so the previous ``_authenticated=True`` boolean — which any caller
#: could satisfy by passing a value — is gone.
_ISSUANCE = object()


def open_authenticated_exp1b_route(
    *,
    exp1b_protocol_path: str | os.PathLike[str],
    exp1b_registry_path: str | os.PathLike[str],
    exp1b_amendment_path: str | os.PathLike[str],
    parent_protocol_path: str | os.PathLike[str],
    parent_registry_path: str | os.PathLike[str],
    parent_population_path: str | os.PathLike[str],
    parent_theory_amendment_path: str | os.PathLike[str],
    evidence_root: str | os.PathLike[str],
    generation_id: str,
    attestation: Exp1bRuntimeAttestation,
    producer_authorization: Exp1bRoleAuthorization,
    evaluator_authorization: Exp1bRoleAuthorization,
) -> Exp1bBridgeRoute:
    """The single production factory. Consumes paths, computes every digest.

    Nothing here trusts a caller-supplied digest, and nothing here trusts a
    caller-supplied *claim*. Each document is read through
    :func:`stable_file_digest`; the provenance comes from the owner-controlled
    evidence generation rather than a free path; and **all eight checkpoint
    artifacts and the admitted base artifact are authenticated against their
    descriptors before the route exists**. The schedule path is derived from the
    generation, so there is exactly one state machine per generation.
    """

    if not isinstance(attestation, Exp1bRuntimeAttestation):
        raise Exp1bBridgeError("Experiment 1B route requires a runtime attestation.")

    try:
        generation = open_evidence_generation(
            evidence_root=evidence_root, generation_id=generation_id
        )
    except Exp1bEvidenceError as exc:
        raise Exp1bBridgeError(str(exc)) from exc

    protocol, protocol_sha = _stable_json(
        exp1b_protocol_path, label="Experiment 1B protocol"
    )
    registry, registry_sha = _stable_json(
        exp1b_registry_path, label="Experiment 1B registry"
    )
    amendment, amendment_sha = _stable_json(
        exp1b_amendment_path, label="Experiment 1B amendment"
    )
    parent_protocol, parent_protocol_sha = _stable_json(
        parent_protocol_path, label="parent protocol"
    )
    _parent_registry, parent_registry_sha = _stable_json(
        parent_registry_path, label="parent registry"
    )
    populations, population_registry_sha = _stable_json(
        parent_population_path, label="parent population registry"
    )
    _parent_theory, parent_theory_sha = _stable_json(
        parent_theory_amendment_path, label="parent theory amendment"
    )
    # Route opening is a recovery read like any other: clear this module's own
    # leftover links before authenticating, or a crash in the link/unlink window
    # makes the generation permanently unopenable. Under the lock, so it cannot
    # delete a concurrent writer's staging temporary.
    generation.reap_under_lock()
    provenance, _provenance_file_sha = _stable_json(
        generation.provenance_path, label="substitute provenance"
    )

    try:
        checked_protocol = validate_exp1b_protocol(protocol)
        checked_registry = validate_exp1b_registry(
            registry, protocol_sha256=protocol_sha
        )
        validate_exp1b_amendment(
            amendment, protocol_sha256=protocol_sha, registry_sha256=registry_sha
        )
    except Exp1bSchemaError as exc:
        raise Exp1bBridgeError(str(exc)) from exc

    parent = checked_protocol["parent"]
    assert isinstance(parent, Mapping)
    for field, computed in (
        ("protocol_sha256", parent_protocol_sha),
        ("registry_sha256", parent_registry_sha),
        ("population_registry_sha256", population_registry_sha),
        ("theory_amendment_sha256", parent_theory_sha),
    ):
        if parent[field] != computed:
            raise Exp1bBridgeError(
                f"Experiment 1B protocol cites a stale parent {field}."
            )

    registry_rows = registry_rows_by_run_id(checked_registry)
    try:
        record = validate_substitute_provenance(
            provenance, registry_rows=registry_rows
        )
    except Exp1bSchemaError as exc:
        raise Exp1bBridgeError(str(exc)) from exc

    if (
        record["exp1b_protocol_sha256"] != protocol_sha
        or record["exp1b_registry_sha256"] != registry_sha
        or record["exp1b_amendment_sha256"] != amendment_sha
    ):
        raise Exp1bBridgeError(
            "Substitute provenance does not bind the authenticated "
            "Experiment 1B documents."
        )
    provenance_parent = record["parent"]
    assert isinstance(provenance_parent, Mapping)
    if dict(provenance_parent) != dict(parent):
        raise Exp1bBridgeError(
            "Substitute provenance cites a different v2 parent than the protocol."
        )

    # Two roles, two comparisons. The provenance carries the *producer*
    # identity, written by Stage A under the full PAR; the live attestation is
    # the *evaluator* identity, running under the theory-bridge PAR. The
    # admission requires the two runtime digests to differ, so comparing the
    # provenance against the live attestation -- which is what the previous
    # version did -- could never succeed. Each is checked against the
    # authorization the owner signed for its own role.
    producer = record["producer_attestation"]
    assert isinstance(producer, Mapping)
    if not attestation_matches_authorization(
        producer, producer_authorization
    ):
        raise Exp1bBridgeError(
            "Substitute provenance was produced under a runtime the admission "
            "did not authorize for the producer role."
        )
    evaluator = role_attestation_document(
        role=THEORY_BRIDGE_ROLE,
        attestation={
            "source_git_commit": attestation.source_git_commit,
            "runtime_sha256": attestation.runtime_sha256,
            "launcher_sha256": attestation.launcher_sha256,
            "runtime_authorization_sha256": attestation.runtime_authorization_sha256,
        },
    )
    if not attestation_matches_authorization(evaluator, evaluator_authorization):
        raise Exp1bBridgeError(
            "The live Experiment 1B evaluator runtime is not the one the "
            "admission authorized for the theory-bridge role."
        )
    if producer["runtime_sha256"] == evaluator["runtime_sha256"]:
        raise Exp1bBridgeError(
            "Producer and evaluator runtimes must be distinct artifacts."
        )

    # The barrier the review found missing: prove the artifacts exist, and that
    # they were produced under the registered configuration on the registered
    # ordered train split -- not merely that eight files are present.
    registered_config = checked_protocol["effective_config"]
    assert isinstance(registered_config, Mapping)
    registered_training = checked_protocol["training_population"]
    assert isinstance(registered_training, Mapping)
    try:
        checkpoints = authenticate_sealed_octet_checkpoints(
            generation=generation,
            provenance=record,
            registry_rows=registry_rows,
            effective_config_sha256=str(
                registered_config["effective_config_sha256"]
            ),
            train_ordered_record_sha256=str(
                registered_training["ordered_record_sha256"]
            ),
        )
        base_digest = authenticate_base_artifact(
            generation=generation, provenance=record
        )
    except Exp1bEvidenceError as exc:
        raise Exp1bBridgeError(str(exc)) from exc
    base_descriptor = record["base_policy_artifact"]
    assert isinstance(base_descriptor, Mapping)
    if base_descriptor["checkpoint_sha256"] != base_digest:
        raise Exp1bBridgeError("Admitted base artifact digest differs.")

    populations_map = (
        populations.get("populations") if isinstance(populations, Mapping) else None
    )
    population = (
        populations_map.get(EVALUATION_POPULATION_ID)
        if isinstance(populations_map, Mapping)
        else None
    )
    if not isinstance(population, Mapping):
        raise Exp1bBridgeError("Registered validation_bridge population is missing.")
    # The registered derivation, not a caller's. This used to be a
    # ``state_id_for`` parameter on this factory, and the digests checked below
    # -- the ordered-record and population-binding digests -- do not cover the
    # identifiers the callback produces. So an alternate derivation opened the
    # route in `sealed_octet_complete`, served all eight requests, and advanced
    # the exact-once state machine irreversibly; publication then refused
    # because the in-process auditor derives the canonical ordering, and
    # reopening under the registered derivation refused every persisted payload.
    # There was no supported retry: the eight durable slots were spent.
    census = census_from_population(population, state_id_for=exp1b_state_id_for)
    _require_registered_state_ids(census)

    protocol_population = checked_protocol["evaluation_population"]
    assert isinstance(protocol_population, Mapping)
    if (
        protocol_population["ordered_record_sha256"] != census.ordered_record_sha256
        or protocol_population["binding_sha256"] != census.binding_sha256
    ):
        raise Exp1bBridgeError(
            "Experiment 1B protocol cites a different validation_bridge census."
        )
    if (
        record["ordered_population_sha256"] != census.ordered_record_sha256
        or record["population_binding_sha256"] != census.binding_sha256
    ):
        raise Exp1bBridgeError(
            "Substitute provenance cites a different validation_bridge census."
        )

    schedule = Exp1bRequestSchedule(
        record, generation=generation, registry_rows=registry_rows
    )
    if record["request_schedule_sha256"] != schedule.schedule_sha256():
        raise Exp1bBridgeError(
            "Substitute provenance cites a different request schedule."
        )
    if record["access_state"] != "sealed_octet_complete":
        raise Exp1bBridgeError(
            "Experiment 1B route opens only from sealed_octet_complete; "
            f"provenance records {record['access_state']!r}."
        )
    return Exp1bBridgeRoute(
        _issued_by=_ISSUANCE,
        provenance=record,
        evaluator_attestation=evaluator,
        schedule=schedule,
        census=census,
        population_registry=populations,
        generation=generation,
        checkpoints=checkpoints,
        attestation=attestation,
        exp1b_protocol_sha256=protocol_sha,
        exp1b_registry_sha256=registry_sha,
        exp1b_amendment_sha256=amendment_sha,
    )


#: Module-private token for a seed claim. Only `claim_seed` mints one, so a
#: caller cannot hand-build a handle and emit a payload without a durable claim.
_CLAIM_ISSUANCE = object()


@dataclass
class Exp1bSeedClaim:
    """One durably claimed seed slot, held for the whole evaluation."""

    _issued_by: Any
    route: "Exp1bBridgeRoute"
    slot: Any
    emitted: bool = False

    def __post_init__(self) -> None:
        if self._issued_by is not _CLAIM_ISSUANCE:
            raise Exp1bBridgeError(
                "Experiment 1B seed claims are issued by claim_seed."
            )

    def serve(
        self,
        *,
        endpoint_values: Callable[[str, int], float],
        action_values: Callable[[str, int], Sequence[float]],
        action_mask: Callable[[str], Sequence[bool]],
        base_probabilities: Callable[[str], Sequence[float]],
        secondary_diagnostics: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    ) -> Exp1bSeedPayload:
        """Evaluate the census and emit, under an already-durable claim.

        ``secondary_diagnostics`` may be a callable so the adapter's 128-member
        traversal happens here rather than during backend preparation. That is
        the point of the split: the traversal is evaluator work and must sit
        inside the claim.
        """

        if self.emitted:
            raise Exp1bBridgeError("This Experiment 1B seed claim already emitted.")
        route = self.route
        slot = self.slot
        computed = (
            secondary_diagnostics()
            if callable(secondary_diagnostics)
            else secondary_diagnostics
        )
        try:
            checked_secondary = validate_secondary_diagnostics(
                _plain_mapping(computed),
                path="exp1b_seed_payload.secondary_diagnostics",
            )
        except Exp1bSchemaError as exc:
            raise Exp1bBridgeError(
                f"Seed {slot.seed_position} secondary diagnostics are invalid: {exc}"
            ) from exc
        # Every witness has to name a state this seed actually evaluated. The
        # schema checks these six fields as nonempty strings and nothing more, so
        # a payload could carry provenance pointers that resolve to nothing --
        # which is exactly what the fixtures were doing, and neither the
        # publication gate nor the auditor noticed. The real Torch adapter does
        # derive them from census members; this is the gate that makes that a
        # requirement rather than a convention.
        _require_census_witnesses(
            checked_secondary,
            census=route._census,
            label=f"Seed {slot.seed_position}",
        )
        diagnostics = []
        for member in route._census.members:
            observation = PairedEndpointObservation(
                state_id=member.state_id,
                record_index=member.record_index,
                dataset_record_sha256=member.dataset_record_sha256,
                snapshot_id=slot.checkpoint_sha256,
                deployed_depth_n=DEPLOYED_DEPTH_N,
                reference_depth_m=REFERENCE_DEPTH_M,
                action_mask=action_mask(member.state_id),
                base_probabilities=base_probabilities(member.state_id),
                endpoint_value_n=endpoint_values(member.state_id, DEPLOYED_DEPTH_N),
                endpoint_value_m=endpoint_values(member.state_id, REFERENCE_DEPTH_M),
                action_values_n=action_values(member.state_id, DEPLOYED_DEPTH_N),
                action_values_m=action_values(member.state_id, REFERENCE_DEPTH_M),
                gamma=GAMMA,
            )
            diagnostics.append(paired_state_diagnostic(observation))
        summary = paired_population_summary(
            snapshot_id=slot.checkpoint_sha256,
            population_id=EVALUATION_POPULATION_ID,
            deployed_depth_n=DEPLOYED_DEPTH_N,
            reference_depth_m=REFERENCE_DEPTH_M,
            expected_population=route._census.members,
            state_diagnostics=diagnostics,
        )
        payload = Exp1bSeedPayload(
            seed_position=slot.seed_position,
            seed=slot.seed,
            run_id=slot.run_id,
            checkpoint_sha256=slot.checkpoint_sha256,
            summary=summary,
            census_ordering_sha256=route._census.ordering_sha256(),
            secondary_diagnostics=MappingProxyType(checked_secondary),
            _route_id=str(route._provenance["route_id"]),
        )
        document = route._payload_document(payload)
        route._schedule.finalize(
            seed_position=slot.seed_position,
            payload_document=document,
            claim_epoch=slot.claim_epoch,
        )
        self.emitted = True
        return payload


class Exp1bBridgeRoute:
    """The authenticated Experiment 1B validation-bridge route.

    Construct through :func:`open_authenticated_exp1b_route`. The constructor
    requires a module-private token object, not a boolean, so there is no value
    a caller can pass to satisfy it.
    """

    def __init__(
        self,
        *,
        provenance: Mapping[str, Any],
        schedule: Exp1bRequestSchedule,
        census: Exp1bCensus,
        population_registry: Mapping[str, Any],
        generation: Exp1bEvidenceGeneration,
        checkpoints: Sequence[AuthenticatedCheckpoint],
        attestation: Exp1bRuntimeAttestation,
        evaluator_attestation: Mapping[str, Any],
        exp1b_protocol_sha256: str,
        exp1b_registry_sha256: str,
        exp1b_amendment_sha256: str,
        _issued_by: object = None,
    ) -> None:
        if _issued_by is not _ISSUANCE:
            raise Exp1bBridgeError(
                "Experiment 1B routes are issued by "
                "open_authenticated_exp1b_route, which authenticates its own "
                "trust anchors; direct construction is refused."
            )
        self._provenance = dict(provenance)
        self._schedule = schedule
        self._census = census
        self._population_registry = dict(population_registry)
        self._generation = generation
        self._checkpoints = tuple(checkpoints)
        self._attestation = attestation
        self._evaluator_attestation = dict(evaluator_attestation)
        self._protocol_sha = exp1b_protocol_sha256
        self._registry_sha = exp1b_registry_sha256
        self._amendment_sha = exp1b_amendment_sha256
        self._provenance_sha = exp1b_document_sha256(self._provenance)
        # Strong references, so a live route keeps the identities it issued
        # valid for the whole publication path.
        self._issued_octets: list[Exp1bSealedOctet] = []

    @property
    def access_state(self) -> str:
        return self._schedule.access_state

    @property
    def authenticated_checkpoints(self) -> tuple[AuthenticatedCheckpoint, ...]:
        """The eight checkpoints proved to exist before this route opened."""

        return self._checkpoints

    @property
    def census_ordering_sha256(self) -> str:
        return self._census.ordering_sha256()

    def reload_durable_state(self) -> dict[str, Any]:
        """Re-read everything the durable generation says, for publication.

        Called with ``generation.exclusive()`` already held. Reloads from disk,
        rebuilds all eight payloads from their canonical artifacts, and returns
        the digests the result must agree with. Nothing here is taken from the
        in-memory route: that is the point.
        """

        self._schedule.reload_locked()
        self._schedule.require_complete()
        payloads = tuple(
            self._reload_payload(position) for position in range(UNITS)
        )
        return {
            "generation_sha256": str(self._generation.generation_sha256),
            "provenance_sha256": self._provenance_sha,
            "schedule_sha256": self._schedule.schedule_sha256(),
            "payload_digests": tuple(self._schedule.payload_digests()),
            "attempt_counts": tuple(self._schedule.attempt_counts()),
            "payloads": payloads,
        }

    def attempt_counts(self) -> tuple[int, ...]:
        """Durable per-slot attempt counts, including dead-claim attempts."""

        return tuple(self._schedule.attempt_counts())

    @property
    def served_positions(self) -> tuple[int, ...]:
        """Durably served seed positions, for restart and resume decisions."""

        return tuple(self._schedule.served_positions)

    @property
    def provenance_document(self) -> Mapping[str, Any]:
        """The authenticated provenance, for the publication-time audit."""

        return MappingProxyType(dict(self._provenance))

    @property
    def population_registry_document(self) -> Mapping[str, Any]:
        """The parent population registry this route authenticated on open.

        Carried so the independent auditor can rederive the census member
        ordering instead of accepting ``census_ordering_sha256`` on the result's
        word. That digest is the route's *recomputation* over the ordered
        members, not the registered ordered-record digest the provenance holds
        under ``ordered_population_sha256``; the two are different quantities, so
        the provenance cannot be its anchor. The protocol's
        ``parent.population_registry_sha256`` is, and the protocol is itself
        pinned to the provenance.
        """

        return MappingProxyType(dict(self._population_registry))

    @property
    def provenance_sha256(self) -> str:
        """Canonical digest of the provenance this route authenticated on open.

        Exposed so publication can prove the provenance still on disk is this
        one. ``provenance_document`` is a cached in-memory copy, and a cached
        anchor is not an anchor: the file can be replaced or deleted after the
        route opens, and a stale route would still finalize a result claiming it.
        """

        return self._provenance_sha

    @property
    def evaluator_attestation(self) -> Mapping[str, Any]:
        """The live Stage B identity this route was opened and authenticated under.

        Exposed so the durable publication comparison can bind the result's
        evaluator identity to it. The standalone auditor checks that identity
        against the signed admission, but only when an admission is supplied, so
        route equality is the check that does not depend on a caller's argument.
        """

        return MappingProxyType(dict(self._evaluator_attestation))

    @property
    def provenance_admission_sha256(self) -> str:
        return str(self._provenance["admission_sha256"])

    @property
    def provenance_base_policy_artifact(self) -> Mapping[str, Any]:
        descriptor = self._provenance["base_policy_artifact"]
        assert isinstance(descriptor, Mapping)
        return MappingProxyType(dict(descriptor))

    @property
    def census_members(self) -> tuple[PopulationMember, ...]:
        """The registered census, in the order the route bound it."""

        return tuple(self._census.members)

    @property
    def exp1b_protocol_sha256(self) -> str:
        return self._protocol_sha

    @property
    def exp1b_registry_sha256(self) -> str:
        return self._registry_sha

    @property
    def exp1b_amendment_sha256(self) -> str:
        return self._amendment_sha

    @property
    def generation(self) -> Exp1bEvidenceGeneration:
        """The authenticated evidence generation this route is bound to.

        Exposed so the runtime publishes the result into the same generation the
        payloads came from, rather than into a separately supplied directory
        that nothing ties back to the octet.
        """

        return self._generation

    def _payload_document(self, payload: Exp1bSeedPayload) -> dict[str, Any]:
        """The canonical, reloadable form of one seed payload."""

        summary = payload.summary
        return {
            "schema_name": PAYLOAD_SCHEMA_NAME,
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "route_id": str(self._provenance["route_id"]),
            "seed_position": payload.seed_position,
            "seed": payload.seed,
            "run_id": payload.run_id,
            "checkpoint_sha256": payload.checkpoint_sha256,
            "census_ordering_sha256": payload.census_ordering_sha256,
            "state_count": summary.state_count,
            "maximum_endpoint_discrepancy": summary.maximum_endpoint_discrepancy.value,
            "maximum_absolute_residual_n": summary.maximum_absolute_residual_n.value,
            "maximum_absolute_residual_m": summary.maximum_absolute_residual_m.value,
            "direct_residual_bound_n": summary.direct_residual_bound_n,
            "finite_reference_bound": summary.finite_reference_bound,
            "signed_gap": summary.signed_gap,
            "discrepancy_witness_state_id": (
                summary.maximum_endpoint_discrepancy.state_id
            ),
            "residual_n_witness_state_id": (
                summary.maximum_absolute_residual_n.state_id
            ),
            "residual_m_witness_state_id": (
                summary.maximum_absolute_residual_m.state_id
            ),
            "secondary_diagnostics": _plain_mapping(payload.secondary_diagnostics),
        }

    @contextmanager
    def claim_seed(
        self,
        seed_position: int,
        *,
        seed: int,
        run_id: str,
        checkpoint_sha256: str,
        evaluation_population: str = EVALUATION_POPULATION_ID,
    ) -> Iterator["Exp1bSeedClaim"]:
        """Take the durable request claim **before** any evaluator work.

        The claim-before-evaluator contract used to be aspirational: the runtime
        constructed the Stage B backend, restored the checkpoint, and eagerly
        traversed all 128 census members before the schedule saw the request at
        all. An out-of-order or concurrent request therefore completed a full
        traversal before being refused, and a malformed record left the slot at
        ``sealed_octet_complete`` with attempt count 0.

        The lease is taken first and held for the whole evaluation. It is an
        ``flock``, so the kernel releases it if this process dies, which is what
        lets a later process prove the claim is a corpse and reclaim it.
        """

        if evaluation_population in _FORBIDDEN_POPULATIONS:
            raise Exp1bBridgeError(
                "Experiment 1B route never opens validation_select or test data."
            )
        lease = self._generation.claim_lease(seed_position)
        try:
            lease.__enter__()
        except Exp1bEvidenceError as exc:
            raise Exp1bBridgeError(str(exc)) from exc
        try:
            # Holding the lease is the liveness proof: the kernel released it if
            # the previous owner died, so acquiring it means a dead claim on this
            # slot can be safely reclaimed, attempt count preserved.
            slot = self._schedule.claim(
                seed_position=seed_position,
                seed=seed,
                run_id=run_id,
                checkpoint_sha256=checkpoint_sha256,
                evaluation_population=evaluation_population,
                depths=(DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M),
                lease_acquired=True,
            )
            claim = Exp1bSeedClaim(
                _issued_by=_CLAIM_ISSUANCE, route=self, slot=slot
            )
            try:
                yield claim
            except BaseException:
                # Pre-emission failure: release the claim, keep the attempt.
                if not claim.emitted:
                    self._schedule.abandon_claim(
                        seed_position=slot.seed_position,
                        claim_epoch=slot.claim_epoch,
                    )
                raise
        finally:
            lease.__exit__(None, None, None)

    def serve(
        self,
        *,
        seed_position: int,
        seed: int,
        run_id: str,
        checkpoint_sha256: str,
        evaluation_population: str,
        endpoint_values: Callable[[str, int], float],
        action_values: Callable[[str, int], Sequence[float]],
        action_mask: Callable[[str], Sequence[bool]],
        base_probabilities: Callable[[str], Sequence[float]],
        secondary_diagnostics: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    ) -> Exp1bSeedPayload:
        """Claim, then evaluate, then emit. One traversal of the fixed census.

        A thin wrapper over :meth:`claim_seed` that keeps the historical
        signature. Everything the evaluator does now happens inside the claim,
        so a request that the schedule would refuse is refused before any
        traversal, and a request that fails mid-evaluation leaves a durable,
        counted, retryable attempt.
        """

        with self.claim_seed(
            seed_position,
            seed=seed,
            run_id=run_id,
            checkpoint_sha256=checkpoint_sha256,
            evaluation_population=evaluation_population,
        ) as claim:
            return claim.serve(
                endpoint_values=endpoint_values,
                action_values=action_values,
                action_mask=action_mask,
                base_probabilities=base_probabilities,
                secondary_diagnostics=secondary_diagnostics,
            )

    def _reload_payload(self, seed_position: int) -> Exp1bSeedPayload:
        """Rebuild one served payload from its durable artifact.

        This is what makes the documented one-process-per-seed flow work: the
        eighth process reconstructs the first seven from disk instead of
        requiring them to be in its own memory.
        """

        try:
            document, digest = self._generation.read_payload(seed_position)
        except Exp1bEvidenceError as exc:
            raise Exp1bBridgeError(str(exc)) from exc
        expected = self._schedule.payload_digests()
        if seed_position >= len(expected) or expected[seed_position] != digest:
            raise Exp1bBridgeError(
                f"Persisted payload {seed_position} does not match the durable "
                "schedule digest."
            )
        slot_descriptor = self._provenance["sealed_checkpoints"][seed_position]
        assert isinstance(slot_descriptor, Mapping)
        if document["route_id"] != BRIDGE_ROUTE_ID:
            raise Exp1bBridgeError(
                f"Persisted payload {seed_position} was produced under route "
                f"{document['route_id']!r}, not the registered Experiment 1B route."
            )
        if (
            document["seed_position"] != seed_position
            or document["seed"] != slot_descriptor["seed"]
            or document["run_id"] != slot_descriptor["run_id"]
            or document["checkpoint_sha256"] != slot_descriptor["checkpoint_sha256"]
        ):
            raise Exp1bBridgeError(
                f"Persisted payload {seed_position} identity differs from provenance."
            )
        if document["census_ordering_sha256"] != self._census.ordering_sha256():
            raise Exp1bBridgeError(
                f"Persisted payload {seed_position} used a different census."
            )
        if document["state_count"] != EVALUATION_RECORD_COUNT:
            raise Exp1bBridgeError(
                f"Persisted payload {seed_position} does not cover the census."
            )
        return Exp1bSeedPayload(
            seed_position=int(document["seed_position"]),
            seed=int(document["seed"]),
            run_id=str(document["run_id"]),
            checkpoint_sha256=str(document["checkpoint_sha256"]),
            summary=None,
            census_ordering_sha256=str(document["census_ordering_sha256"]),
            secondary_diagnostics=MappingProxyType(
                dict(document["secondary_diagnostics"])
            ),
            persisted=_PersistedSeedResult(
                state_count=int(document["state_count"]),
                maximum_endpoint_discrepancy=float(
                    document["maximum_endpoint_discrepancy"]
                ),
                maximum_absolute_residual_n=float(
                    document["maximum_absolute_residual_n"]
                ),
                maximum_absolute_residual_m=float(
                    document["maximum_absolute_residual_m"]
                ),
                direct_residual_bound_n=float(document["direct_residual_bound_n"]),
                finite_reference_bound=float(document["finite_reference_bound"]),
                signed_gap=float(document["signed_gap"]),
                discrepancy_witness_state_id=str(
                    document["discrepancy_witness_state_id"]
                ),
                residual_n_witness_state_id=str(
                    document["residual_n_witness_state_id"]
                ),
                residual_m_witness_state_id=str(
                    document["residual_m_witness_state_id"]
                ),
                route_id=str(document["route_id"]),
                secondary_diagnostics=validate_secondary_diagnostics(
                    document["secondary_diagnostics"],
                    path=f"persisted payload {seed_position}.secondary_diagnostics",
                ),
                payload_sha256=digest,
            ),
        )

    def sealed_octet(self) -> Exp1bSealedOctet:
        """Issue the completed octet, rebuilding every payload from disk."""

        self._schedule.require_complete()
        payloads = tuple(
            self._reload_payload(position) for position in range(UNITS)
        )
        if tuple(payload.seed for payload in payloads) != REGISTERED_SEEDS:
            raise Exp1bBridgeError(
                "Experiment 1B payloads are not in registered seed order."
            )
        configs = {item.effective_config_sha256 for item in self._checkpoints}
        if len(configs) != 1:
            raise Exp1bBridgeError("Sealed octet spans more than one effective config.")
        octet = Exp1bSealedOctet(
            _issued_by=_OCTET_ISSUANCE,
            route_id=str(self._provenance["route_id"]),
            producer_attestation=MappingProxyType(
                dict(self._provenance["producer_attestation"])
            ),
            evaluator_attestation=MappingProxyType(
                dict(self._evaluator_attestation)
            ),
            provenance=MappingProxyType(dict(self._provenance)),
            provenance_sha256=self._provenance_sha,
            schedule_sha256=self._schedule.schedule_sha256(),
            generation_sha256=str(self._generation.generation_sha256),
            attempt_counts=self._schedule.attempt_counts(),
            payload_digests=self._schedule.payload_digests(),
            census_ordering_sha256=self._census.ordering_sha256(),
            census_binding_sha256=self._census.binding_sha256,
            exp1b_protocol_sha256=self._protocol_sha,
            exp1b_registry_sha256=self._registry_sha,
            exp1b_amendment_sha256=self._amendment_sha,
            effective_config_sha256=next(iter(configs)),
            authenticated_checkpoints=self._checkpoints,
            runtime_attestation=self._attestation,
            payloads=payloads,
        )
        _ISSUED_OCTETS[id(octet)] = octet
        self._issued_octets.append(octet)
        return octet
