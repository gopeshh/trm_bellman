#!/usr/bin/env fbpython
"""Independent consumer for a published Experiment 1B result.

This is the audit side of the slice, and it is deliberately *not* built out of
the producer's objects. It imports neither ``policy_improvement_exp1b_aggregate``
nor ``policy_improvement_exp1b_bridge``: it takes the published canonical bytes
and the registered study documents, and re-derives every published number from
them.

What "independent" buys
-----------------------
The per-seed algebra is recomputed here from the paper contract directly rather
than by calling ``policy_improvement_exp1_diagnostics``:

    R_hat_n = eps_n / (1 - gamma)
    B_hat   = D_hat + eps_m / (1 - gamma)
    G       = R_hat_n - B_hat

with ``K = 1``, so ``1 - gamma**K`` is bitwise ``1 - gamma``. Those three lines
are the whole claim. If the producing core ever regressed to, say, a maximum of
recordwise sums, the published maxima would no longer reproduce the published
bound, and this consumer would say so. A consumer that called the same core
would agree with the bug.

The bootstrap is the one thing recomputed *through* the shared module. That is
intentional and not a hole: D2 freezes one namespace, one seed, one draw order
and one replicate count, so "independent" would mean "a second implementation of
a bit-exact spec", which is a second thing to keep in sync, not a second opinion.
What this consumer checks instead is that the published vector reproduces the
published digest and the published endpoints, and that the observed centre
equals the mean it recomputed itself.

Access posture
--------------
Reads a result document, the four registered study documents, and -- when the
caller supplies it -- the owner-signed execution admission. Nothing else. It
opens no checkpoint, no dataset, and no evidence payload; it needs none of them,
because everything it verifies is already in the documents it is handed.

The admission is optional so an outside reviewer who does not hold the signed
document can still audit a published result in full. Without it the producer and
evaluator identities are checked for shape, for agreement with the provenance,
and for being distinct artifacts, but not against the owner's authorizations;
the report says so by omitting ``role_scoped_admission_binding`` from its checks.
The publication gate always supplies it.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.policy_improvement_exp1b_bootstrap import (
    _exact_mean,
    BOOTSTRAP_RNG_NAMESPACE,
    BOOTSTRAP_SCHEME,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    decode_replicate_vector,
    Exp1bBootstrapError,
    INTERVAL_CONVENTION,
    LOWER_PROBABILITY,
    paired_seed_percentile_interval,
    REPLICATE_ENCODING,
    REPLICATES,
    replicate_digest,
    UNITS,
    UPPER_PROBABILITY,
)
from scripts.policy_improvement_exp1b_schema import (
    BELLMAN_HORIZON_K,
    BRIDGE_ROUTE_ID,
    DEPLOYED_DEPTH_N,
    EVALUATION_POPULATION_ID,
    EVALUATION_RECORD_COUNT,
    exp1b_state_id_for,
    attestation_matches_authorization,
    Exp1bSchemaError,
    exp1b_document_sha256,
    FULL_ROLE,
    GAMMA,
    PROTOCOL_ID,
    REFERENCE_DEPTH_M,
    REGISTERED_SEEDS,
    SECONDARY_DIAGNOSTIC_NAMES,
    TERMINAL_ENVIRONMENT_INTERACTIONS,
    THEORY_BRIDGE_ROLE,
    validate_exp1b_admission,
    validate_exp1b_amendment,
    validate_exp1b_protocol,
    validate_exp1b_registry,
    validate_secondary_diagnostics,
    validate_substitute_provenance,
)
from scripts.policy_improvement_schema import (
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
)

__all__ = [
    "Exp1bAuditError",
    "Exp1bAuditReport",
    "audit_exp1b_result_document",
    "main",
]

RESULT_SCHEMA_NAME = "policy_improvement_exp1b_result_v1"
RESULT_SCHEMA_VERSION = 2

_RESULT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "protocol_id",
        "route_id",
        "evaluation_population",
        "deployed_depth_n",
        "reference_depth_m",
        "bellman_horizon",
        "gamma",
        "terminal_environment_interactions",
        "seed_count",
        "seed_rows",
        "signed_gaps",
        "mean_signed_gap",
        "interval",
        "generation_sha256",
        "effective_config_sha256",
        "payload_digests",
        "provenance_sha256",
        "schedule_sha256",
        "attempt_counts",
        "census_ordering_sha256",
        "census_binding_sha256",
        "exp1b_protocol_sha256",
        "exp1b_registry_sha256",
        "exp1b_amendment_sha256",
        "producer_attestation",
        "evaluator_attestation",
        "scientific_selection",
        "validation_select_access",
        "test_access",
    }
)

_ROW_FIELDS = frozenset(
    {
        "seed_position",
        "seed",
        "run_id",
        "checkpoint_sha256",
        "state_count",
        "maximum_endpoint_discrepancy",
        "maximum_absolute_residual_n",
        "maximum_absolute_residual_m",
        "direct_residual_bound_n",
        "finite_reference_bound",
        "signed_gap",
        "discrepancy_witness_state_id",
        "residual_n_witness_state_id",
        "residual_m_witness_state_id",
        "secondary_diagnostics",
    }
)

_INTERVAL_FIELDS = frozenset(
    {
        "status",
        "scheme",
        "interval_convention",
        "replicates",
        "units",
        "bootstrap_seed",
        "rng_namespace",
        "confidence_level",
        "lower_probability",
        "upper_probability",
        "interval_lower",
        "interval_upper",
        "replicate_encoding",
        "replicate_means_hex",
        "replicate_vector_sha256",
        "resample_plan_sha256",
    }
)

_HEX = frozenset("0123456789abcdef")


class Exp1bAuditError(ValueError):
    """Raised when a published Experiment 1B result fails independent audit."""


@dataclass(frozen=True)
class Exp1bAuditReport:
    """What the audit re-derived, for the caller to print or assert on."""

    document_sha256: str
    protocol_sha256: str
    registry_sha256: str
    amendment_sha256: str
    provenance_sha256: str
    recomputed_mean_signed_gap: float
    recomputed_interval_lower: float
    recomputed_interval_upper: float
    recomputed_replicate_vector_sha256: str
    checks: tuple[str, ...]


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Exp1bAuditError(f"{label} must be an object.")
    return value


def _exact(value: object, expected: frozenset[str], *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != set(expected):
        raise Exp1bAuditError(
            f"{label} field inventory differs: {sorted(set(result) ^ set(expected))!r}."
        )
    return result


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise Exp1bAuditError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _real(value: object, *, label: str) -> float:
    """A published real must be a JSON float, not an int and not a bool.

    ``1`` and ``1.0`` are equal in Python and distinct in canonical JSON, so a
    document carrying integer-typed reals would round-trip to a different digest
    than the one it claims. Reject the type, not just the value.
    """

    if isinstance(value, bool) or not isinstance(value, float):
        raise Exp1bAuditError(f"{label} must be a binary64 float.")
    if not math.isfinite(value):
        raise Exp1bAuditError(f"{label} must be finite.")
    return value


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Exp1bAuditError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise Exp1bAuditError(f"{label} is below its minimum.")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise Exp1bAuditError(f"{label} must be nonempty ASCII.")
    return value


def _false(value: object, *, label: str) -> None:
    if value is not False:
        raise Exp1bAuditError(f"{label} must be exactly false.")


def _role_attestation(
    value: object, *, expected_role: str, label: str
) -> dict[str, Any]:
    attestation = _exact(
        value,
        frozenset(
            {
                "role",
                "source_git_commit",
                "runtime_sha256",
                "launcher_sha256",
                "runtime_authorization_sha256",
            }
        ),
        label=label,
    )
    if attestation["role"] != expected_role:
        raise Exp1bAuditError(f"{label} is not the {expected_role!r} identity.")
    commit = attestation["source_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in _HEX for character in commit)
    ):
        raise Exp1bAuditError(f"{label}.source_git_commit is not a commit.")
    for field in (
        "runtime_sha256",
        "launcher_sha256",
        "runtime_authorization_sha256",
    ):
        _digest(attestation[field], label=f"{label}.{field}")
    return dict(attestation)


#: The six secondary witness fields, by the record that carries each. The route
#: gate uses the same list (``scripts/policy_improvement_exp1b_bridge.py``); a
#: consistency test pins the two together. Duplicated rather than imported for
#: the same reason every other derivation here is: this module imports no
#: producer module.
SECONDARY_WITNESS_FIELDS: tuple[tuple[str, str], ...] = (
    ("target_lag", "target_lag_witness_state_id"),
    ("centering_parity", "centering_witness_state_id"),
    ("centering_parity", "parity_witness_state_id"),
    ("centering_parity", "centering_defect_witness_state_id"),
    ("mixture_identity", "identity_witness_state_id"),
    ("deployment_mismatch", "mismatch_witness_state_id"),
)


def _audit_census_ordering(
    result: Mapping[str, object],
    checked_protocol: Mapping[str, object],
    population_registry: Mapping[str, object],
) -> tuple[str, frozenset[str]]:
    """Rederive the census member ordering and require the result to name it.

    ``census_ordering_sha256`` is the *recomputed member ordering*, not the
    registered ordered-record digest the provenance carries under
    ``ordered_population_sha256``. Two different quantities, which is why the
    provenance cannot be its anchor and why this field was previously checked for
    digest shape alone: any other valid digest audited clean.

    Its durable source of truth is the parent population registry. The protocol
    pins that registry's digest under ``parent.population_registry_sha256``, and
    the provenance pins the protocol, so the chain terminates in the durable
    document publication read off the disk.

    The derivation is written out here rather than imported from the bridge. That
    is not duplication by accident: this module imports no producer module, and
    ``test_the_consumer_imports_no_producer_module`` enforces it, because an audit
    that shares the producer's objects agrees with the producer's bugs. It also
    means the state identifiers are derived with the *registered*
    ``exp1b_state_id_for`` rather than whatever callable a route was opened with,
    so a route opened under a non-registered derivation produces a result this
    auditor refuses.

    Optional on the standalone auditor: an outside reviewer may hold the result
    and the four study documents without the parent registry. Not optional on the
    publication path, which passes the registry the route authenticated and
    separately binds the field to ``route.census_ordering_sha256``.
    """

    parent = checked_protocol["parent"]
    if not isinstance(parent, Mapping):
        raise Exp1bAuditError("Registered protocol carries no parent block.")
    expected_registry_sha256 = _digest(
        parent["population_registry_sha256"],
        label="exp1b_protocol.parent.population_registry_sha256",
    )
    if exp1b_document_sha256(population_registry) != expected_registry_sha256:
        raise Exp1bAuditError(
            "Supplied population registry is not the one the registered protocol "
            "cites; it cannot anchor the census ordering."
        )
    populations = population_registry.get("populations")
    population = (
        populations.get(EVALUATION_POPULATION_ID)
        if isinstance(populations, Mapping)
        else None
    )
    if not isinstance(population, Mapping):
        raise Exp1bAuditError(
            "Population registry holds no registered validation_bridge population."
        )
    if population.get("split") != "validation":
        raise Exp1bAuditError("Registered census population is not the validation split.")
    indices = population.get("indices")
    digests = population.get("record_sha256s")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes))
        or not isinstance(digests, Sequence)
        or isinstance(digests, (str, bytes))
        or len(indices) != len(digests)
        or len(indices) != EVALUATION_RECORD_COUNT
    ):
        raise Exp1bAuditError(
            "Registered validation_bridge population does not carry the ordered "
            f"{EVALUATION_RECORD_COUNT}-record census."
        )
    try:
        members = [
            {
                "state_id": exp1b_state_id_for(int(index), str(digest)),
                "record_index": int(index),
                "dataset_record_sha256": str(digest),
            }
            for index, digest in zip(indices, digests)
        ]
    except Exp1bSchemaError as exc:
        raise Exp1bAuditError(
            f"Registered census record identities are not well formed: {exc}"
        ) from exc
    protocol_population = checked_protocol["evaluation_population"]
    if not isinstance(protocol_population, Mapping):
        raise Exp1bAuditError("Registered protocol carries no evaluation population.")
    for field in ("ordered_record_sha256", "binding_sha256"):
        if population.get(field) != protocol_population.get(field):
            raise Exp1bAuditError(
                f"Registered population {field} differs from the protocol's "
                "evaluation population."
            )
    if result["census_ordering_sha256"] != exp1b_document_sha256(members):
        raise Exp1bAuditError(
            "Published census_ordering_sha256 is not the ordering the registered "
            "validation_bridge population produces."
        )
    return (
        "census_member_ordering_rederived_from_registered_population",
        frozenset(member["state_id"] for member in members),
    )


def audit_exp1b_result_document(
    document_bytes: bytes,
    *,
    protocol: Mapping[str, object],
    registry: Mapping[str, object],
    amendment: Mapping[str, object],
    provenance: Mapping[str, object],
    population_registry: Mapping[str, object],
    admission: Mapping[str, object] | None = None,
) -> Exp1bAuditReport:
    """Independently verify one published Experiment 1B result.

    ``document_bytes`` is exactly what was written, trailing newline included.
    The audit re-parses it strictly, re-derives every bound from the published
    maxima, recomputes the equal-seed mean and the frozen bootstrap interval, and
    checks every identity binding against the four study documents.
    """

    if not isinstance(document_bytes, (bytes, bytearray)):
        raise Exp1bAuditError("Experiment 1B audit requires the published bytes.")
    payload = bytes(document_bytes)
    if payload.endswith(b"\n"):
        payload = payload[:-1]
    try:
        parsed = load_strict_json_bytes(payload)
    except PolicyImprovementSchemaError as exc:
        raise Exp1bAuditError(f"Published result is not strict JSON: {exc}") from exc

    checks: list[str] = []
    result = _exact(parsed, _RESULT_FIELDS, label="exp1b_result")

    # The document must hash to itself under the namespace convention. This is
    # what a downstream citation would quote.
    document_sha256 = exp1b_document_sha256(result)
    checks.append("document_canonical_digest")

    # --- registered identity -------------------------------------------------
    if (
        result["schema_name"] != RESULT_SCHEMA_NAME
        or _integer(result["schema_version"], label="schema_version")
        != RESULT_SCHEMA_VERSION
        or result["protocol_id"] != PROTOCOL_ID
        or result["route_id"] != BRIDGE_ROUTE_ID
        or result["evaluation_population"] != EVALUATION_POPULATION_ID
    ):
        raise Exp1bAuditError("Published result identity is not the registered study.")
    if (
        _integer(result["deployed_depth_n"], label="deployed_depth_n") != DEPLOYED_DEPTH_N
        or _integer(result["reference_depth_m"], label="reference_depth_m")
        != REFERENCE_DEPTH_M
        or _integer(result["bellman_horizon"], label="bellman_horizon")
        != BELLMAN_HORIZON_K
        or _integer(
            result["terminal_environment_interactions"],
            label="terminal_environment_interactions",
        )
        != TERMINAL_ENVIRONMENT_INTERACTIONS
        or _integer(result["seed_count"], label="seed_count") != UNITS
    ):
        raise Exp1bAuditError("Published result study settings differ.")
    gamma = _real(result["gamma"], label="gamma")
    if gamma != GAMMA:
        raise Exp1bAuditError("Published result uses a different discount factor.")
    for name in ("scientific_selection", "validation_select_access", "test_access"):
        _false(result[name], label=f"exp1b_result.{name}")
    checks.append("registered_identity_and_access_flags")

    # --- study-document bindings --------------------------------------------
    checked_protocol = validate_exp1b_protocol(protocol)
    protocol_sha256 = exp1b_document_sha256(protocol)
    validate_exp1b_registry(registry, protocol_sha256=protocol_sha256)
    registry_sha256 = exp1b_document_sha256(registry)
    validate_exp1b_amendment(
        amendment, protocol_sha256=protocol_sha256, registry_sha256=registry_sha256
    )
    amendment_sha256 = exp1b_document_sha256(amendment)
    checked_provenance = validate_substitute_provenance(provenance)
    provenance_sha256 = exp1b_document_sha256(provenance)

    for field, expected in (
        ("exp1b_protocol_sha256", protocol_sha256),
        ("exp1b_registry_sha256", registry_sha256),
        ("exp1b_amendment_sha256", amendment_sha256),
        ("provenance_sha256", provenance_sha256),
    ):
        if _digest(result[field], label=f"exp1b_result.{field}") != expected:
            raise Exp1bAuditError(
                f"Published {field} does not match the supplied study document."
            )
    # The check above pins the three study digests to the *supplied* documents,
    # which the caller chooses. On its own that is caller-versus-caller: a result
    # and a protocol forged together agree with each other and with nothing else.
    # The provenance is the durable anchor -- publication reads it off the
    # authenticated route, out of the evidence generation -- and it records which
    # study documents the octet was finalized against, so bind to that too.
    for field in (
        "exp1b_protocol_sha256",
        "exp1b_registry_sha256",
        "exp1b_amendment_sha256",
    ):
        if result[field] != checked_provenance[field]:
            raise Exp1bAuditError(
                f"Published {field} is not the one the authenticated provenance "
                "was finalized against."
            )
    for field, source in (
        ("census_binding_sha256", "population_binding_sha256"),
        ("schedule_sha256", "request_schedule_sha256"),
    ):
        if result[field] != checked_provenance[source]:
            raise Exp1bAuditError(
                f"Published {field} differs from the substitute provenance."
            )
    for field in ("generation_sha256", "census_ordering_sha256"):
        _digest(result[field], label=f"exp1b_result.{field}")
    # `census_ordering_sha256` is the *recomputed member ordering*, not the
    # registered ordered-record digest the provenance carries under
    # `ordered_population_sha256`. Two different quantities, so the provenance
    # cannot be its anchor and the field was previously checked for digest shape
    # alone -- any other valid digest audited clean. Its durable source of truth
    # is the parent population registry, whose digest the protocol pins under
    # `parent.population_registry_sha256`, and the protocol is itself pinned to
    # the provenance a few checks above. Rederive with the same sanctioned
    # constructor the route uses; a second copy of the derivation could drift.
    #
    # Mandatory, not optional. Optional meant the packaged CLI -- which had no
    # argument for it and never supplied it -- skipped the one check it
    # advertises: a document with only `census_ordering_sha256` changed printed a
    # successful audit and returned 0. An audit that silently omits a check it
    # claims to perform is worse than one that refuses to run.
    census_check, census_members = _audit_census_ordering(
        result, checked_protocol, population_registry
    )
    checks.append(census_check)
    # Exact equality, not merely digest syntax. The provenance names the one
    # evidence generation it was finalized into
    # (scripts/policy_improvement_exp1b_schema.py, provenance schema), so a
    # result claiming a different generation is a result about other evidence.
    # Checking only the shape let a document whose *only* change was this field
    # pass the audit.
    if result["generation_sha256"] != checked_provenance["generation_sha256"]:
        raise Exp1bAuditError(
            "Published generation digest is not the one the authenticated "
            "provenance was finalized into."
        )

    # Two roles, audited separately. The producer identity must be exactly the
    # one the provenance was finalized under; the evaluator identity must be a
    # well-formed theory-bridge identity for a *different* runtime artifact.
    # Folding them into one set of flat fields is what made Stage A evidence
    # unopenable under Stage B in the first place.
    producer = _role_attestation(
        result["producer_attestation"],
        expected_role=FULL_ROLE,
        label="exp1b_result.producer_attestation",
    )
    evaluator = _role_attestation(
        result["evaluator_attestation"],
        expected_role=THEORY_BRIDGE_ROLE,
        label="exp1b_result.evaluator_attestation",
    )
    if producer != dict(checked_provenance["producer_attestation"]):
        raise Exp1bAuditError(
            "Published producer attestation differs from the authenticated "
            "provenance."
        )
    if producer["runtime_sha256"] == evaluator["runtime_sha256"]:
        raise Exp1bAuditError(
            "Published producer and evaluator runtimes are the same artifact; "
            "the two Experiment 1B stages run distinct PARs."
        )
    if admission is not None:
        checked_admission = validate_exp1b_admission(
            admission,
            protocol_sha256=protocol_sha256,
            registry_sha256=registry_sha256,
            prior_amendment_sha256=amendment_sha256,
        )
        if checked_admission.admission_sha256 != checked_provenance[
            "admission_sha256"
        ]:
            raise Exp1bAuditError(
                "Supplied admission is not the one the evidence was finalized "
                "under."
            )
        for role, observed in (
            (FULL_ROLE, producer),
            (THEORY_BRIDGE_ROLE, evaluator),
        ):
            if not attestation_matches_authorization(
                observed, checked_admission.authorization_for(role)
            ):
                raise Exp1bAuditError(
                    f"Published {role} identity is not the admitted one."
                )
        checks.append("role_scoped_admission_binding")
    effective = checked_protocol["effective_config"]
    assert isinstance(effective, Mapping)
    if result["effective_config_sha256"] != effective["effective_config_sha256"]:
        raise Exp1bAuditError(
            "Published effective-config digest is not the registered configuration."
        )
    checks.append("study_document_bindings")

    # --- per-seed algebra, recomputed from the paper contract ----------------
    rows = result["seed_rows"]
    if not isinstance(rows, list) or len(rows) != UNITS:
        raise Exp1bAuditError("Published result does not carry eight seed rows.")
    gaps = result["signed_gaps"]
    if not isinstance(gaps, list) or len(gaps) != UNITS:
        raise Exp1bAuditError("Published signed gaps have the wrong arity.")
    digests = result["payload_digests"]
    attempts = result["attempt_counts"]
    for name, item in (("payload_digests", digests), ("attempt_counts", attempts)):
        if not isinstance(item, list) or len(item) != UNITS:
            raise Exp1bAuditError(f"Published {name} has the wrong arity.")

    # K = 1, so the K-step denominator is bitwise the one-step denominator.
    denominator = 1.0 - gamma
    if denominator != 1.0 - gamma**BELLMAN_HORIZON_K:
        raise Exp1bAuditError("K=1 denominator identity does not hold.")

    recomputed_gaps: list[float] = []
    audited_secondary: list[Mapping[str, Any]] = []
    seen_runs: set[str] = set()
    seen_checkpoints: set[str] = set()
    registered_checkpoints = checked_provenance["sealed_checkpoints"]
    assert isinstance(registered_checkpoints, Sequence)
    for offset, raw in enumerate(rows):
        row = _exact(raw, _ROW_FIELDS, label=f"exp1b_result.seed_rows[{offset}]")
        label = f"seed_rows[{offset}]"
        position = _integer(row["seed_position"], label=f"{label}.seed_position")
        seed = _integer(row["seed"], label=f"{label}.seed")
        if position != offset or seed != REGISTERED_SEEDS[offset]:
            raise Exp1bAuditError(
                f"{label} is not the registered seed in its registered position."
            )
        run_id = _text(row["run_id"], label=f"{label}.run_id")
        checkpoint = _digest(row["checkpoint_sha256"], label=f"{label}.checkpoint_sha256")
        if run_id in seen_runs or checkpoint in seen_checkpoints:
            raise Exp1bAuditError(f"{label} repeats a run or checkpoint identity.")
        seen_runs.add(run_id)
        seen_checkpoints.add(checkpoint)
        registered = registered_checkpoints[offset]
        assert isinstance(registered, Mapping)
        if (
            registered["run_id"] != run_id
            or registered["seed"] != seed
            or registered["seed_position"] != position
            or registered["checkpoint_sha256"] != checkpoint
        ):
            raise Exp1bAuditError(
                f"{label} differs from the provenance's sealed checkpoint identity."
            )
        if (
            _integer(row["state_count"], label=f"{label}.state_count")
            != EVALUATION_RECORD_COUNT
        ):
            raise Exp1bAuditError(
                f"{label} does not cover the registered {EVALUATION_RECORD_COUNT}-record "
                "census."
            )
        for name in (
            "discrepancy_witness_state_id",
            "residual_n_witness_state_id",
            "residual_m_witness_state_id",
        ):
            if _text(row[name], label=f"{label}.{name}") not in census_members:
                raise Exp1bAuditError(
                    f"{label}.{name} names a state that is not a member of the "
                    "registered validation_bridge census."
                )
        # The five secondary checks, audited as separate records. They are not
        # inputs to any published number, which is the point: an auditor that
        # only re-derived the signed gap would never notice a seed whose mixture
        # identity or persistent carry had failed.
        secondary = validate_secondary_diagnostics(
            row["secondary_diagnostics"], path=f"{label}.secondary_diagnostics"
        )
        for name in SECONDARY_DIAGNOSTIC_NAMES:
            if name not in secondary:
                raise Exp1bAuditError(f"{label} omits the {name} diagnostic.")
        # All six, not four. The previous set omitted both centering
        # sub-witnesses, and it checked shape rather than membership, so a
        # published payload could point at states that resolve to nothing. The
        # field list is shared with the route gate so the two cannot check
        # different subsets.
        for record, field in SECONDARY_WITNESS_FIELDS:
            witness = secondary[record][field]
            if not isinstance(witness, str) or not witness:
                raise Exp1bAuditError(
                    f"{label}.secondary_diagnostics.{record}.{field} names no witness."
                )
            if witness not in census_members:
                raise Exp1bAuditError(
                    f"{label}.secondary_diagnostics.{record}.{field} names a state "
                    "that is not a member of the registered validation_bridge census."
                )
        audited_secondary.append(secondary)

        discrepancy = _real(
            row["maximum_endpoint_discrepancy"],
            label=f"{label}.maximum_endpoint_discrepancy",
        )
        residual_n = _real(
            row["maximum_absolute_residual_n"],
            label=f"{label}.maximum_absolute_residual_n",
        )
        residual_m = _real(
            row["maximum_absolute_residual_m"],
            label=f"{label}.maximum_absolute_residual_m",
        )
        if discrepancy < 0.0 or residual_n < 0.0 or residual_m < 0.0:
            raise Exp1bAuditError(f"{label} publishes a negative maximum.")

        # The three lines under audit. Separate maxima, then one sum.
        direct = residual_n / denominator
        finite_reference = discrepancy + residual_m / denominator
        signed_gap = direct - finite_reference
        for name, expected, observed in (
            (
                "direct_residual_bound_n",
                direct,
                _real(row["direct_residual_bound_n"], label=f"{label}.direct"),
            ),
            (
                "finite_reference_bound",
                finite_reference,
                _real(row["finite_reference_bound"], label=f"{label}.finite"),
            ),
            ("signed_gap", signed_gap, _real(row["signed_gap"], label=f"{label}.gap")),
        ):
            if observed.hex() != expected.hex():
                raise Exp1bAuditError(
                    f"{label}.{name} is not what its own published maxima imply."
                )
        published_gap = gaps[offset]
        if not isinstance(published_gap, float) or published_gap.hex() != signed_gap.hex():
            raise Exp1bAuditError(
                f"Published signed_gaps[{offset}] differs from its own seed row."
            )
        recomputed_gaps.append(signed_gap)
        _digest(digests[offset], label=f"payload_digests[{offset}]")
        _integer(attempts[offset], label=f"attempt_counts[{offset}]", minimum=1)
    if len(set(digests)) != UNITS:
        raise Exp1bAuditError("Published payload digests repeat.")
    checks.append("per_seed_bound_algebra")

    if len(audited_secondary) != UNITS:
        raise Exp1bAuditError("Published result omits per-seed secondary diagnostics.")
    for offset, secondary in enumerate(audited_secondary):
        if secondary["mixture_identity"]["identity_holds"] is not True:
            raise Exp1bAuditError(
                f"Seed {offset} published a failed exact-mixture identity."
            )
        if (
            secondary["mixture_identity"]["maximum_identity_total_variation"]
            > secondary["mixture_identity"]["identity_tolerance"]
        ):
            raise Exp1bAuditError(f"Seed {offset} mixture identity exceeds tolerance.")
        if (
            secondary["centering_parity"]["constructed_centering_roundoff"]
            > secondary["centering_parity"]["constructed_centering_tolerance"]
        ):
            raise Exp1bAuditError(f"Seed {offset} centering parity exceeds tolerance.")
        if (
            secondary["persistent_state"]["states_with_carried_latent"]
            != EVALUATION_RECORD_COUNT
        ):
            raise Exp1bAuditError(
                f"Seed {offset} did not carry the persistent latent at every state."
            )
    checks.append("secondary_diagnostics")

    # --- equal-seed centre and the frozen interval ---------------------------
    mean_signed_gap = _exact_mean(tuple(recomputed_gaps))
    published_mean = _real(result["mean_signed_gap"], label="mean_signed_gap")
    if published_mean.hex() != mean_signed_gap.hex():
        raise Exp1bAuditError(
            "Published mean signed gap is not the equal-seed mean of its own gaps."
        )

    interval = _exact(result["interval"], _INTERVAL_FIELDS, label="exp1b_result.interval")
    if interval["status"] != "available":
        raise Exp1bAuditError("Published result carries no available interval.")
    if (
        interval["scheme"] != BOOTSTRAP_SCHEME
        or interval["interval_convention"] != INTERVAL_CONVENTION
        or interval["rng_namespace"] != BOOTSTRAP_RNG_NAMESPACE
        or interval["replicate_encoding"] != REPLICATE_ENCODING
        or _integer(interval["replicates"], label="interval.replicates") != REPLICATES
        or _integer(interval["units"], label="interval.units") != UNITS
        or _integer(interval["bootstrap_seed"], label="interval.bootstrap_seed")
        != BOOTSTRAP_SEED
        or _real(interval["confidence_level"], label="interval.confidence_level")
        != CONFIDENCE_LEVEL
        or _real(interval["lower_probability"], label="interval.lower_probability")
        != LOWER_PROBABILITY
        or _real(interval["upper_probability"], label="interval.upper_probability")
        != UPPER_PROBABILITY
    ):
        raise Exp1bAuditError("Published interval is not the frozen D2/D3 contract.")

    vector = interval["replicate_means_hex"]
    if not isinstance(vector, list) or len(vector) != REPLICATES:
        raise Exp1bAuditError("Published replicate vector has the wrong arity.")
    decoded = decode_replicate_vector(vector)
    recomputed_vector_sha256 = replicate_digest(decoded)
    if recomputed_vector_sha256 != _digest(
        interval["replicate_vector_sha256"], label="interval.replicate_vector_sha256"
    ):
        raise Exp1bAuditError(
            "Published replicate vector does not match its recorded digest."
        )

    try:
        recomputed = paired_seed_percentile_interval(seed_gaps=tuple(recomputed_gaps))
    except Exp1bBootstrapError as exc:
        raise Exp1bAuditError(
            f"Published gaps do not admit the frozen interval: {exc}"
        ) from exc
    if tuple(recomputed.replicate_means_hex) != tuple(vector):
        raise Exp1bAuditError(
            "Recomputed replicate vector differs from the published one."
        )
    if (
        recomputed.replicate_vector_sha256 != recomputed_vector_sha256
        or recomputed.resample_plan_sha256
        != _digest(interval["resample_plan_sha256"], label="interval.resample_plan_sha256")
    ):
        raise Exp1bAuditError("Recomputed bootstrap digests differ from the published.")
    lower = _real(interval["interval_lower"], label="interval.interval_lower")
    upper = _real(interval["interval_upper"], label="interval.interval_upper")
    if (
        lower.hex() != recomputed.interval_lower.hex()
        or upper.hex() != recomputed.interval_upper.hex()
    ):
        raise Exp1bAuditError("Recomputed interval endpoints differ from the published.")
    if lower > upper:
        raise Exp1bAuditError("Published interval is inverted.")
    if recomputed.observed_mean.hex() != mean_signed_gap.hex():
        raise Exp1bAuditError(
            "Bootstrap centre disagrees with the recomputed equal-seed mean."
        )
    checks.append("frozen_bootstrap_interval")

    return Exp1bAuditReport(
        document_sha256=document_sha256,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        amendment_sha256=amendment_sha256,
        provenance_sha256=provenance_sha256,
        recomputed_mean_signed_gap=mean_signed_gap,
        recomputed_interval_lower=lower,
        recomputed_interval_upper=upper,
        recomputed_replicate_vector_sha256=recomputed_vector_sha256,
        checks=tuple(checks),
    )


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Exp1bAuditError(f"{label} cannot be read.") from exc
    try:
        parsed = load_strict_json_bytes(raw)
    except PolicyImprovementSchemaError as exc:
        raise Exp1bAuditError(f"{label} is not strict JSON.") from exc
    return _mapping(parsed, label=label)


def main(argv: Sequence[str] | None = None) -> int:
    """Audit a published result from the command line.

    Prints the re-derived digests and endpoints on success and returns 0; prints
    the refusal and returns 1 on any failure. No writes, no network, no
    checkpoint or dataset access.
    """

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--result", required=True)
    parser.add_argument("--reduced-study-protocol", required=True)
    parser.add_argument("--reduced-study-registry", required=True)
    parser.add_argument("--reduced-study-amendment", required=True)
    parser.add_argument("--substitute-provenance", required=True)
    # Required. The census-ordering rederivation is one of the checks this
    # binary advertises, and without the protocol-pinned parent registry it
    # cannot run it. It used to be absent from the argument list, the Buck
    # resources, and the audit source profile all at once, so the CLI printed a
    # successful audit for a document whose census ordering had been replaced.
    parser.add_argument("--parent-populations", required=True)
    # Optional on purpose. An outside reviewer auditing a published result may
    # not hold the owner-signed admission; without it every other check still
    # runs, and the role-scoped authorization binding is simply reported as not
    # performed. The publication gate always supplies it.
    parser.add_argument("--execution-admission", default=None)
    arguments = parser.parse_args(argv)

    try:
        document = Path(arguments.result).read_bytes()
        report = audit_exp1b_result_document(
            document,
            protocol=_read_json(
                Path(arguments.reduced_study_protocol), label="reduced-study protocol"
            ),
            registry=_read_json(
                Path(arguments.reduced_study_registry), label="reduced-study registry"
            ),
            amendment=_read_json(
                Path(arguments.reduced_study_amendment), label="reduced-study amendment"
            ),
            provenance=_read_json(
                Path(arguments.substitute_provenance), label="substitute provenance"
            ),
            population_registry=_read_json(
                Path(arguments.parent_populations),
                label="parent population registry",
            ),
            admission=(
                _read_json(
                    Path(arguments.execution_admission),
                    label="execution admission amendment",
                )
                if arguments.execution_admission is not None
                else None
            ),
        )
    except (Exp1bAuditError, Exp1bSchemaError, OSError) as exc:
        print(f"exp1b-audit: REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"exp1b-audit: document_sha256={report.document_sha256}")
    print(f"exp1b-audit: mean_signed_gap={report.recomputed_mean_signed_gap!r}")
    print(
        "exp1b-audit: interval="
        f"[{report.recomputed_interval_lower!r}, "
        f"{report.recomputed_interval_upper!r}]"
    )
    print(f"exp1b-audit: checks={','.join(report.checks)}")
    if "role_scoped_admission_binding" not in report.checks:
        print(
            "exp1b-audit: NOTE: no --execution-admission supplied, so the "
            "producer/evaluator identities were checked for shape and "
            "distinctness but not against the owner-signed authorizations.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - packaged consumer entry point
    raise SystemExit(main())
