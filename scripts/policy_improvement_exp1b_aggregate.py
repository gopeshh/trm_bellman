#!/usr/bin/env fbpython
"""Experiment 1B result schema and aggregator.

Consumes the eight sealed per-seed bridge payloads and produces the registered
Experiment 1B result: the three separate population maxima per seed, the two
depth bounds, the signed per-seed gap, the equal-seed mean, and — once the RNG
namespace literal arrives — the frozen 10,000-resample paired-seed percentile
interval.

Division of labour
------------------
The per-state and per-population arithmetic is **not** reimplemented here. It
comes from ``scripts.policy_improvement_exp1_diagnostics``, which is the
reviewed numerical core: separate maxima rather than a maximum of recordwise
sums, the base-policy expectation taken before the absolute value, signed ``G``
with zeros and negatives preserved, and an exact-rational equal-seed mean. This
module is the schema, the ordering discipline, and the interval.

What this module will not do
----------------------------
It accepts only an :class:`~scripts.policy_improvement_exp1b_bridge.Exp1bSealedOctet`
issued by a completed authenticated route. There is no loose-payload entry
point, so a caller cannot assemble eight plausible objects into a registered
result, substitute a one-state population, or present one real checkpoint behind
eight envelope identities.

The interval is **mandatory** and takes no arguments. The RNG namespace, seed,
replicate count, resampling unit, and quantile convention are owner-frozen;
exposing any of them would let a caller publish a non-D2 number under the
registered protocol identity. Interval failure is result failure.

The complete ordered 10,000-value replicate vector is persisted in the result,
hex-encoded so it round-trips bit for bit, and revalidated against a fresh
recomputation before emission.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from scripts.policy_improvement_exp1_diagnostics import (
    equal_seed_weight_gap,
    PairedPopulationSummary,
    SeedSummary,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
)
from scripts.policy_improvement_exp1b_bootstrap import (
    _exact_mean,
    decode_replicate_vector,
    Exp1bBootstrapError,
    replicate_digest,
    REPLICATES,
    paired_seed_percentile_interval,
    revalidate_replicate_vector,
    UNITS,
)
from scripts.policy_improvement_exp1b_bridge import (
    AuthenticatedCheckpoint,
    Exp1bBridgeRoute,
    Exp1bRuntimeAttestation,
    Exp1bSealedOctet,
    Exp1bSeedPayload,
)
from scripts.policy_improvement_exp1b_schema import (
    BRIDGE_ROUTE_ID,
    Exp1bSchemaError,
    validate_secondary_diagnostics,
    validate_substitute_provenance,
    DEPLOYED_DEPTH_N,
    EVALUATION_POPULATION_ID,
    EVALUATION_RECORD_COUNT,
    exp1b_document_sha256,
    FULL_ROLE,
    GAMMA,
    PROTOCOL_ID,
    THEORY_BRIDGE_ROLE,
    REFERENCE_DEPTH_M,
    REGISTERED_SEEDS,
    TERMINAL_ENVIRONMENT_INTERACTIONS,
)

__all__ = [
    "Exp1bAggregateError",
    "Exp1bResult",
    "Exp1bSeedRow",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "aggregate_exp1b_result",
    "validated_exp1b_document",
]


#: Version 2: role-scoped producer/evaluator attestations replace the four flat
#: runtime fields, and every seed row carries the five secondary diagnostics.
RESULT_SCHEMA_NAME = "policy_improvement_exp1b_result_v1"
RESULT_SCHEMA_VERSION = 2


class Exp1bAggregateError(ValueError):
    """Raised when Experiment 1B aggregation would violate its contract."""


@dataclass(frozen=True)
class Exp1bSeedRow:
    """One seed's registered result row."""

    seed_position: int
    seed: int
    run_id: str
    checkpoint_sha256: str
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
    #: The five required secondary checks, as a separate record. Deliberately
    #: adjacent to the primary values rather than reduced into them.
    secondary_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class Exp1bResult:
    """The registered Experiment 1B result document."""

    schema_name: str
    schema_version: int
    protocol_id: str
    route_id: str
    evaluation_population: str
    deployed_depth_n: int
    reference_depth_m: int
    bellman_horizon: int
    gamma: float
    terminal_environment_interactions: int
    seed_count: int
    seed_rows: tuple[Exp1bSeedRow, ...]
    signed_gaps: tuple[float, ...]
    mean_signed_gap: float
    interval: Mapping[str, Any]
    generation_sha256: str
    effective_config_sha256: str
    payload_digests: tuple[str, ...]
    provenance_sha256: str
    schedule_sha256: str
    attempt_counts: tuple[int, ...]
    census_ordering_sha256: str
    census_binding_sha256: str
    exp1b_protocol_sha256: str
    exp1b_registry_sha256: str
    exp1b_amendment_sha256: str
    producer_attestation: Mapping[str, Any]
    evaluator_attestation: Mapping[str, Any]
    scientific_selection: bool
    validation_select_access: bool
    test_access: bool

    def document_sha256(self) -> str:
        return exp1b_document_sha256(self.as_document())

    def as_document(self) -> dict[str, Any]:
        """Canonical-JSON-ready view of the result."""

        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "route_id": self.route_id,
            "evaluation_population": self.evaluation_population,
            "deployed_depth_n": self.deployed_depth_n,
            "reference_depth_m": self.reference_depth_m,
            "bellman_horizon": self.bellman_horizon,
            "gamma": self.gamma,
            "terminal_environment_interactions": (
                self.terminal_environment_interactions
            ),
            "seed_count": self.seed_count,
            "seed_rows": [
                {
                    "seed_position": row.seed_position,
                    "seed": row.seed,
                    "run_id": row.run_id,
                    "checkpoint_sha256": row.checkpoint_sha256,
                    "state_count": row.state_count,
                    "maximum_endpoint_discrepancy": row.maximum_endpoint_discrepancy,
                    "maximum_absolute_residual_n": row.maximum_absolute_residual_n,
                    "maximum_absolute_residual_m": row.maximum_absolute_residual_m,
                    "direct_residual_bound_n": row.direct_residual_bound_n,
                    "finite_reference_bound": row.finite_reference_bound,
                    "signed_gap": row.signed_gap,
                    "discrepancy_witness_state_id": row.discrepancy_witness_state_id,
                    "residual_n_witness_state_id": row.residual_n_witness_state_id,
                    "residual_m_witness_state_id": row.residual_m_witness_state_id,
                    "secondary_diagnostics": _plain(row.secondary_diagnostics),
                }
                for row in self.seed_rows
            ],
            "signed_gaps": list(self.signed_gaps),
            "mean_signed_gap": self.mean_signed_gap,
            "interval": _plain(self.interval),
            "generation_sha256": self.generation_sha256,
            "effective_config_sha256": self.effective_config_sha256,
            "payload_digests": list(self.payload_digests),
            "provenance_sha256": self.provenance_sha256,
            "schedule_sha256": self.schedule_sha256,
            "attempt_counts": list(self.attempt_counts),
            "census_ordering_sha256": self.census_ordering_sha256,
            "census_binding_sha256": self.census_binding_sha256,
            "exp1b_protocol_sha256": self.exp1b_protocol_sha256,
            "exp1b_registry_sha256": self.exp1b_registry_sha256,
            "exp1b_amendment_sha256": self.exp1b_amendment_sha256,
            "producer_attestation": _plain(self.producer_attestation),
            "evaluator_attestation": _plain(self.evaluator_attestation),
            "scientific_selection": self.scientific_selection,
            "validation_select_access": self.validation_select_access,
            "test_access": self.test_access,
        }


def _freeze(value: Any) -> Any:
    """Recursively convert a result fragment into an immutable structure.

    ``Exp1bResult`` is a frozen dataclass, but a frozen dataclass holding a
    ``dict`` still lets a caller clear the mandatory interval before
    ``as_document()`` runs. Mappings become ``MappingProxyType`` and lists
    become tuples, so the whole result is immutable in depth.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    """Inverse of :func:`_freeze` for canonical-JSON serialisation."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _require_sealed_octet(value: object) -> Exp1bSealedOctet:
    if not isinstance(value, Exp1bSealedOctet):
        raise Exp1bAggregateError(
            "Experiment 1B aggregation accepts only a sealed octet issued by a "
            "completed authenticated route."
        )
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise Exp1bAggregateError(f"{label} must be nonempty ASCII.")
    return value


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Exp1bAggregateError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Exp1bAggregateError(f"{label} must be a nonnegative integer.")
    return value


def _real(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise Exp1bAggregateError(f"{label} must be a binary64 float.")
    if not math.isfinite(value):
        raise Exp1bAggregateError(f"{label} must be finite.")
    return value


def aggregate_exp1b_result(sealed_octet: object) -> Exp1bResult:
    """Aggregate one authenticated sealed octet into the registered result.

    Takes the octet object and nothing else. There is no namespace override, no
    seed override, and no way to suppress the interval: all three would let a
    caller publish a non-D2 number under the registered protocol identity.

    Every identity is rechecked here even though the route already checked it.
    This is the last barrier before a number is published, and the octet's
    payloads are the only place the checkpoint identity and the computed summary
    can be compared to each other.
    """

    octet = _require_sealed_octet(sealed_octet)
    if not octet.is_issued():
        raise Exp1bAggregateError(
            "Sealed octet was not issued by a completed authenticated route; a "
            "replaced copy is refused."
        )
    if octet.route_id != BRIDGE_ROUTE_ID:
        raise Exp1bAggregateError("Sealed octet is not the registered route.")

    # The octet is data, not a capability. Revalidate its provenance and
    # recompute the digest it claims, so a copied octet with edited identity
    # fields cannot publish those edits under the registered protocol ID.
    provenance_record = octet.provenance
    if not isinstance(provenance_record, Mapping):
        raise Exp1bAggregateError("Sealed octet carries no provenance record.")
    try:
        revalidated = validate_substitute_provenance(_plain(provenance_record))
    except Exp1bSchemaError as exc:
        raise Exp1bAggregateError(
            f"Sealed octet provenance is not valid: {exc}"
        ) from exc
    if exp1b_document_sha256(_plain(provenance_record)) != octet.provenance_sha256:
        raise Exp1bAggregateError(
            "Sealed octet provenance digest does not match its own provenance."
        )
    attestation_now = octet.runtime_attestation
    if not isinstance(attestation_now, Exp1bRuntimeAttestation):
        raise Exp1bAggregateError("Sealed octet carries no runtime attestation.")
    # The two roles are compared to two different things. The producer identity
    # must equal the one the provenance was finalized under; the evaluator
    # identity must equal the live attestation that served the octet.
    producer = _plain(octet.producer_attestation)
    evaluator = _plain(octet.evaluator_attestation)
    if not isinstance(producer, Mapping) or not isinstance(evaluator, Mapping):
        raise Exp1bAggregateError("Sealed octet carries no role attestations.")
    if producer != _plain(revalidated["producer_attestation"]):
        raise Exp1bAggregateError(
            "Sealed octet producer attestation differs from its provenance."
        )
    if producer.get("role") != FULL_ROLE or evaluator.get("role") != (
        THEORY_BRIDGE_ROLE
    ):
        raise Exp1bAggregateError("Sealed octet role attestations are mislabelled.")
    for field in (
        "source_git_commit",
        "runtime_sha256",
        "launcher_sha256",
        "runtime_authorization_sha256",
    ):
        if evaluator.get(field) != getattr(attestation_now, field):
            raise Exp1bAggregateError(
                f"Sealed octet evaluator {field} differs from the live attestation."
            )
    if producer.get("runtime_sha256") == evaluator.get("runtime_sha256"):
        raise Exp1bAggregateError(
            "Sealed octet producer and evaluator runtimes are the same artifact."
        )
    for field, claimed in (
        ("exp1b_protocol_sha256", octet.exp1b_protocol_sha256),
        ("exp1b_registry_sha256", octet.exp1b_registry_sha256),
        ("exp1b_amendment_sha256", octet.exp1b_amendment_sha256),
        ("ordered_population_sha256", octet.census_ordering_sha256),
        ("population_binding_sha256", octet.census_binding_sha256),
        ("request_schedule_sha256", octet.schedule_sha256),
    ):
        if field in {"ordered_population_sha256"}:
            # The census ordering digest is the route's recomputation; the
            # provenance carries the registered ordered-record digest. Both are
            # bound at route open, so only the document digests are compared
            # here.
            continue
        if revalidated[field] != claimed:
            raise Exp1bAggregateError(
                f"Sealed octet {field} differs from its provenance."
            )
    checkpoints = octet.authenticated_checkpoints
    if not isinstance(checkpoints, tuple) or len(checkpoints) != UNITS:
        raise Exp1bAggregateError(
            "Sealed octet does not carry eight authenticated checkpoints."
        )
    for item in checkpoints:
        if not isinstance(item, AuthenticatedCheckpoint):
            raise Exp1bAggregateError("Sealed octet checkpoint is not authenticated.")
        if item.effective_config_sha256 != octet.effective_config_sha256:
            raise Exp1bAggregateError(
                "Sealed octet effective-config digest is inconsistent."
            )
    attempts = octet.attempt_counts
    if not isinstance(attempts, tuple) or len(attempts) != UNITS:
        raise Exp1bAggregateError("Sealed octet attempt counts have the wrong arity.")
    for attempt in attempts:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise Exp1bAggregateError("Sealed octet attempt count is invalid.")
    digests = octet.payload_digests
    if not isinstance(digests, tuple) or len(digests) != UNITS:
        raise Exp1bAggregateError("Sealed octet payload digests have the wrong arity.")
    for digest in digests:
        _digest(digest, label="Sealed octet payload digest")
    for label, value in (
        ("provenance digest", octet.provenance_sha256),
        ("schedule digest", octet.schedule_sha256),
        ("census ordering digest", octet.census_ordering_sha256),
        ("census binding digest", octet.census_binding_sha256),
        ("protocol digest", octet.exp1b_protocol_sha256),
        ("registry digest", octet.exp1b_registry_sha256),
        ("amendment digest", octet.exp1b_amendment_sha256),
    ):
        _digest(value, label=f"Sealed octet {label}")
    attestation = octet.runtime_attestation
    if not isinstance(attestation, Exp1bRuntimeAttestation):
        raise Exp1bAggregateError("Sealed octet carries no runtime attestation.")

    payloads = list(octet.payloads)
    if len(payloads) != UNITS:
        raise Exp1bAggregateError(
            f"Experiment 1B aggregates exactly {UNITS} seed payloads; "
            f"received {len(payloads)}."
        )
    provenance = octet.provenance
    if not isinstance(provenance, Mapping):
        raise Exp1bAggregateError("Sealed octet carries no provenance record.")
    registered_checkpoints = provenance.get("sealed_checkpoints")
    if not isinstance(registered_checkpoints, Sequence) or len(
        registered_checkpoints
    ) != UNITS:
        raise Exp1bAggregateError("Sealed octet provenance has the wrong arity.")

    seeds: list[int] = []
    positions: list[int] = []
    run_ids: list[str] = []
    snapshots: list[str] = []
    values: list[Any] = []
    summaries: list[PairedPopulationSummary] = []
    for offset, payload in enumerate(payloads):
        if not isinstance(payload, Exp1bSeedPayload):
            raise Exp1bAggregateError(f"Sealed octet payload {offset} has the wrong type.")
        position = _index(payload.seed_position, label=f"payload {offset} position")
        seed = _index(payload.seed, label=f"payload {offset} seed")
        run_id = _text(payload.run_id, label=f"payload {offset} run ID")
        snapshot = _digest(
            payload.checkpoint_sha256, label=f"payload {offset} checkpoint digest"
        )
        if position != offset or seed != REGISTERED_SEEDS[offset]:
            raise Exp1bAggregateError(
                f"Sealed octet payload {offset} is not the registered seed in order."
            )
        registered = registered_checkpoints[offset]
        assert isinstance(registered, Mapping)
        if (
            registered["run_id"] != run_id
            or registered["seed"] != seed
            or registered["seed_position"] != position
            or registered["checkpoint_sha256"] != snapshot
        ):
            raise Exp1bAggregateError(
                f"Sealed octet payload {offset} differs from its registered "
                "checkpoint identity."
            )
        if payload.route_id != BRIDGE_ROUTE_ID:
            raise Exp1bAggregateError(
                f"Payload {offset} was produced under route "
                f"{payload.route_id!r}, not the registered Experiment 1B route."
            )
        if payload.census_ordering_sha256 != octet.census_ordering_sha256:
            raise Exp1bAggregateError(
                f"Payload {offset} was computed over a different census ordering."
            )
        if payload.payload_sha256() != digests[offset]:
            raise Exp1bAggregateError(
                f"Payload {offset} digest differs from the durable schedule record."
            )
        if (
            checkpoints[offset].run_id != run_id
            or checkpoints[offset].checkpoint_sha256 != snapshot
        ):
            raise Exp1bAggregateError(
                f"Payload {offset} is not the authenticated checkpoint for its seed."
            )
        result_values = payload.result_values()
        # Revalidated here too. Aggregation is the last barrier before a number
        # is published, and a seed whose secondary checks do not hold is not a
        # seed this study may report.
        try:
            validate_secondary_diagnostics(
                _plain(result_values.secondary_diagnostics),
                path=f"payload {offset}.secondary_diagnostics",
            )
        except Exp1bSchemaError as exc:
            raise Exp1bAggregateError(
                f"Payload {offset} secondary diagnostics are invalid: {exc}"
            ) from exc
        if result_values.state_count != EVALUATION_RECORD_COUNT:
            raise Exp1bAggregateError(
                f"Payload {offset} covers {result_values.state_count} states, not "
                f"the registered {EVALUATION_RECORD_COUNT}."
            )
        for label, number in (
            ("maximum_endpoint_discrepancy", result_values.maximum_endpoint_discrepancy),
            ("maximum_absolute_residual_n", result_values.maximum_absolute_residual_n),
            ("maximum_absolute_residual_m", result_values.maximum_absolute_residual_m),
            ("direct_residual_bound_n", result_values.direct_residual_bound_n),
            ("finite_reference_bound", result_values.finite_reference_bound),
            ("signed_gap", result_values.signed_gap),
        ):
            _real(number, label=f"payload {offset} {label}")
        summary = payload.summary
        if summary is not None:
            # Present only in the serving process. When it is, use the reviewed
            # core's strict revalidation as an extra barrier.
            if not isinstance(summary, PairedPopulationSummary):
                raise Exp1bAggregateError(
                    f"Payload {offset} carries a malformed population summary."
                )
            if summary.snapshot_id != snapshot:
                raise Exp1bAggregateError(
                    f"Payload {offset} checkpoint identity differs from the snapshot "
                    "its summary was computed under."
                )
            if (
                summary.deployed_depth_n != DEPLOYED_DEPTH_N
                or summary.reference_depth_m != REFERENCE_DEPTH_M
                or summary.population_id != EVALUATION_POPULATION_ID
            ):
                raise Exp1bAggregateError(
                    f"Payload {offset} was not produced on the registered route."
                )
            if summary.gamma != GAMMA:
                raise Exp1bAggregateError(f"Payload {offset} uses a different discount.")
            summaries.append(summary)
        seeds.append(seed)
        positions.append(position)
        run_ids.append(run_id)
        snapshots.append(snapshot)
        values.append(result_values)

    if tuple(seeds) != REGISTERED_SEEDS:
        raise Exp1bAggregateError(
            "Experiment 1B payloads are not the registered seeds in order."
        )
    if len(set(run_ids)) != UNITS:
        raise Exp1bAggregateError("Experiment 1B payloads repeat a run ID.")
    if len(set(snapshots)) != UNITS:
        raise Exp1bAggregateError(
            "Experiment 1B payloads repeat a sealed checkpoint digest."
        )
    if summaries and len(summaries) != UNITS:
        raise Exp1bAggregateError(
            "Experiment 1B octet mixes in-memory and reloaded payloads."
        )

    signed_gaps = tuple(item.signed_gap for item in values)
    if summaries:
        # Single-process octet: the reviewed core revalidates every summary
        # against its own retained rows before averaging.
        aggregate = equal_seed_weight_gap(
            expected_seed_ids=tuple(run_ids),
            seed_summaries=[
                SeedSummary(seed_id=run_id, summary=summary)
                for run_id, summary in zip(run_ids, summaries)
            ],
        )
        if aggregate.signed_gaps != signed_gaps:
            raise Exp1bAggregateError(
                "Payload gaps differ from their own population summaries."
            )
        mean_signed_gap = aggregate.mean_signed_gap
    else:
        # Reloaded octet: the same exact-then-round rule the core uses.
        mean_signed_gap = _exact_mean(signed_gaps)

    rows = tuple(
        Exp1bSeedRow(
            seed_position=position,
            seed=seed,
            run_id=run_id,
            checkpoint_sha256=snapshot,
            state_count=item.state_count,
            maximum_endpoint_discrepancy=item.maximum_endpoint_discrepancy,
            maximum_absolute_residual_n=item.maximum_absolute_residual_n,
            maximum_absolute_residual_m=item.maximum_absolute_residual_m,
            direct_residual_bound_n=item.direct_residual_bound_n,
            finite_reference_bound=item.finite_reference_bound,
            signed_gap=item.signed_gap,
            discrepancy_witness_state_id=item.discrepancy_witness_state_id,
            residual_n_witness_state_id=item.residual_n_witness_state_id,
            residual_m_witness_state_id=item.residual_m_witness_state_id,
            secondary_diagnostics=_freeze(item.secondary_diagnostics),
        )
        for position, seed, run_id, snapshot, item in zip(
            positions, seeds, run_ids, snapshots, values
        )
    )

    # The interval is mandatory. A failure here is a result failure.
    try:
        computed = paired_seed_percentile_interval(seed_gaps=signed_gaps)
        revalidate_replicate_vector(computed, seed_gaps=signed_gaps)
    except Exp1bBootstrapError as exc:
        raise Exp1bAggregateError(
            f"Experiment 1B interval could not be produced: {exc}"
        ) from exc
    if computed.observed_mean != mean_signed_gap:
        raise Exp1bAggregateError(
            "Bootstrap centre disagrees with the equal-seed point estimate."
        )
    interval = {
        "status": "available",
        "scheme": computed.scheme,
        "interval_convention": computed.interval_convention,
        "replicates": computed.replicates,
        "units": computed.units,
        "bootstrap_seed": computed.bootstrap_seed,
        "rng_namespace": computed.rng_namespace,
        "confidence_level": computed.confidence_level,
        "lower_probability": computed.lower_probability,
        "upper_probability": computed.upper_probability,
        "interval_lower": computed.interval_lower,
        "interval_upper": computed.interval_upper,
        "replicate_encoding": computed.replicate_encoding,
        "replicate_means_hex": list(computed.replicate_means_hex),
        "replicate_vector_sha256": computed.replicate_vector_sha256,
        "resample_plan_sha256": computed.resample_plan_sha256,
    }

    return Exp1bResult(
        schema_name=RESULT_SCHEMA_NAME,
        schema_version=RESULT_SCHEMA_VERSION,
        protocol_id=PROTOCOL_ID,
        route_id=BRIDGE_ROUTE_ID,
        evaluation_population=EVALUATION_POPULATION_ID,
        deployed_depth_n=DEPLOYED_DEPTH_N,
        reference_depth_m=REFERENCE_DEPTH_M,
        bellman_horizon=1,
        gamma=GAMMA,
        terminal_environment_interactions=TERMINAL_ENVIRONMENT_INTERACTIONS,
        seed_count=UNITS,
        seed_rows=rows,
        signed_gaps=signed_gaps,
        mean_signed_gap=mean_signed_gap,
        interval=_freeze(interval),
        generation_sha256=octet.generation_sha256,
        effective_config_sha256=octet.effective_config_sha256,
        payload_digests=tuple(digests),
        provenance_sha256=octet.provenance_sha256,
        schedule_sha256=octet.schedule_sha256,
        attempt_counts=tuple(octet.attempt_counts),
        census_ordering_sha256=octet.census_ordering_sha256,
        census_binding_sha256=octet.census_binding_sha256,
        exp1b_protocol_sha256=octet.exp1b_protocol_sha256,
        exp1b_registry_sha256=octet.exp1b_registry_sha256,
        exp1b_amendment_sha256=octet.exp1b_amendment_sha256,
        producer_attestation=_freeze(producer),
        evaluator_attestation=_freeze(evaluator),
        scientific_selection=False,
        validation_select_access=False,
        test_access=False,
    )


def _require_durable_agreement(
    result: Exp1bResult, route: Any
) -> Mapping[str, Any]:
    """Re-derive the result's every durable claim from the generation itself.

    Under the generation lock, reload the provenance, the completed schedule,
    all eight canonical payloads, their digests, and their attempt counts, and
    require exact equality with what is about to be published.

    This exists because the previous boundary accepted a public, replaceable
    ``Exp1bResult``: a completed octet whose generation digest and attempt counts
    had been rewritten still validated, and a seed's secondary record could be
    edited while keeping its original payload digest.
    """

    generation = route.generation
    with generation.exclusive():
        reloaded = route.reload_durable_state()
    if result.generation_sha256 != reloaded["generation_sha256"]:
        raise Exp1bAggregateError(
            "Published generation digest is not the durable generation's."
        )
    if result.provenance_sha256 != reloaded["provenance_sha256"]:
        raise Exp1bAggregateError(
            "Published provenance digest is not the durable provenance's."
        )
    if result.schedule_sha256 != reloaded["schedule_sha256"]:
        raise Exp1bAggregateError(
            "Published schedule digest is not the durable schedule's."
        )
    if tuple(result.payload_digests) != tuple(reloaded["payload_digests"]):
        raise Exp1bAggregateError(
            "Published payload digests are not the durable schedule's."
        )
    if tuple(result.attempt_counts) != tuple(reloaded["attempt_counts"]):
        raise Exp1bAggregateError(
            "Published attempt counts are not the durable schedule's."
        )
    # The census ordering is a top-level provenance claim, and it was the one
    # identity field this comparison did not cover: replacing it with another
    # valid digest left every row, payload, and payload hash untouched, so a
    # result could name an ordering its evidence did not come from. The route
    # holds the ordering it authenticated; the payloads each record the one they
    # were produced under.
    if result.census_ordering_sha256 != route.census_ordering_sha256:
        raise Exp1bAggregateError(
            "Published census ordering digest is not the authenticated route's."
        )
    # Likewise the evaluator identity. The auditor checks it against the signed
    # admission, but only inside `if admission is not None`, so a caller who omits
    # the admission was checked on shape alone. The route's identity is not a
    # caller argument.
    if _plain(result.evaluator_attestation) != _plain(route.evaluator_attestation):
        raise Exp1bAggregateError(
            "Published evaluator attestation is not the authenticated route's "
            "live Stage B identity."
        )
    payloads = reloaded["payloads"]
    if len(payloads) != UNITS or len(result.seed_rows) != UNITS:
        raise Exp1bAggregateError("Durable generation does not hold eight payloads.")
    for offset, (row, payload) in enumerate(zip(result.seed_rows, payloads)):
        values = payload.result_values()
        if (
            row.seed_position != payload.seed_position
            or row.seed != payload.seed
            or row.run_id != payload.run_id
            or row.checkpoint_sha256 != payload.checkpoint_sha256
        ):
            raise Exp1bAggregateError(
                f"Published row {offset} identity differs from its durable payload."
            )
        for name in (
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
        ):
            if getattr(row, name) != getattr(values, name):
                raise Exp1bAggregateError(
                    f"Published row {offset} {name} differs from its durable payload."
                )
        if _plain(row.secondary_diagnostics) != _plain(
            payload.secondary_diagnostics
        ):
            raise Exp1bAggregateError(
                f"Published row {offset} secondary diagnostics differ from the "
                "durable payload."
            )
        if payload.census_ordering_sha256 != result.census_ordering_sha256:
            raise Exp1bAggregateError(
                f"Durable payload {offset} was produced under a different census "
                "ordering than the one being published."
            )
        if payload.payload_sha256() != result.payload_digests[offset]:
            raise Exp1bAggregateError(
                f"Durable payload {offset} does not hash to the published digest."
            )
    return reloaded


def validated_exp1b_document(
    result: Exp1bResult,
    *,
    route: Any,
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    amendment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> bytes:
    """Serialize, re-derive, and independently audit; return the canonical bytes.

    Called by ``Exp1bEvidenceGeneration.publish_result`` *as part of* publishing,
    not by a caller who then hands the writer a result. Returning a proof object
    instead of the bytes was the previous shape and it failed: the mint accepted
    arbitrary bytes, so possessing a proof said nothing about what had been
    checked. The writer now runs this function and writes what it returns, so the
    audited bytes and the published bytes are the same object.

    The four study documents are required, not optional. The previous version
    rechecked a handful of fields on the object it was handed, which a
    ``dataclasses.replace`` could satisfy while carrying a changed RNG namespace,
    a changed bootstrap seed, a changed endpoint, and a self-consistent but
    different 10,000-value vector.

    This version serializes the canonical bytes and then runs the *independent
    consumer* over them -- the same auditor an outside reviewer would run, which
    recomputes every per-row bound from the published maxima, recomputes the
    equal-seed mean, recomputes the whole frozen bootstrap, and rebinds every
    identity against the registered documents. Publication and audit therefore
    cannot disagree, because publication is gated on the audit passing.
    """

    if not isinstance(result, Exp1bResult):
        raise Exp1bAggregateError("Publication requires an Experiment 1B result.")
    # Finding 5: the result is data, not authority. Everything it claims about
    # the durable generation is re-derived from the generation under its lock.
    if not isinstance(route, Exp1bBridgeRoute):
        raise Exp1bAggregateError(
            "Publication requires the authenticated route that served the octet, "
            "not an object that merely exposes its methods."
        )
    # Mandatory, not optional. The auditor checks both role identities against
    # the signed authorizations only inside `if admission is not None`, so an
    # optional admission on the publication path meant the canonical durable
    # result could be produced with the authorization check skipped entirely.
    # Standalone auditing keeps the optional argument -- an outside reviewer may
    # not hold the admission -- but publishing is not auditing.
    if not isinstance(admission, Mapping) or not admission:
        raise Exp1bAggregateError(
            "Publication requires the owner-signed admission; it carries the "
            "role authorizations the producer and evaluator identities are "
            "checked against."
        )
    _require_durable_agreement(result, route)
    document = result.as_document()
    payload = canonical_json_bytes(document) + b"\n"

    # Structural pre-checks, so a malformed result fails with a publication
    # error rather than surfacing as an audit refusal.
    reloaded = load_strict_json_bytes(payload[:-1])
    if not isinstance(reloaded, Mapping):
        raise Exp1bAggregateError("Serialized Experiment 1B result is not an object.")
    if (
        reloaded.get("schema_name") != RESULT_SCHEMA_NAME
        or reloaded.get("schema_version") != RESULT_SCHEMA_VERSION
        or reloaded.get("protocol_id") != PROTOCOL_ID
        or reloaded.get("route_id") != BRIDGE_ROUTE_ID
    ):
        raise Exp1bAggregateError("Serialized Experiment 1B result identity differs.")
    interval = reloaded.get("interval")
    if not isinstance(interval, Mapping) or interval.get("status") != "available":
        raise Exp1bAggregateError(
            "Serialized Experiment 1B result carries no available interval."
        )
    vector = interval.get("replicate_means_hex")
    if not isinstance(vector, list) or len(vector) != REPLICATES:
        raise Exp1bAggregateError(
            "Serialized Experiment 1B interval does not carry the full vector."
        )
    if replicate_digest(decode_replicate_vector(vector)) != interval.get(
        "replicate_vector_sha256"
    ):
        raise Exp1bAggregateError(
            "Serialized replicate vector does not match its recorded digest."
        )
    for field, expected in (
        ("seed_count", UNITS),
        ("deployed_depth_n", DEPLOYED_DEPTH_N),
        ("reference_depth_m", REFERENCE_DEPTH_M),
        ("terminal_environment_interactions", TERMINAL_ENVIRONMENT_INTERACTIONS),
    ):
        if reloaded.get(field) != expected:
            raise Exp1bAggregateError(
                f"Serialized Experiment 1B result {field} differs."
            )
    for field in ("seed_rows", "signed_gaps", "payload_digests", "attempt_counts"):
        item = reloaded.get(field)
        if not isinstance(item, list) or len(item) != UNITS:
            raise Exp1bAggregateError(
                f"Serialized Experiment 1B result {field} has the wrong arity."
            )
    for field in ("scientific_selection", "validation_select_access", "test_access"):
        if reloaded.get(field) is not False:
            raise Exp1bAggregateError(
                f"Serialized Experiment 1B result {field} must be false."
            )

    # Recompute the interval from the published gaps and re-bind the persisted
    # vector to it. This is the check a self-consistent replacement defeats.
    published_gaps = tuple(reloaded["signed_gaps"])
    for offset, gap in enumerate(published_gaps):
        if isinstance(gap, bool) or not isinstance(gap, float):
            raise Exp1bAggregateError(
                f"Serialized signed_gaps[{offset}] is not a binary64 float."
            )
    try:
        recomputed = paired_seed_percentile_interval(seed_gaps=published_gaps)
        revalidate_replicate_vector(recomputed, seed_gaps=published_gaps)
    except Exp1bBootstrapError as exc:
        raise Exp1bAggregateError(
            f"Serialized Experiment 1B interval could not be reproduced: {exc}"
        ) from exc
    if tuple(recomputed.replicate_means_hex) != tuple(vector):
        raise Exp1bAggregateError(
            "Serialized replicate vector is not the one its own gaps produce."
        )

    # The independent consumer is the gate. Imported here rather than at module
    # scope: the auditor must not depend on this module, and a cycle would make
    # that dependency direction ambiguous.
    from scripts.policy_improvement_exp1b_auditor import (
        audit_exp1b_result_document,
        Exp1bAuditError,
    )

    try:
        audit_exp1b_result_document(
            payload,
            protocol=protocol,
            registry=registry,
            amendment=amendment,
            provenance=provenance,
            admission=admission,
            # Optional to a standalone reviewer, mandatory here: the route
            # authenticated this registry on open, so the independent auditor
            # can rederive the census member ordering rather than accept the
            # result's word for it.
            population_registry=route.population_registry_document,
        )
    except Exp1bAuditError as exc:
        raise Exp1bAggregateError(
            f"Experiment 1B result failed independent audit before publication: {exc}"
        ) from exc

    # These exact bytes passed the audit, so these exact bytes are what the
    # writer installs. No token, no registry, no second object to carry the
    # claim: the claim is the return value.
    return payload
