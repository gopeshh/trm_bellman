#!/usr/bin/env fbpython
"""Deterministic paired-residual arithmetic for the Experiment 1 diagnostic.

This module is the numerical core only. It compares, on one fixed finite
diagnostic population and one frozen value-head snapshot, the finite-reference
diagnostic against the direct depth-n Bellman-residual diagnostic.

Scientific quantities, for a deployed depth ``n``, a comparison depth ``m``,
a Bellman horizon ``K=1``, and a frozen base policy ``pi_base``:

    Q_q(s,a)   = E[r_folded + gamma * 1_nonterminal * U_q(s') | s,a]
    operator_q(s)
               = sum_a pi_base(a|s) Q_q(s,a)              (K=1 expectation)
    residual_q(s)
               = U_q(s) - operator_q(s)                   (signed)
    discrepancy(s)
               = abs(U_n(s) - U_m(s))

    epsilon_hat_q = max_{s in S_B} abs(residual_q(s))      q in {n, m}
    D_hat         = max_{s in S_B} discrepancy(s)

    R_hat_n = epsilon_hat_n / (1 - gamma)
    B_hat   = D_hat + epsilon_hat_m / (1 - gamma)
    G       = R_hat_n - B_hat

``G`` is signed. Positive ``G`` favors the finite reference. Zero and negative
values are preserved exactly; nothing here takes an absolute value of ``G``.

The finite-reference aggregate is a maximum discrepancy plus a maximum
reference-residual penalty. It is *not* the maximum of recordwise
discrepancy-plus-penalty sums. Those two differ whenever the two maxima are
attained at different states, and this module computes only the former.

Bellman horizon
---------------
This module implements the ``K=1`` contract only. The ``1/(1-gamma)`` factor is
the ``K=1`` case of ``1/(1-gamma**K)``. There is no ``K`` parameter, and the
per-state operator is the one-step expectation under the frozen base policy
taken *before* the absolute value. A sampled absolute TD error and the
candidate or deployed-mixture policy are different objects and cannot be
substituted for that operator.

Provenance this module cannot establish
---------------------------------------
``Q_n`` and ``Q_m`` must come from the *same* deployed-depth transition
construction: the same enumerated outcomes, the same successor identities, the
same folded rewards, and the same terminal mask, differing only in which
endpoint function evaluates a nonterminal successor. This helper accepts those
arrays already materialized and cannot authenticate that provenance. Neither
can it authenticate the snapshot, the population registration, the persistent
``(x,y,z,h)`` semantics, the centering contract, or the exact-mixture identity.
Those remain the production adapter's obligations and are separate diagnostics.

One call to :func:`paired_population_summary` represents exactly one frozen
snapshot evaluated on exactly one complete diagnostic population, and that is
enforced rather than assumed: every state record carries the snapshot label,
both endpoint depths, and the discount it was produced under, and aggregation
rejects a population that mixes any of them. Snapshots are therefore never
pooled before the within-seed maxima are taken. The snapshot label is an opaque
caller-supplied string, typically the authenticated checkpoint SHA-256; this
module compares labels and cannot authenticate what they name.

Numerical contract
------------------
Two arithmetics are in play, and the boundary between them is exact:

* Every per-state and population quantity — operators, residuals, discrepancy,
  the three maxima, ``R_hat_n``, ``B_hat``, and ``G`` — is computed and compared
  in **binary64**. Values arriving on a public record are canonicalized through
  :func:`_finite` before any identity is derived or compared, so a row carrying
  Python ints cannot re-enter arbitrary-precision integer arithmetic and report
  a residual this module would not compute.
* Seed averaging alone uses **exact rational intermediates**: the per-seed gaps
  are summed as :class:`fractions.Fraction` and rounded to binary64 exactly
  once, at the division. The inputs and the result are binary64; only the
  intermediate total is exact.

* The base-policy expectation is written as the
  same builtin ``sum`` over the same generator of valid actions in ascending
  action-index order that the existing bridge uses
  (``scripts/policy_improvement_theory_bridge_v2.py``, the ``horizon == 1``
  branch of ``_evaluate``), so given the *same probability vector* the two
  agree bit for bit. Two caveats. On CPython 3.12+ builtin ``sum`` applies
  Neumaier compensation to floats, so this is not a plain left-to-right
  accumulation and must not be rewritten as an accumulator loop. And the bridge
  renormalizes its law by the valid-action mass before summing, while this
  module does not, so parity holds only when the adapter passes the already
  normalized vector.
* The equal-seed mean is order-independent because exact rational addition is
  associative, and unlike the ``math.fsum`` it replaced it has no running
  partial that can overflow while the mean
  itself is representable. ``fractions`` is standard library and pulls in no
  source of randomness.
* ``PROBABILITY_TOLERANCE`` (1e-10, absolute) is the only numerical slack. It
  matches ``_PROBABILITY_TOLERANCE`` in the v2 theory bridge and is applied in
  two places: the valid-action probability mass must be within it of 1.0, and a
  masked action's probability magnitude must not exceed it.
* Inputs are validated and never repaired. This module does not renormalize,
  does not zero masked leakage, does not clip, and does not drop records.
  Normalization is the upstream adapter's job. A residual valid-mass error of
  up to ``PROBABILITY_TOLERANCE`` therefore propagates linearly into
  ``operator_q`` with a coefficient bounded by ``max_a abs(Q_q(s,a))``, and at
  the study's gamma is then amplified by ``1/(1-gamma) = 100``.
* Masked actions never enter the expectation, because the sum ranges over valid
  actions only. Their supplied probability is checked but left unmodified.
* Nonfinite inputs are rejected, and every derived result is rechecked for
  finiteness so that an overflow from finite inputs cannot escape. Because the
  public dataclasses have no constructor validation, both aggregators
  revalidate the records handed to them rather than trusting that they came
  from this module. A nonfinite value that merely lost a ``>`` comparison and
  vanished from a maximum would be a dropped record, which this module treats
  as a defect, not an outcome.
* Population maxima are first-occurrence-wins in population order.

Scope
-----
These are finite-population diagnostics, not certified uniform bounds. Exact
action enumeration does not establish population coverage, and a numerical
centering tolerance does not prove uniform exact centering. Bellman residual,
endpoint discrepancy, target lag, contraction, centering error, and deployment
mismatch stay separate quantities; only the first two appear here.

The functions in this module consume no randomness, open no files, and touch no
dataset, checkpoint, or registry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction

__all__ = [
    "BELLMAN_HORIZON_K",
    "DIAGNOSTIC_KIND",
    "EqualSeedWeightGap",
    "Exp1DiagnosticsError",
    "PairedEndpointObservation",
    "PairedPopulationSummary",
    "PairedStateDiagnostic",
    "PopulationMaximum",
    "PopulationMember",
    "PROBABILITY_TOLERANCE",
    "SeedSummary",
    "equal_seed_weight_gap",
    "paired_population_summary",
    "paired_state_diagnostic",
]


DIAGNOSTIC_KIND = "policy_improvement_exp1_paired_residual_v1"

#: This module implements the K=1 contract only; there is no K parameter.
BELLMAN_HORIZON_K = 1

#: Absolute tolerance for probability validation. Mirrors the v2 theory
#: bridge's ``_PROBABILITY_TOLERANCE``. Nothing else here is tolerant.
PROBABILITY_TOLERANCE = 1e-10

_HEX_DIGITS = frozenset("0123456789abcdef")


class Exp1DiagnosticsError(ValueError):
    """Raised when paired-residual inputs or results violate their contract."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Exp1DiagnosticsError(f"{label} must be a nonempty string.")
    return value


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Exp1DiagnosticsError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Exp1DiagnosticsError(f"{label} must be a nonnegative integer.")
    return value


def _depth(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Exp1DiagnosticsError(f"{label} must be a nonnegative integer depth.")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exp1DiagnosticsError(f"{label} must be numeric.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        # An int wider than binary64 would otherwise escape as OverflowError
        # and bypass this module's single error type.
        raise Exp1DiagnosticsError(
            f"{label} is not representable in binary64."
        ) from exc
    if not math.isfinite(result):
        raise Exp1DiagnosticsError(f"{label} must be finite.")
    return result


def _gamma(value: object, *, label: str = "gamma") -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result < 1.0:
        raise Exp1DiagnosticsError(f"{label} must lie in [0, 1).")
    return result


def _tuple(value: object, *, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise Exp1DiagnosticsError(f"{label} must be a sequence.")
    return tuple(value)


def _mask(value: Sequence[object], *, label: str) -> tuple[bool, ...]:
    if not value:
        raise Exp1DiagnosticsError(f"{label} must not be empty.")
    if any(not isinstance(item, bool) for item in value):
        raise Exp1DiagnosticsError(f"{label} must contain only booleans.")
    result = tuple(bool(item) for item in value)
    if not any(result):
        raise Exp1DiagnosticsError(f"{label} must admit at least one valid action.")
    return result


def _action_values(
    value: Sequence[object],
    mask: Sequence[bool],
    *,
    label: str,
) -> tuple[float, ...]:
    if len(value) != len(mask):
        raise Exp1DiagnosticsError(f"{label} has the wrong action inventory.")
    return tuple(
        _finite(item, label=f"{label}[{offset}]") for offset, item in enumerate(value)
    )


def _base_probabilities(
    value: Sequence[object],
    mask: Sequence[bool],
    *,
    label: str,
) -> tuple[float, ...]:
    """Validate a base-policy law without repairing it.

    Rejects a negative entry, a masked entry carrying more than
    ``PROBABILITY_TOLERANCE`` of mass, and a valid-action mass that is not
    within ``PROBABILITY_TOLERANCE`` of one. The returned tuple is the supplied
    law unchanged: no renormalization, no masked zeroing.
    """

    if len(value) != len(mask):
        raise Exp1DiagnosticsError(f"{label} has the wrong action inventory.")
    checked = tuple(
        _finite(item, label=f"{label}[{offset}]") for offset, item in enumerate(value)
    )
    for offset, (valid, probability) in enumerate(zip(mask, checked)):
        if probability < 0.0:
            raise Exp1DiagnosticsError(f"{label}[{offset}] is negative.")
        if not valid and abs(probability) > PROBABILITY_TOLERANCE:
            raise Exp1DiagnosticsError(
                f"{label}[{offset}] assigns mass to a masked action."
            )
    valid_mass = sum(
        probability for valid, probability in zip(mask, checked) if valid
    )
    if not math.isclose(
        valid_mass, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
    ):
        raise Exp1DiagnosticsError(f"{label} does not sum to one on valid actions.")
    return checked


def _base_expectation(
    action_values: Sequence[float],
    probabilities: Sequence[float],
    mask: Sequence[bool],
) -> float:
    """Expectation over valid actions, in ascending action-index order.

    The expression deliberately mirrors the existing v2 bridge's
    ``horizon == 1`` operator character for character, so the two agree bit for
    bit when handed the same probability vector.

    This is builtin ``sum`` over a generator, which on CPython 3.12+ applies
    Neumaier compensation to float operands. It is therefore *not* a plain
    left-to-right accumulation: rewriting it as an accumulator loop changes
    results and breaks parity with the bridge. Keep the expression as written.
    """

    return sum(
        probability * action_value
        for probability, action_value, valid in zip(probabilities, action_values, mask)
        if valid
    )


@dataclass(frozen=True)
class PopulationMember:
    """One ordered identity in a fixed diagnostic population."""

    state_id: str
    record_index: int
    dataset_record_sha256: str

    def __post_init__(self) -> None:
        _text(self.state_id, label="population member state ID")
        _index(self.record_index, label="population member record index")
        _digest(
            self.dataset_record_sha256,
            label="population member dataset record SHA-256",
        )


@dataclass(frozen=True)
class PairedEndpointObservation:
    """Both endpoint payloads for one augmented state, in one record.

    Holding ``U_n``/``Q_n`` and ``U_m``/``Q_m`` on a single record is
    deliberate: separately ordered per-depth arrays can silently mispair
    depths, and no downstream check would notice.

    ``action_values_n`` and ``action_values_m`` must already be the exact
    action-value arrays
    ``Q_q(s,a) = E[r_folded + gamma * 1_nonterminal * U_q(s') | s,a]``
    built from one common enumeration of outcomes under the deployed depth-n
    transition. Entries at masked actions are validated for finiteness and
    otherwise ignored.

    ``snapshot_id``, ``deployed_depth_n``, ``reference_depth_m``, and ``gamma``
    describe the evaluator and settings those arrays were produced under. They
    travel on the state record so that aggregation can *enforce* the
    one-snapshot-one-population invariant instead of merely asserting it: a
    population that mixes value-head snapshots, endpoint depths, or discounts
    is rejected rather than silently maximized across. ``snapshot_id`` is an
    opaque caller-supplied label, typically the authenticated checkpoint
    SHA-256; this module compares it but cannot authenticate it.
    """

    state_id: str
    record_index: int
    dataset_record_sha256: str
    snapshot_id: str
    deployed_depth_n: int
    reference_depth_m: int
    action_mask: Sequence[bool]
    base_probabilities: Sequence[float]
    endpoint_value_n: float
    endpoint_value_m: float
    action_values_n: Sequence[float]
    action_values_m: Sequence[float]
    gamma: float

    def __post_init__(self) -> None:
        # Copy every supplied sequence so the record cannot alias, and can
        # never mutate, a caller-owned list.
        for name in (
            "action_mask",
            "base_probabilities",
            "action_values_n",
            "action_values_m",
        ):
            object.__setattr__(
                self,
                name,
                _tuple(getattr(self, name), label=f"observation {name}"),
            )

    def member(self) -> PopulationMember:
        return PopulationMember(
            state_id=self.state_id,
            record_index=self.record_index,
            dataset_record_sha256=self.dataset_record_sha256,
        )


@dataclass(frozen=True)
class PairedStateDiagnostic:
    """Per-state paired quantities. State level, never a population maximum.

    The snapshot, depth, and discount fields are carried forward from the
    observation so that aggregation can reject a population assembled from more
    than one evaluator or setting. Every numeric field is revalidated at
    aggregation, because this dataclass is public and can be reconstructed by a
    caller that never went through :func:`paired_state_diagnostic`.
    """

    state_id: str
    record_index: int
    dataset_record_sha256: str
    snapshot_id: str
    deployed_depth_n: int
    reference_depth_m: int
    gamma: float
    endpoint_value_n: float
    endpoint_value_m: float
    base_operator_n: float
    base_operator_m: float
    signed_residual_n: float
    signed_residual_m: float
    absolute_residual_n: float
    absolute_residual_m: float
    endpoint_discrepancy: float

    def member(self) -> PopulationMember:
        return PopulationMember(
            state_id=self.state_id,
            record_index=self.record_index,
            dataset_record_sha256=self.dataset_record_sha256,
        )


@dataclass(frozen=True)
class PopulationMaximum:
    """A population maximum together with the state that attains it."""

    value: float
    state_id: str
    record_index: int
    dataset_record_sha256: str


@dataclass(frozen=True)
class PairedPopulationSummary:
    """One frozen snapshot on one complete diagnostic population."""

    snapshot_id: str
    population_id: str
    diagnostic_kind: str
    bellman_horizon: int
    gamma: float
    deployed_depth_n: int
    reference_depth_m: int
    state_count: int
    population_members: tuple[PopulationMember, ...]
    maximum_endpoint_discrepancy: PopulationMaximum
    maximum_absolute_residual_n: PopulationMaximum
    maximum_absolute_residual_m: PopulationMaximum
    direct_residual_bound_n: float
    finite_reference_bound: float
    signed_gap: float
    state_diagnostics: tuple[PairedStateDiagnostic, ...]


@dataclass(frozen=True)
class SeedSummary:
    """One training seed or run ID paired with its snapshot summary."""

    seed_id: str
    summary: PairedPopulationSummary


@dataclass(frozen=True)
class EqualSeedWeightGap:
    """Equal-weight arithmetic mean of the signed per-seed gaps."""

    seed_ids: tuple[str, ...]
    seed_count: int
    signed_gaps: tuple[float, ...]
    mean_signed_gap: float
    gamma: float
    deployed_depth_n: int
    reference_depth_m: int
    population_id: str
    population_members: tuple[PopulationMember, ...]
    snapshot_ids: tuple[str, ...]


def paired_state_diagnostic(
    observation: PairedEndpointObservation,
) -> PairedStateDiagnostic:
    """Compute both endpoint operators, residuals, and the discrepancy.

    The residual is ``U_q(s)`` minus the ``pi_base``-weighted mean of
    ``Q_q(s, .)``: the expectation is taken before the absolute value. So the
    residual is zero exactly when that weighted mean equals the endpoint, which
    is a strictly weaker condition than the action values straddling it. The
    mean absolute deviation of the action values is a different quantity and is
    not what this returns.
    """

    if not isinstance(observation, PairedEndpointObservation):
        raise Exp1DiagnosticsError("Paired observation has the wrong type.")

    state_id = _text(observation.state_id, label="state ID")
    record_index = _index(observation.record_index, label="record index")
    dataset_record_sha256 = _digest(
        observation.dataset_record_sha256, label="dataset record SHA-256"
    )
    snapshot_id = _text(observation.snapshot_id, label="snapshot ID")
    depth_n = _depth(observation.deployed_depth_n, label="deployed depth n")
    depth_m = _depth(observation.reference_depth_m, label="reference depth m")
    if depth_n >= depth_m:
        raise Exp1DiagnosticsError("Reference depth m must exceed deployed depth n.")
    gamma = _gamma(observation.gamma)
    mask = _mask(observation.action_mask, label="action mask")
    probabilities = _base_probabilities(
        observation.base_probabilities, mask, label="base policy"
    )
    action_values_n = _action_values(
        observation.action_values_n, mask, label="action values Q_n"
    )
    action_values_m = _action_values(
        observation.action_values_m, mask, label="action values Q_m"
    )
    endpoint_value_n = _finite(observation.endpoint_value_n, label="U_n")
    endpoint_value_m = _finite(observation.endpoint_value_m, label="U_m")

    base_operator_n = _finite(
        _base_expectation(action_values_n, probabilities, mask),
        label="base operator at depth n",
    )
    base_operator_m = _finite(
        _base_expectation(action_values_m, probabilities, mask),
        label="base operator at depth m",
    )
    signed_residual_n = _finite(
        endpoint_value_n - base_operator_n, label="signed residual at depth n"
    )
    signed_residual_m = _finite(
        endpoint_value_m - base_operator_m, label="signed residual at depth m"
    )
    endpoint_discrepancy = _finite(
        abs(endpoint_value_n - endpoint_value_m), label="endpoint discrepancy"
    )
    return PairedStateDiagnostic(
        state_id=state_id,
        record_index=record_index,
        dataset_record_sha256=dataset_record_sha256,
        snapshot_id=snapshot_id,
        deployed_depth_n=depth_n,
        reference_depth_m=depth_m,
        gamma=gamma,
        endpoint_value_n=endpoint_value_n,
        endpoint_value_m=endpoint_value_m,
        base_operator_n=base_operator_n,
        base_operator_m=base_operator_m,
        signed_residual_n=signed_residual_n,
        signed_residual_m=signed_residual_m,
        absolute_residual_n=abs(signed_residual_n),
        absolute_residual_m=abs(signed_residual_m),
        endpoint_discrepancy=endpoint_discrepancy,
    )


def _expected_population(
    value: Sequence[PopulationMember],
) -> tuple[PopulationMember, ...]:
    members = _tuple(value, label="expected population")
    if not members:
        raise Exp1DiagnosticsError("Expected population must not be empty.")
    seen_states: set[str] = set()
    seen_records: set[int] = set()
    result: list[PopulationMember] = []
    for offset, member in enumerate(members):
        if not isinstance(member, PopulationMember):
            raise Exp1DiagnosticsError(
                f"Expected population entry {offset} is not a population member."
            )
        if member.state_id in seen_states or member.record_index in seen_records:
            raise Exp1DiagnosticsError(
                "Expected population repeats a state or record identity."
            )
        seen_states.add(member.state_id)
        seen_records.add(member.record_index)
        result.append(member)
    return tuple(result)


#: Every numeric field of a state diagnostic, revalidated at aggregation.
_STATE_NUMERIC_FIELDS = (
    "endpoint_value_n",
    "endpoint_value_m",
    "base_operator_n",
    "base_operator_m",
    "signed_residual_n",
    "signed_residual_m",
    "absolute_residual_n",
    "absolute_residual_m",
    "endpoint_discrepancy",
)

#: The three fields population maxima are taken over. Each is an absolute
#: value or a magnitude, so each must be nonnegative at every state.
_STATE_MAGNITUDE_FIELDS = (
    "absolute_residual_n",
    "absolute_residual_m",
    "endpoint_discrepancy",
)


@dataclass(frozen=True)
class _CanonicalRow:
    """One validated state row reduced to canonical binary64 values.

    Everything downstream of validation reads these fields rather than the
    supplied record's attributes, so no later step can re-enter Python's
    arbitrary-precision integer arithmetic.
    """

    diagnostic: PairedStateDiagnostic
    gamma: float
    absolute_residual_n: float
    absolute_residual_m: float
    endpoint_discrepancy: float


def _validate_state_row(
    diagnostic: PairedStateDiagnostic,
    *,
    offset: int,
    snapshot_id: str,
    deployed_depth_n: int,
    reference_depth_m: int,
) -> _CanonicalRow:
    """Fully revalidate one state row and return its canonical binary64 values.

    ``PairedStateDiagnostic`` is public and has no constructor validation, so a
    caller can hand back a record that never passed through
    :func:`paired_state_diagnostic` — deserializing the per-state rows this
    module deliberately retains is the obvious route. Every stored field is
    therefore rechecked here, not trusted.

    Three classes of check, in order:

    1. *Metadata through the strict validators.* ``snapshot_id`` through
       :func:`_text`, both depths through :func:`_depth`, gamma through
       :func:`_gamma`, and only then compared against the population's values.
       Comparing raw fields by equality is not enough: ``False == 0.0`` and
       ``2.0 == 2`` are both true, so a bool gamma or a float depth would slip
       past an equality-only check while contradicting the strict input
       contract.
    2. *Canonicalization, finiteness, and sign.* Every numeric field is put
       through :func:`_finite`, and the returned binary64 value is what every
       later step uses. The three magnitude fields must be nonnegative. Without
       the finiteness check a NaN loses every ``>`` comparison in
       :func:`_maximum` and is silently excluded from the maximum rather than
       rejected.
    3. *The defining equations, recomputed in binary64 and compared exactly.*
       Both signed residuals, both absolute residuals, and the endpoint
       discrepancy are re-derived from the *canonicalized* endpoint values and
       operators. Each is one binary64 subtraction or absolute value, so exact
       equality is the right test and no tolerance is involved.

       Canonicalizing first is load-bearing, not tidiness. A public row may
       carry Python ints, and ``int`` arithmetic is arbitrary precision, so
       deriving from the raw attributes would check a different arithmetic than
       the module performs. Two ints that alias to one binary64 value expose
       the difference: with ``base = 2**1023`` and ``delta = 2**969``,
       ``float(base + delta) == float(base)``, so the binary64 residual is
       ``0.0`` while the exact integer residual is ``delta``. Checking in
       integer arithmetic would accept a row whose reported residual is
       ``4.99e291`` when the value this module computes is zero.

    Known limit, accepted for this contract: ``base_operator_n`` and
    ``base_operator_m`` cannot be re-derived, because the row does not retain
    the action mask, the base law, or the action-value arrays they were computed
    from. A row whose *operators* are wrong but whose residuals and discrepancy
    are consistent with those operators is still accepted. Closing that would
    require the row to carry its source arrays, or aggregation to consume
    :class:`PairedEndpointObservation` instead. Everything derivable from what
    is stored is derived and checked.
    """

    label = f"state diagnostic {offset}"
    if _text(diagnostic.snapshot_id, label=f"{label} snapshot ID") != snapshot_id:
        raise Exp1DiagnosticsError(
            f"{label} was produced under a different value-head snapshot."
        )
    if (
        _depth(diagnostic.deployed_depth_n, label=f"{label} deployed depth n")
        != deployed_depth_n
        or _depth(diagnostic.reference_depth_m, label=f"{label} reference depth m")
        != reference_depth_m
    ):
        raise Exp1DiagnosticsError(
            f"{label} was produced at a different endpoint depth pair."
        )
    gamma = _gamma(diagnostic.gamma, label=f"{label} gamma")

    canonical = {
        field: _finite(getattr(diagnostic, field), label=f"{label} {field}")
        for field in _STATE_NUMERIC_FIELDS
    }
    for field in _STATE_MAGNITUDE_FIELDS:
        if canonical[field] < 0.0:
            raise Exp1DiagnosticsError(f"{label} {field} is negative.")

    for field, expected in (
        (
            "signed_residual_n",
            canonical["endpoint_value_n"] - canonical["base_operator_n"],
        ),
        (
            "signed_residual_m",
            canonical["endpoint_value_m"] - canonical["base_operator_m"],
        ),
        ("absolute_residual_n", abs(canonical["signed_residual_n"])),
        ("absolute_residual_m", abs(canonical["signed_residual_m"])),
        (
            "endpoint_discrepancy",
            abs(canonical["endpoint_value_n"] - canonical["endpoint_value_m"]),
        ),
    ):
        if canonical[field] != expected:
            raise Exp1DiagnosticsError(
                f"{label} {field} does not equal the value its own endpoint "
                "values and operators define in binary64."
            )
    return _CanonicalRow(
        diagnostic=diagnostic,
        gamma=gamma,
        absolute_residual_n=canonical["absolute_residual_n"],
        absolute_residual_m=canonical["absolute_residual_m"],
        endpoint_discrepancy=canonical["endpoint_discrepancy"],
    )


def _maximum(
    rows: Sequence[_CanonicalRow],
    attribute: str,
) -> PopulationMaximum:
    """First-occurrence-wins maximum over the population, in population order.

    Reads the canonical binary64 values produced by :func:`_validate_state_row`,
    never the supplied record's attributes, so the comparison and the reported
    maximum are the same arithmetic the rest of the module uses. That validator
    has already rejected any nonfinite value; a NaN would otherwise lose every
    ``>`` comparison and vanish from the maximum without a word.
    """

    best = rows[0]
    best_value = float(getattr(best, attribute))
    for row in rows[1:]:
        value = float(getattr(row, attribute))
        if value > best_value:
            best = row
            best_value = value
    return PopulationMaximum(
        value=best_value,
        state_id=best.diagnostic.state_id,
        record_index=best.diagnostic.record_index,
        dataset_record_sha256=best.diagnostic.dataset_record_sha256,
    )


def paired_population_summary(
    *,
    snapshot_id: str,
    population_id: str,
    deployed_depth_n: int,
    reference_depth_m: int,
    expected_population: Sequence[PopulationMember],
    state_diagnostics: Sequence[PairedStateDiagnostic],
) -> PairedPopulationSummary:
    """Aggregate one frozen snapshot over one complete diagnostic population.

    ``expected_population`` is the caller-supplied ordered inventory. It is
    never inferred from ``state_diagnostics``; the observations are checked
    against it, position by position, and a missing, duplicated, extra, or
    misordered identity is rejected rather than repaired.

    The inventory must carry a distinct ``state_id`` *and* a distinct
    ``record_index`` per entry. That is deliberately stricter than distinct
    full identity triples, and it encodes the study's one-initial-augmented-
    state-per-record population. A population that legitimately needs several
    states per record would be rejected here loudly rather than counted
    silently, and would need this constraint revisited first.

    ``gamma``, ``snapshot_id``, and both endpoint depths are carried on every
    state diagnostic and must agree across the population *and* with the
    arguments supplied here. That is what enforces one call = one frozen
    snapshot, one discount, one endpoint pair, and one population. Pooling two
    value heads is rejected, not silently maximized across: the maxima would
    otherwise belong to no single evaluator.

    Every numeric field of every state diagnostic is revalidated here. The
    dataclass is public and has no constructor validation, and a nonfinite
    field would otherwise lose every ``>`` comparison and disappear from the
    maximum instead of being rejected.
    """

    snapshot = _text(snapshot_id, label="snapshot ID")
    population = _text(population_id, label="population ID")
    depth_n = _depth(deployed_depth_n, label="deployed depth n")
    depth_m = _depth(reference_depth_m, label="reference depth m")
    if depth_n >= depth_m:
        raise Exp1DiagnosticsError("Reference depth m must exceed deployed depth n.")

    expected = _expected_population(expected_population)
    observed = _tuple(state_diagnostics, label="state diagnostics")
    if len(observed) != len(expected):
        raise Exp1DiagnosticsError(
            "Diagnostic population size differs from its registered inventory: "
            f"expected {len(expected)} states, received {len(observed)}."
        )
    rows: list[_CanonicalRow] = []
    for offset, (member, diagnostic) in enumerate(zip(expected, observed)):
        if not isinstance(diagnostic, PairedStateDiagnostic):
            raise Exp1DiagnosticsError(
                f"State diagnostic {offset} has the wrong type."
            )
        if diagnostic.member() != member:
            raise Exp1DiagnosticsError(
                f"State diagnostic {offset} does not match its registered "
                "population identity."
            )
        rows.append(
            _validate_state_row(
                diagnostic,
                offset=offset,
                snapshot_id=snapshot,
                deployed_depth_n=depth_n,
                reference_depth_m=depth_m,
            )
        )

    diagnostics = [row.diagnostic for row in rows]
    gamma = rows[0].gamma
    if any(row.gamma != gamma for row in rows):
        raise Exp1DiagnosticsError("Diagnostic population mixes discount factors.")

    maximum_discrepancy = _maximum(rows, "endpoint_discrepancy")
    maximum_residual_n = _maximum(rows, "absolute_residual_n")
    maximum_residual_m = _maximum(rows, "absolute_residual_m")

    denominator = 1.0 - gamma
    direct_bound = _finite(
        maximum_residual_n.value / denominator, label="direct residual bound R_hat_n"
    )
    # Separate population maxima, then one sum. Not the maximum of recordwise
    # discrepancy-plus-penalty sums, which is a strictly different quantity.
    finite_reference_bound = _finite(
        maximum_discrepancy.value + maximum_residual_m.value / denominator,
        label="finite reference bound B_hat",
    )
    signed_gap = _finite(
        direct_bound - finite_reference_bound, label="signed gap G"
    )
    return PairedPopulationSummary(
        snapshot_id=snapshot,
        population_id=population,
        diagnostic_kind=DIAGNOSTIC_KIND,
        bellman_horizon=BELLMAN_HORIZON_K,
        gamma=gamma,
        deployed_depth_n=depth_n,
        reference_depth_m=depth_m,
        state_count=len(diagnostics),
        population_members=expected,
        maximum_endpoint_discrepancy=maximum_discrepancy,
        maximum_absolute_residual_n=maximum_residual_n,
        maximum_absolute_residual_m=maximum_residual_m,
        direct_residual_bound_n=direct_bound,
        finite_reference_bound=finite_reference_bound,
        signed_gap=signed_gap,
        state_diagnostics=tuple(diagnostics),
    )


def _exact_mean(values: Sequence[float]) -> float:
    """Arithmetic mean of finite binary64 values, without an overflowing total.

    ``math.fsum`` accumulates a running partial and raises
    ``OverflowError: intermediate overflow in fsum`` when that partial leaves
    binary64 range, even though the mean itself is representable. Two seeds
    with ``G_j = 1e308`` are legitimately producible by this module and have
    mean ``1e308``; ``fsum`` refuses them. Worse, it refuses them only for some
    orderings — ``(1e308, 1e308, -1e308, -1e308)`` raises while
    ``(1e308, -1e308, 1e308, -1e308)`` returns 0.0 — which contradicts the
    order-independence this module promises.

    Summing as exact rationals has neither problem. Every finite binary64 value
    is exactly a rational, integer addition cannot overflow, and the quotient is
    rounded to binary64 exactly once at the end. The result is exact-then-
    rounded and independent of the order the values arrive in.

    The mean of finite values is bounded in magnitude by the largest of them, so
    a finite input set always has a representable mean. The guard below is for
    the impossible case, not the expected one.
    """

    if not values:
        raise Exp1DiagnosticsError("Cannot average an empty set of values.")
    try:
        total = Fraction(0)
        for value in values:
            total += Fraction(value)
        return float(total / len(values))
    except (ArithmeticError, OverflowError, ValueError) as exc:
        # ArithmeticError covers OverflowError and ZeroDivisionError; the
        # explicit names keep the intent readable. Never let a raw arithmetic
        # exception escape this module's single error type.
        raise Exp1DiagnosticsError(
            "Mean signed gap is not representable in binary64."
        ) from exc


def _identical(left: object, right: object) -> bool:
    """Equality that also requires identical types, recursing into records.

    Plain ``==`` is type-blind in exactly the ways this module rejects
    everywhere else: ``False == 0.0``, ``True == 1``, and ``1 == 1.0`` are all
    true, and dataclass ``__eq__`` delegates straight to them. Comparing a
    supplied summary against a freshly rebuilt one with ``==`` therefore admits
    bool and int lookalikes in every float field and bool lookalikes in every
    integer field, including inside a nested :class:`PopulationMaximum`.

    The rebuilt summary is canonical by construction, so requiring identical
    types propagates canonical typing through the whole structure.
    """

    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            _identical(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        return len(left) == len(right) and all(
            _identical(item, other) for item, other in zip(left, right)
        )
    return bool(left == right)


def _validate_summary_scalars(
    summary: PairedPopulationSummary,
    *,
    label: str,
) -> None:
    """Put every public summary scalar through its proper validator.

    Runs before the rebuild, so a malformed field produces a specific message
    instead of a generic mismatch, and so fields the rebuild never sees
    (``bellman_horizon``, ``state_count``, ``gamma``, the three bounds) are
    checked at all. The strict validators are what reject the bool and float
    lookalikes that a bare equality test accepts.
    """

    _text(summary.snapshot_id, label=f"{label} snapshot ID")
    _text(summary.population_id, label=f"{label} population ID")
    if _text(summary.diagnostic_kind, label=f"{label} diagnostic kind") != (
        DIAGNOSTIC_KIND
    ):
        raise Exp1DiagnosticsError(f"{label} is not an Experiment 1 paired summary.")
    horizon = summary.bellman_horizon
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon != BELLMAN_HORIZON_K
    ):
        raise Exp1DiagnosticsError(f"{label} was not produced at K=1.")
    _gamma(summary.gamma, label=f"{label} gamma")
    depth_n = _depth(summary.deployed_depth_n, label=f"{label} deployed depth n")
    depth_m = _depth(summary.reference_depth_m, label=f"{label} reference depth m")
    if depth_n >= depth_m:
        raise Exp1DiagnosticsError(
            f"{label} reference depth m must exceed deployed depth n."
        )
    if not isinstance(summary.state_diagnostics, tuple) or not isinstance(
        summary.population_members, tuple
    ):
        raise Exp1DiagnosticsError(
            f"{label} must retain its rows and members as tuples."
        )
    if _index(summary.state_count, label=f"{label} state count") != len(
        summary.state_diagnostics
    ):
        raise Exp1DiagnosticsError(
            f"{label} state count differs from the rows it retains."
        )
    for name in (
        "maximum_endpoint_discrepancy",
        "maximum_absolute_residual_n",
        "maximum_absolute_residual_m",
    ):
        maximum = getattr(summary, name)
        if not isinstance(maximum, PopulationMaximum):
            raise Exp1DiagnosticsError(f"{label} {name} is not a population maximum.")
        if _finite(maximum.value, label=f"{label} {name} value") < 0.0:
            raise Exp1DiagnosticsError(f"{label} {name} value is negative.")
        _text(maximum.state_id, label=f"{label} {name} state ID")
        _index(maximum.record_index, label=f"{label} {name} record index")
        _digest(
            maximum.dataset_record_sha256,
            label=f"{label} {name} dataset record SHA-256",
        )
    for name in (
        "direct_residual_bound_n",
        "finite_reference_bound",
        "signed_gap",
    ):
        _finite(getattr(summary, name), label=f"{label} {name}")


def _revalidated_summary(
    summary: PairedPopulationSummary,
    *,
    label: str,
) -> None:
    """Re-derive a supplied summary from its own retained rows.

    ``PairedPopulationSummary`` is public and has no constructor validation, so
    a summary reaching the seed aggregator need not have come from
    :func:`paired_population_summary` — deserializing the retained per-state
    rows that contract item 12 requires is the obvious route in. Since this is
    the layer that emits the study's primary endpoint, the summary is verified
    rather than trusted, in two steps: every public scalar goes through its
    strict validator, then the maxima and all three aggregates are recomputed
    from ``state_diagnostics`` and must reproduce the stored values under
    :func:`_identical`, which requires matching types as well as values. Every
    step is deterministic binary64, so exact equality is the right test.

    Nothing here repairs a bad summary. The rebuilt object is used only as the
    comparison oracle and is discarded.
    """

    if not isinstance(summary, PairedPopulationSummary):
        raise Exp1DiagnosticsError(f"{label} is not a paired population summary.")
    _validate_summary_scalars(summary, label=label)
    rebuilt = paired_population_summary(
        snapshot_id=summary.snapshot_id,
        population_id=summary.population_id,
        deployed_depth_n=summary.deployed_depth_n,
        reference_depth_m=summary.reference_depth_m,
        expected_population=summary.population_members,
        state_diagnostics=summary.state_diagnostics,
    )
    if not _identical(rebuilt, summary):
        raise Exp1DiagnosticsError(
            f"{label} does not reproduce from its own retained state rows."
        )


def equal_seed_weight_gap(
    *,
    expected_seed_ids: Sequence[str],
    seed_summaries: Sequence[SeedSummary],
) -> EqualSeedWeightGap:
    """Arithmetic mean of the signed per-seed gaps, one unit weight per seed.

    The seed or run inventory is caller-supplied and never inferred. A
    duplicate, missing, or unexpected ID is rejected rather than absorbed, so
    the number of available records can never silently redefine the study.

    ``signed_gaps`` is reported in ``expected_seed_ids`` order regardless of the
    order the summaries arrive in. The gaps themselves are binary64, but they
    are summed as exact rationals and rounded to binary64 once, at the division
    — see :func:`_exact_mean`. So the result is the correctly rounded arithmetic
    mean, does not depend on caller ordering, and has no intermediate that can
    overflow while the mean itself is representable.

    Every summary has each of its public scalars strictly validated and is then
    re-derived from its own retained per-state rows, with the comparison
    requiring matching types as well as values. Every summary must agree on
    gamma, both endpoint depths, and the diagnostic population identity. Snapshot IDs are retained but not
    required to be distinct across seeds; binding one authenticated snapshot
    per seed is the registration's obligation, not this function's.

    This returns a point estimate only. It does not take absolute values, drop
    negative gaps, weight by puzzle count, action count, or Monte Carlo
    repeats, take a maximum across seeds, or produce any interval or
    significance decision.
    """

    expected = _tuple(expected_seed_ids, label="expected seed inventory")
    if not expected:
        raise Exp1DiagnosticsError("Expected seed inventory must not be empty.")
    ordered_ids: list[str] = []
    for offset, seed_id in enumerate(expected):
        text = _text(seed_id, label=f"expected seed ID {offset}")
        if text in ordered_ids:
            raise Exp1DiagnosticsError(
                "Expected seed inventory repeats a seed or run ID."
            )
        ordered_ids.append(text)

    supplied = _tuple(seed_summaries, label="seed summaries")
    by_seed: dict[str, PairedPopulationSummary] = {}
    for offset, item in enumerate(supplied):
        if not isinstance(item, SeedSummary):
            raise Exp1DiagnosticsError(f"Seed summary {offset} has the wrong type.")
        seed_id = _text(item.seed_id, label=f"seed summary {offset} seed ID")
        if not isinstance(item.summary, PairedPopulationSummary):
            raise Exp1DiagnosticsError(
                f"Seed summary {offset} does not carry a population summary."
            )
        if seed_id in by_seed:
            raise Exp1DiagnosticsError(f"Seed or run ID {seed_id!r} is duplicated.")
        if seed_id not in ordered_ids:
            raise Exp1DiagnosticsError(f"Seed or run ID {seed_id!r} is unexpected.")
        _revalidated_summary(item.summary, label=f"Summary for seed {seed_id!r}")
        by_seed[seed_id] = item.summary
    missing = [seed_id for seed_id in ordered_ids if seed_id not in by_seed]
    if missing:
        raise Exp1DiagnosticsError(
            f"Seed or run inventory is incomplete: missing {missing}."
        )

    reference = by_seed[ordered_ids[0]]
    for seed_id in ordered_ids[1:]:
        summary = by_seed[seed_id]
        if (
            summary.gamma != reference.gamma
            or summary.deployed_depth_n != reference.deployed_depth_n
            or summary.reference_depth_m != reference.reference_depth_m
            or summary.population_id != reference.population_id
            or summary.population_members != reference.population_members
        ):
            raise Exp1DiagnosticsError(
                f"Seed or run {seed_id!r} does not share the reference gamma, "
                "endpoint depths, and diagnostic population."
            )

    gaps = tuple(
        _finite(by_seed[seed_id].signed_gap, label=f"signed gap for {seed_id}")
        for seed_id in ordered_ids
    )
    mean = _finite(_exact_mean(gaps), label="mean signed gap")
    return EqualSeedWeightGap(
        seed_ids=tuple(ordered_ids),
        seed_count=len(ordered_ids),
        signed_gaps=gaps,
        mean_signed_gap=mean,
        gamma=reference.gamma,
        deployed_depth_n=reference.deployed_depth_n,
        reference_depth_m=reference.reference_depth_m,
        population_id=reference.population_id,
        population_members=reference.population_members,
        snapshot_ids=tuple(by_seed[seed_id].snapshot_id for seed_id in ordered_ids),
    )
