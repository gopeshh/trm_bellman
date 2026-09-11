#!/usr/bin/env fbpython
"""Frozen Experiment 1B paired-seed bootstrap: counter-based RNG and interval.

This module implements exactly the owner-frozen D2 and D3 decisions and nothing
else. It is deliberately standard-library only and imports nothing from
``scripts.policy_improvement_statistics``: not
``paired_seed_cluster_puzzle_bootstrap``, not its PRNG seed, not its private
``_quantile``, and above all not ``paired_seed_permutation_test``. That module's
scheme is registered against a different endpoint, takes puzzle-level paired
outcomes, and carries the sign-flip rule the Experiment 1B contract forbids.

D2 — resampling RNG (frozen)
----------------------------
* Exactly ``REPLICATES`` = 10,000 replicates over the eight seed-level scalar
  ``G_j`` values.
* Each replicate draws exactly ``UNITS`` = 8 seed positions with replacement
  from the registered seed order.
* Counter-based SHA-256. For replicate ``r`` and draw position ``d``::

      digest = SHA256(
          namespace_bytes
          || b"\\x00"
          || seed.to_bytes(8, "big")
          || r.to_bytes(4, "big")
          || d.to_bytes(1, "big")
      )
      sampled_index = digest[0] & 7

  ``digest[0]`` is uniform on 0..255 and 256 is divisible by 8, so the low three
  bits are exactly uniform on 0..7. No rejection sampling is needed and none is
  performed; adding any would change the draw sequence.
* Draw order is replicate ``r = 0..9999`` outer, draw position ``d = 0..7``
  inner. The order is part of the frozen contract: the same seed under a
  different traversal yields a different interval.
* ``BOOTSTRAP_SEED`` = 3246702300714487323.
* The full ordered replicate vector and its digest are persisted.

D3 — interval convention (frozen)
---------------------------------
Two-sided 95% percentile interval, Hyndman-Fan type 7, at ``p`` = 0.025 and
0.975::

    h = (B - 1) * p
    value = sorted[floor(h)] + (h - floor(h)) * (sorted[ceil(h)] - sorted[floor(h)])

At ``B`` = 10,000 that is ``h`` = 249.975 and 9749.025, so the interval reads
order statistics 249/250 and 9749/9750 with weights 0.975 and 0.025.

Counter-based, not sequential
-----------------------------
Every draw is an independent hash of its own ``(namespace, seed, r, d)``
coordinates. There is no evolving generator state, so a partial recomputation
of any replicate reproduces bit-for-bit, and the plan can be audited one draw at
a time without replaying the sequence. That is why this is specified as a
counter-based construction rather than a stateful PRNG.

RNG namespace
-------------
``BOOTSTRAP_RNG_NAMESPACE`` is the owner-frozen ASCII literal
``upi-trm-exp1b-seed-bootstrap-v1`` (31 bytes). Together with
``BOOTSTRAP_SEED`` it fixes every one of the 80,000 draws.

The public entry point :func:`paired_seed_percentile_interval` takes **no**
namespace or seed argument, so a registered result cannot be produced from a
non-D2 plan. Parameterised generation lives in the private
:func:`_interval_for_test`, which is used only to prove the plan is sensitive
to those inputs and cannot reach a registered result.

Overflow safety and rounding
----------------------------
Every mean here is computed exactly over integer ratios and rounded to binary64
exactly once, matching the arithmetic core's point estimate. That matters at the
extremes: eight gaps of ``1e308`` have the representable mean ``1e308``, but a
naive running total or ``math.fsum`` overflows before the division. Type-7
interpolation is likewise computed without forming ``high - low``, which
overflows for a widely separated pair even when the interpolated point is
finite.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "BOOTSTRAP_RNG_NAMESPACE",
    "BOOTSTRAP_SCHEME",
    "BOOTSTRAP_SEED",
    "CONFIDENCE_LEVEL",
    "Exp1bBootstrapError",
    "INTERVAL_CONVENTION",
    "LOWER_PROBABILITY",
    "PairedSeedInterval",
    "REPLICATE_ENCODING",
    "REPLICATES",
    "UNITS",
    "UPPER_PROBABILITY",
    "paired_seed_percentile_interval",
    "decode_replicate_vector",
    "encode_replicate_vector",
    "replicate_digest",
    "resample_plan",
    "revalidate_replicate_vector",
    "type7_quantile",
]


BOOTSTRAP_SCHEME = "exp1b_paired_seed_counter_sha256_v1"
INTERVAL_CONVENTION = "hyndman_fan_type7_two_sided_percentile_v1"

#: Frozen by owner decision D2. Not derived, not configurable.
REPLICATES = 10000
UNITS = 8
BOOTSTRAP_SEED = 3246702300714487323
#: Owner-approved ASCII literal. 31 bytes; pinned by golden-value tests.
BOOTSTRAP_RNG_NAMESPACE = "upi-trm-exp1b-seed-bootstrap-v1"
#: Bit-preserving persistence encoding for the ordered replicate vector.
REPLICATE_ENCODING = "float_hex_v1"

#: Frozen by owner decision D3.
CONFIDENCE_LEVEL = 0.95
LOWER_PROBABILITY = 0.025
UPPER_PROBABILITY = 0.975

_SEED_BYTES = 8
_REPLICATE_BYTES = 4
_DRAW_BYTES = 1
_INDEX_MASK = 0b111


class Exp1bBootstrapError(ValueError):
    """Raised when a bootstrap input or derived value violates the contract."""


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exp1bBootstrapError(f"{label} must be numeric.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Exp1bBootstrapError(
            f"{label} is not representable in binary64."
        ) from exc
    if not math.isfinite(result):
        raise Exp1bBootstrapError(f"{label} must be finite.")
    return result


def _namespace_bytes(value: object) -> bytes:
    """Validate an RNG namespace. Production callers take the frozen default."""

    if isinstance(value, str):
        if not value or not value.isascii():
            raise Exp1bBootstrapError("Bootstrap RNG namespace must be nonempty ASCII.")
        return value.encode("ascii")
    if isinstance(value, (bytes, bytearray)):
        if not value:
            raise Exp1bBootstrapError("Bootstrap RNG namespace must be nonempty.")
        data = bytes(value)
        if any(byte > 0x7F for byte in data):
            raise Exp1bBootstrapError("Bootstrap RNG namespace must be ASCII.")
        return data
    raise Exp1bBootstrapError("Bootstrap RNG namespace must be ASCII text or bytes.")


def resample_plan(
    *,
    namespace: str | bytes = BOOTSTRAP_RNG_NAMESPACE,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = REPLICATES,
    units: int = UNITS,
) -> tuple[tuple[int, ...], ...]:
    """Return the frozen ``replicates x units`` matrix of sampled seed indices.

    Deterministic and independent of any process state. ``namespace`` and
    ``seed`` together fix every index; both default to their frozen values and
    are parameters only so tests can prove the plan is sensitive to them.
    """

    namespace_bytes = _namespace_bytes(namespace)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Exp1bBootstrapError("Bootstrap seed must be a nonnegative integer.")
    if seed >= 1 << (8 * _SEED_BYTES):
        raise Exp1bBootstrapError("Bootstrap seed does not fit in eight bytes.")
    if replicates != REPLICATES:
        raise Exp1bBootstrapError(
            f"Experiment 1B freezes exactly {REPLICATES} replicates."
        )
    if units != UNITS:
        raise Exp1bBootstrapError(
            f"Experiment 1B freezes exactly {UNITS} paired seed units."
        )

    seed_field = seed.to_bytes(_SEED_BYTES, "big")
    prefix = namespace_bytes + b"\x00" + seed_field
    plan: list[tuple[int, ...]] = []
    for replicate in range(replicates):
        replicate_field = replicate.to_bytes(_REPLICATE_BYTES, "big")
        draws: list[int] = []
        for draw in range(units):
            digest = hashlib.sha256(
                prefix + replicate_field + draw.to_bytes(_DRAW_BYTES, "big")
            ).digest()
            draws.append(digest[0] & _INDEX_MASK)
        plan.append(tuple(draws))
    return tuple(plan)


def _exact_mean(values: Sequence[float]) -> float:
    """Arithmetic mean over integer ratios: exact, then rounded exactly once.

    Every finite binary64 value is exactly ``n / d`` with ``d`` a power of two,
    so the sum is exact in integer arithmetic and cannot overflow. Only the
    final integer division rounds, and CPython rounds it correctly. This is the
    same guarantee the arithmetic core's equal-seed mean gives, reached without
    ``Fraction``'s normalisation cost — it runs 10,000 times per interval.

    A naive running total, or ``math.fsum`` followed by a division, overflows on
    inputs whose mean is perfectly representable: eight gaps of ``1e308`` mean
    ``1e308``, but their sum does not exist in binary64.
    """

    if not values:
        raise Exp1bBootstrapError("Cannot average an empty set of values.")
    ratios = [value.as_integer_ratio() for value in values]
    denominator = 1
    for _, item in ratios:
        if item > denominator:
            denominator = item
    total = 0
    for numerator, item in ratios:
        total += numerator * (denominator // item)
    try:
        return total / (denominator * len(values))
    except (ArithmeticError, OverflowError, ValueError) as exc:
        raise Exp1bBootstrapError(
            "Mean is not representable in binary64."
        ) from exc


def _exact_interpolate(low: float, high: float, weight: float) -> float:
    """``low*(1-weight) + high*weight``, exact then rounded exactly once.

    Deliberately avoids forming ``high - low``. That intermediate overflows for
    a widely separated pair even when the interpolated point is finite: with
    ``low = -1.7e308`` and ``high = 1.7e308`` the canonical
    ``low + w*(high-low)`` returns ``inf``, while the true type-7 endpoint is
    about ``1.615e308``.
    """

    low_n, low_d = low.as_integer_ratio()
    high_n, high_d = high.as_integer_ratio()
    weight_n, weight_d = weight.as_integer_ratio()
    # low * (weight_d - weight_n) / (low_d * weight_d)
    #   + high * weight_n / (high_d * weight_d)
    numerator = low_n * (weight_d - weight_n) * high_d + high_n * weight_n * low_d
    denominator = low_d * high_d * weight_d
    try:
        return numerator / denominator
    except (ArithmeticError, OverflowError, ValueError) as exc:
        raise Exp1bBootstrapError(
            "Interpolated quantile is not representable in binary64."
        ) from exc


def type7_quantile(ordered: Sequence[float], probability: float) -> float:
    """Hyndman-Fan type 7 quantile of an already-sorted sequence.

    ``h = (B - 1) * p``, then linear interpolation between ``sorted[floor(h)]``
    and ``sorted[ceil(h)]``. Implemented here rather than imported so that
    Experiment 1B carries no dependency on the v2 statistics module, and
    computed exactly so a finite convex combination of finite order statistics
    never returns a nonfinite endpoint.
    """

    if not ordered:
        raise Exp1bBootstrapError("Cannot take a quantile of an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise Exp1bBootstrapError("Quantile probability must lie in [0, 1].")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return _finite(ordered[lower], label="quantile value")
    low = _finite(ordered[lower], label="quantile lower value")
    high = _finite(ordered[upper], label="quantile upper value")
    return _finite(
        _exact_interpolate(low, high, position - lower),
        label="interpolated quantile",
    )


def replicate_digest(values: Sequence[float]) -> str:
    """SHA-256 over the ordered replicate vector, in its computed order.

    Each value is serialised with ``float.hex()``, which is exact and
    round-trippable, so the digest pins the binary64 bits rather than a decimal
    rendering of them.
    """

    digest = hashlib.sha256()
    digest.update(f"{BOOTSTRAP_SCHEME}\n".encode("ascii"))
    for value in values:
        digest.update(f"{_finite(value, label='replicate').hex()}\n".encode("ascii"))
    return digest.hexdigest()


def encode_replicate_vector(values: Sequence[float]) -> tuple[str, ...]:
    """Bit-preserving persistence of the ordered replicate vector.

    ``float.hex()`` round-trips exactly, so an auditor can recompute the vector
    digest and both type-7 endpoints from the persisted result alone. A decimal
    rendering could not guarantee that.
    """

    if len(values) != REPLICATES:
        raise Exp1bBootstrapError(
            f"Replicate vector must hold exactly {REPLICATES} values."
        )
    return tuple(
        _finite(value, label=f"replicate {offset}").hex()
        for offset, value in enumerate(values)
    )


def decode_replicate_vector(encoded: Sequence[str]) -> tuple[float, ...]:
    """Inverse of :func:`encode_replicate_vector`, with strict revalidation."""

    if len(encoded) != REPLICATES:
        raise Exp1bBootstrapError(
            f"Encoded replicate vector must hold exactly {REPLICATES} values, "
            f"received {len(encoded)}."
        )
    decoded: list[float] = []
    for offset, item in enumerate(encoded):
        if not isinstance(item, str) or not item:
            raise Exp1bBootstrapError(
                f"Encoded replicate {offset} must be a nonempty string."
            )
        try:
            value = float.fromhex(item)
        except (TypeError, ValueError) as exc:
            raise Exp1bBootstrapError(
                f"Encoded replicate {offset} is not a binary64 hex literal."
            ) from exc
        decoded.append(_finite(value, label=f"decoded replicate {offset}"))
        if decoded[-1].hex() != item:
            raise Exp1bBootstrapError(
                f"Encoded replicate {offset} is not in canonical hex form."
            )
    return tuple(decoded)


@dataclass(frozen=True)
class PairedSeedInterval:
    """One frozen paired-seed percentile interval and its full audit trail."""

    scheme: str
    interval_convention: str
    replicates: int
    units: int
    bootstrap_seed: int
    rng_namespace: str
    confidence_level: float
    lower_probability: float
    upper_probability: float
    observed_mean: float
    interval_lower: float
    interval_upper: float
    replicate_means: tuple[float, ...]
    replicate_encoding: str
    replicate_means_hex: tuple[str, ...]
    replicate_vector_sha256: str
    resample_plan_sha256: str


def _plan_digest(plan: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{BOOTSTRAP_SCHEME}:plan\n".encode("ascii"))
    for row in plan:
        digest.update(bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _interval_for_test(
    *,
    seed_gaps: Sequence[float],
    namespace: str | bytes,
    seed: int,
) -> PairedSeedInterval:
    """Parameterised generation. Private, and never reaches a registered result.

    Exists only so tests can prove the plan is sensitive to the namespace and
    the seed. :func:`paired_seed_percentile_interval` is the production entry
    point and takes neither, so a registered Experiment 1B result cannot carry a
    non-D2 plan.
    """

    gaps = tuple(
        _finite(value, label=f"seed gap {offset}")
        for offset, value in enumerate(seed_gaps)
    )
    if len(gaps) != UNITS:
        raise Exp1bBootstrapError(
            f"Experiment 1B requires exactly {UNITS} seed-level gaps, "
            f"received {len(gaps)}."
        )
    namespace_bytes = _namespace_bytes(namespace)
    plan = resample_plan(namespace=namespace_bytes, seed=seed)

    replicate_means = [
        _exact_mean([gaps[index] for index in row]) for row in plan
    ]
    for offset, value in enumerate(replicate_means):
        _finite(value, label=f"replicate mean {offset}")

    ordered = sorted(replicate_means)
    observed = _exact_mean(gaps)
    lower = _finite(
        type7_quantile(ordered, LOWER_PROBABILITY), label="interval lower"
    )
    upper = _finite(
        type7_quantile(ordered, UPPER_PROBABILITY), label="interval upper"
    )
    if lower > upper:
        raise Exp1bBootstrapError("Interval lower endpoint exceeds its upper.")
    return PairedSeedInterval(
        scheme=BOOTSTRAP_SCHEME,
        interval_convention=INTERVAL_CONVENTION,
        replicates=REPLICATES,
        units=UNITS,
        bootstrap_seed=seed,
        rng_namespace=namespace_bytes.decode("ascii"),
        confidence_level=CONFIDENCE_LEVEL,
        lower_probability=LOWER_PROBABILITY,
        upper_probability=UPPER_PROBABILITY,
        observed_mean=_finite(observed, label="observed mean"),
        interval_lower=lower,
        interval_upper=upper,
        replicate_means=tuple(replicate_means),
        replicate_encoding=REPLICATE_ENCODING,
        replicate_means_hex=encode_replicate_vector(replicate_means),
        replicate_vector_sha256=replicate_digest(replicate_means),
        resample_plan_sha256=_plan_digest(plan),
    )


def paired_seed_percentile_interval(
    *,
    seed_gaps: Sequence[float],
) -> PairedSeedInterval:
    """The frozen 95% paired-seed percentile interval over ``G_j``.

    Takes no namespace and no seed: both are owner-frozen, and exposing them
    would let a caller publish a non-D2 plan under the registered protocol
    identity. Interval computation failure is result failure — there is no
    degraded output.

    ``seed_gaps`` must be exactly eight finite values in the registered seed
    order. Position matters: the plan indexes into this order, so supplying the
    gaps in any other order silently changes every replicate.

    Replicate statistics are exact eight-value means, rounded once. The
    equal-seed point estimate is computed separately by
    ``scripts.policy_improvement_exp1_diagnostics.equal_seed_weight_gap``; this
    function recomputes the observed mean only as the interval's centre and does
    not replace that endpoint. Both use the same exact-then-round rule, so they
    agree bit for bit.
    """

    return _interval_for_test(
        seed_gaps=seed_gaps,
        namespace=BOOTSTRAP_RNG_NAMESPACE,
        seed=BOOTSTRAP_SEED,
    )


def revalidate_replicate_vector(
    interval: PairedSeedInterval,
    *,
    seed_gaps: Sequence[float],
) -> None:
    """Strictly recheck a persisted interval against its own vector.

    Bit-exact by construction. Three properties the previous version did not
    have:

    * The digest is recomputed **from the decoded persisted vector**, not only
      from a fresh recomputation, so a persisted vector whose bytes disagree
      with its own digest is rejected.
    * Comparison is on canonical ``float.hex()`` strings, not numeric equality.
      ``+0.0 == -0.0`` in binary64, so a vector whose every value had its sign
      bit flipped previously passed while carrying a digest computed over the
      other sign.
    * Every frozen metadata field is checked: scheme, convention, namespace,
      seed, replicate and unit counts, encoding, confidence level, and both
      tail probabilities.
    """

    def _exact_int(value: object, *, label: str) -> int:
        """``True`` and ``10000.0`` are not integers; both compare equal to one."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise Exp1bBootstrapError(f"Persisted interval {label} is not an integer.")
        return value

    def _exact_float(value: object, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise Exp1bBootstrapError(
                f"Persisted interval {label} is not a binary64 float."
            )
        return value

    if interval.scheme != BOOTSTRAP_SCHEME:
        raise Exp1bBootstrapError("Persisted interval is not the frozen scheme.")
    if interval.interval_convention != INTERVAL_CONVENTION:
        raise Exp1bBootstrapError("Persisted interval is not the frozen convention.")
    if interval.rng_namespace != BOOTSTRAP_RNG_NAMESPACE:
        raise Exp1bBootstrapError("Persisted interval used a foreign RNG namespace.")
    if _exact_int(interval.bootstrap_seed, label="bootstrap seed") != BOOTSTRAP_SEED:
        raise Exp1bBootstrapError("Persisted interval used a foreign bootstrap seed.")
    if (
        _exact_int(interval.replicates, label="replicate count") != REPLICATES
        or _exact_int(interval.units, label="unit count") != UNITS
    ):
        raise Exp1bBootstrapError("Persisted interval has the wrong arity.")
    if interval.replicate_encoding != REPLICATE_ENCODING:
        raise Exp1bBootstrapError("Persisted replicate vector encoding differs.")
    if (
        _exact_float(interval.confidence_level, label="confidence level")
        != CONFIDENCE_LEVEL
    ):
        raise Exp1bBootstrapError("Persisted interval confidence level differs.")
    if (
        _exact_float(interval.lower_probability, label="lower probability")
        != LOWER_PROBABILITY
        or _exact_float(interval.upper_probability, label="upper probability")
        != UPPER_PROBABILITY
    ):
        raise Exp1bBootstrapError("Persisted interval tail probabilities differ.")
    for name in ("observed_mean", "interval_lower", "interval_upper"):
        _exact_float(getattr(interval, name), label=name)
    if len(interval.replicate_means) != REPLICATES:
        raise Exp1bBootstrapError("Persisted replicate vector has the wrong length.")
    for offset, value in enumerate(interval.replicate_means):
        if isinstance(value, bool) or not isinstance(value, float):
            raise Exp1bBootstrapError(
                f"Persisted replicate mean {offset} is not a binary64 float."
            )

    # Decode the persisted bytes and hash *those*, so the stored digest is
    # checked against the stored vector rather than only against a recomputation.
    persisted = decode_replicate_vector(interval.replicate_means_hex)
    persisted_hex = encode_replicate_vector(persisted)
    if tuple(interval.replicate_means_hex) != persisted_hex:
        raise Exp1bBootstrapError(
            "Persisted replicate hex is not in canonical form."
        )
    if replicate_digest(persisted) != interval.replicate_vector_sha256:
        raise Exp1bBootstrapError(
            "Persisted replicate vector does not match its own recorded digest."
        )
    # Bit comparison, not numeric: +0.0 and -0.0 are numerically equal.
    if encode_replicate_vector(interval.replicate_means) != persisted_hex:
        raise Exp1bBootstrapError(
            "Persisted replicate vector and its hex encoding differ in bits."
        )

    recomputed = paired_seed_percentile_interval(seed_gaps=seed_gaps)
    if encode_replicate_vector(recomputed.replicate_means) != persisted_hex:
        raise Exp1bBootstrapError(
            "Persisted replicate vector does not reproduce from its seed gaps."
        )
    for field in (
        "observed_mean",
        "interval_lower",
        "interval_upper",
        "replicate_vector_sha256",
        "resample_plan_sha256",
    ):
        mine = getattr(interval, field)
        theirs = getattr(recomputed, field)
        if isinstance(mine, float) and isinstance(theirs, float):
            if mine.hex() != theirs.hex():
                raise Exp1bBootstrapError(
                    f"Persisted interval {field} does not reproduce bit for bit."
                )
        elif mine != theirs:
            raise Exp1bBootstrapError(
                f"Persisted interval {field} does not reproduce."
            )
