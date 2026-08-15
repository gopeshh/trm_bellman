#!/usr/bin/env fbpython
"""Registered paired statistics for policy-improvement evidence.

Training seeds are the outer clusters.  Puzzle resampling is paired within a
seed, so neither method nor puzzle-level observations are treated as
independent training replicates.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.policy_improvement_schema import PolicyImprovementSchemaError


BOOTSTRAP_SCHEMA_VERSION = 1


def _validate_pairs(
    pairs_by_seed: Mapping[int, Sequence[tuple[float, float]]],
) -> dict[int, tuple[tuple[float, float], ...]]:
    if not pairs_by_seed:
        raise PolicyImprovementSchemaError("Paired outcomes cannot be empty.")
    checked: dict[int, tuple[tuple[float, float], ...]] = {}
    for seed, pairs in pairs_by_seed.items():
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PolicyImprovementSchemaError(
                "Paired-outcome seeds must be nonnegative integers."
            )
        if not pairs:
            raise PolicyImprovementSchemaError(
                f"Seed {seed} has no paired puzzle outcomes."
            )
        seed_pairs: list[tuple[float, float]] = []
        for index, pair in enumerate(pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise PolicyImprovementSchemaError(
                    f"Seed {seed} pair {index} must contain treatment and control."
                )
            treatment, control = pair
            if (
                isinstance(treatment, bool)
                or isinstance(control, bool)
                or not isinstance(treatment, (int, float))
                or not isinstance(control, (int, float))
                or not math.isfinite(float(treatment))
                or not math.isfinite(float(control))
            ):
                raise PolicyImprovementSchemaError(
                    f"Seed {seed} pair {index} is non-finite."
                )
            seed_pairs.append((float(treatment), float(control)))
        checked[seed] = tuple(seed_pairs)
    return checked


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise AssertionError("Validated values cannot be empty.")
    return sum(values) / len(values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AssertionError("Validated values cannot be empty.")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def paired_seed_cluster_puzzle_bootstrap(
    pairs_by_seed: Mapping[int, Sequence[tuple[float, float]]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    scale: float = 100.0,
) -> dict[str, Any]:
    """Resample seeds, then paired puzzles, and return a deterministic interval."""

    checked = _validate_pairs(pairs_by_seed)
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise PolicyImprovementSchemaError("Bootstrap replicates must be positive.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PolicyImprovementSchemaError("Bootstrap seed must be nonnegative.")
    if not 0.0 < confidence_level < 1.0:
        raise PolicyImprovementSchemaError("Confidence level must lie in (0,1).")
    if not math.isfinite(scale) or scale <= 0.0:
        raise PolicyImprovementSchemaError("Bootstrap scale must be positive.")
    seeds = tuple(sorted(checked))
    observed_seed_differences = [
        _mean([treatment - control for treatment, control in checked[item]])
        for item in seeds
    ]
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        cluster_differences: list[float] = []
        for _ in seeds:
            selected_seed = seeds[generator.randrange(len(seeds))]
            pairs = checked[selected_seed]
            resampled = [
                pairs[generator.randrange(len(pairs))] for _ in range(len(pairs))
            ]
            cluster_differences.append(
                _mean([treatment - control for treatment, control in resampled])
            )
        samples.append(scale * _mean(cluster_differences))
    ordered = sorted(samples)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "schema_name": "paired_seed_cluster_puzzle_bootstrap_v1",
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "seed_count": len(seeds),
        "replicates": replicates,
        "prng_seed": seed,
        "confidence_level": confidence_level,
        "scale": scale,
        "observed_difference": scale * _mean(observed_seed_differences),
        "interval_lower": _quantile(ordered, tail),
        "interval_upper": _quantile(ordered, 1.0 - tail),
        "replicate_differences": samples,
    }


def paired_seed_permutation_test(seed_differences: Sequence[float]) -> dict[str, Any]:
    """Return the exact two-sided paired sign-flip test over training seeds."""

    if not seed_differences or len(seed_differences) > 20:
        raise PolicyImprovementSchemaError(
            "Exact seed permutation requires between 1 and 20 paired seeds."
        )
    differences = []
    for index, value in enumerate(seed_differences):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyImprovementSchemaError(
                f"Seed difference {index} must be numeric."
            )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise PolicyImprovementSchemaError(
                f"Seed difference {index} must be finite."
            )
        differences.append(normalized)
    observed = abs(_mean(differences))
    total = 2 ** len(differences)
    extreme = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(
            _mean([sign * value for sign, value in zip(signs, differences)])
        )
        if statistic + tolerance >= observed:
            extreme += 1
    return {
        "schema_name": "paired_seed_sign_flip_v1",
        "schema_version": 1,
        "seed_count": len(differences),
        "observed_absolute_mean": observed,
        "enumerated_assignments": total,
        "extreme_assignments": extreme,
        "two_sided_p_value": extreme / total,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values under stable ID tie-breaking."""

    if not p_values:
        raise PolicyImprovementSchemaError("Holm input cannot be empty.")
    checked: list[tuple[str, float]] = []
    for contrast, value in p_values.items():
        if not isinstance(contrast, str) or not contrast or not contrast.isascii():
            raise PolicyImprovementSchemaError("Holm contrast IDs must be ASCII.")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise PolicyImprovementSchemaError(
                f"Holm p-value for {contrast!r} must lie in [0,1]."
            )
        checked.append((contrast, float(value)))
    checked.sort(key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(checked)
    for index, (contrast, value) in enumerate(checked):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[contrast] = running
    return {contrast: adjusted[contrast] for contrast in sorted(adjusted)}
