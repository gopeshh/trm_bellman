#!/usr/bin/env fbpython
"""Synthetic regression tests for the Experiment 1 paired-residual core.

Every fixture here is invented. No registered record, checkpoint, learned
model, or prior result is read, and no scientific condition is chosen. The
alternative gamma values, depths, and seed counts below are numerical fixtures
only; they do not add study conditions.

Floating-point policy in these tests:

* Fixtures at gamma = 1/2 and gamma = 0 are dyadic, so every expected value is
  exactly representable in binary64 and is asserted with ``assertEqual``.
* The gamma = 0.99 fixture is not dyadic. Its expected values are derived from
  ``fractions.Fraction`` over the exact binary64 inputs, independently of the
  implementation's operation order, and compared with
  :meth:`_ExactArithmeticMixin.assert_close`, which allows at most 4 units in
  the last place of the expected magnitude. Four ULP covers the two roundings
  the specified formula needs (one division, one addition) with margin.
"""

from __future__ import annotations

import builtins
import itertools
import math
import random
import types
import unittest
from dataclasses import replace
from fractions import Fraction
from typing import Any
from unittest import mock

from scripts import policy_improvement_exp1_diagnostics as exp1
from scripts.policy_improvement_exp1_diagnostics import (
    equal_seed_weight_gap,
    Exp1DiagnosticsError,
    paired_population_summary,
    paired_state_diagnostic,
    PairedEndpointObservation,
    PairedStateDiagnostic,
    PopulationMember,
    SeedSummary,
)
from scripts.policy_improvement_theory_bridge_v2 import (
    _exact_action_values,
    _PROBABILITY_TOLERANCE as _BRIDGE_PROBABILITY_TOLERANCE,
    state_identity,
    TheoryOutcomeV2,
    TheoryStateV2,
)


def _digest(value: str) -> str:
    """A deterministic, obviously synthetic 64-hex record digest.

    Deliberately not ``hash()``: string hashing is salted per process, and
    these fixtures must be reproducible across runs.
    """

    total = 0
    for character in value:
        total = (total * 131 + ord(character)) % (1 << 64)
    return f"{total:016x}" + "0" * 48


_STATE_A = _digest("state-A")
_STATE_B = _digest("state-B")


_SNAPSHOT = "snapshot-0"


def _observation(
    *,
    state_id: str,
    record_index: int,
    dataset_record_sha256: str,
    gamma: float,
    endpoint_value_n: float,
    endpoint_value_m: float,
    action_values_n: Any,
    action_values_m: Any,
    base_probabilities: Any = (0.5, 0.5),
    action_mask: Any = (True, True),
    snapshot_id: Any = _SNAPSHOT,
    deployed_depth_n: Any = 2,
    reference_depth_m: Any = 8,
) -> PairedEndpointObservation:
    return PairedEndpointObservation(
        state_id=state_id,
        record_index=record_index,
        dataset_record_sha256=dataset_record_sha256,
        snapshot_id=snapshot_id,
        deployed_depth_n=deployed_depth_n,
        reference_depth_m=reference_depth_m,
        action_mask=action_mask,
        base_probabilities=base_probabilities,
        endpoint_value_n=endpoint_value_n,
        endpoint_value_m=endpoint_value_m,
        action_values_n=action_values_n,
        action_values_m=action_values_m,
        gamma=gamma,
    )


def _separate_maxima_observations(
    gamma: float,
    *,
    depth_n_action_values_a: tuple[float, float] = (-6.0, -4.0),
    snapshot_id: str = _SNAPSHOT,
    reference_depth_m: int = 8,
) -> tuple[PairedEndpointObservation, PairedEndpointObservation]:
    """Two paired states with deliberately misaligned maxima.

    State A carries the largest discrepancy (3) and, by default, the largest
    depth-n residual (5), with a depth-m residual of 0. State B carries the
    largest depth-m residual (2), a discrepancy of 0, and a depth-n residual of
    1/2. Because the discrepancy maximum and the depth-m residual maximum are
    attained at different states, the maximum of recordwise sums is strictly
    smaller than the sum of separate maxima.
    """

    state_a = _observation(
        state_id="state-A",
        record_index=0,
        dataset_record_sha256=_STATE_A,
        gamma=gamma,
        endpoint_value_n=0.0,
        endpoint_value_m=3.0,
        action_values_n=depth_n_action_values_a,
        action_values_m=(2.0, 4.0),
        snapshot_id=snapshot_id,
        reference_depth_m=reference_depth_m,
    )
    state_b = _observation(
        state_id="state-B",
        record_index=1,
        dataset_record_sha256=_STATE_B,
        gamma=gamma,
        endpoint_value_n=1.0,
        endpoint_value_m=1.0,
        action_values_n=(0.25, 0.75),
        action_values_m=(-1.5, -0.5),
        snapshot_id=snapshot_id,
        reference_depth_m=reference_depth_m,
    )
    return state_a, state_b


_EXPECTED_POPULATION = (
    PopulationMember(
        state_id="state-A", record_index=0, dataset_record_sha256=_STATE_A
    ),
    PopulationMember(
        state_id="state-B", record_index=1, dataset_record_sha256=_STATE_B
    ),
)

_SINGLETON_POPULATION = (_EXPECTED_POPULATION[0],)


def _extreme_gap_summary(snapshot_id: str, target_gap: float) -> Any:
    """A genuinely module-produced one-state summary with an arbitrary gap.

    Used for magnitudes the ordinary fixtures cannot reach. At gamma = 0 the
    penalty factor is 1, so a nonnegative target is carried entirely by the
    depth-n residual (``B_hat = 0``) and a negative target entirely by the
    endpoint discrepancy (``R_hat_n = 0``). Every intermediate stays finite.
    """

    if target_gap >= 0.0:
        endpoint_n, endpoint_m = target_gap, target_gap
        values_n, values_m = (0.0, 0.0), (target_gap, target_gap)
    else:
        endpoint_n, endpoint_m = 0.0, -target_gap
        values_n, values_m = (0.0, 0.0), (-target_gap, -target_gap)
    observation = _observation(
        state_id="state-A",
        record_index=0,
        dataset_record_sha256=_STATE_A,
        gamma=0.0,
        endpoint_value_n=endpoint_n,
        endpoint_value_m=endpoint_m,
        action_values_n=values_n,
        action_values_m=values_m,
        snapshot_id=snapshot_id,
    )
    return paired_population_summary(
        snapshot_id=snapshot_id,
        population_id="synthetic_paired_population",
        deployed_depth_n=2,
        reference_depth_m=8,
        expected_population=_SINGLETON_POPULATION,
        state_diagnostics=[paired_state_diagnostic(observation)],
    )


def _mean_of_gaps(gaps: Any) -> float:
    """Drive the public seed aggregator with one module-built summary per gap."""

    seed_ids = tuple(f"seed-{offset}" for offset in range(len(gaps)))
    summaries = [
        SeedSummary(
            seed_id=seed_id,
            summary=_extreme_gap_summary(f"snapshot-{offset}", gap),
        )
        for offset, (seed_id, gap) in enumerate(zip(seed_ids, gaps))
    ]
    return equal_seed_weight_gap(
        expected_seed_ids=seed_ids, seed_summaries=summaries
    ).mean_signed_gap


def _summarize(
    observations: Any,
    *,
    snapshot_id: str = _SNAPSHOT,
    population_id: str = "synthetic_paired_population",
    deployed_depth_n: int = 2,
    reference_depth_m: int = 8,
    expected_population: Any = _EXPECTED_POPULATION,
) -> Any:
    return paired_population_summary(
        snapshot_id=snapshot_id,
        population_id=population_id,
        deployed_depth_n=deployed_depth_n,
        reference_depth_m=reference_depth_m,
        expected_population=expected_population,
        state_diagnostics=[
            paired_state_diagnostic(observation) for observation in observations
        ],
    )


class _RefusingBackend:
    """Synthetic action-value backend that refuses every forbidden call.

    ``_exact_action_values`` needs exactly two operations: enumerate one
    action's outcomes, and evaluate a *nonterminal* successor's endpoint. A
    terminal endpoint lookup, any rollout sampler, any checkpoint loader, and
    any dataset loader raise.
    """

    _FORBIDDEN = (
        "sample_bellman_rollout",
        "sample_current_policy_return",
        "sample_paired_policy_returns",
        "training_advantage_estimator",
        "persistent_endpoint_witness",
        "read_only_snapshot",
        "registered_states",
        "identity_bundle",
        "normalization_diagnostic",
        "begin_read_only_evaluation",
        "end_read_only_evaluation",
        "load_checkpoint",
        "load_dataset_records",
        "close",
    )

    def __init__(
        self,
        *,
        outcomes: dict[tuple[str, int], tuple[TheoryOutcomeV2, ...]],
        endpoints: dict[tuple[str, int], float],
        terminal_state_ids: tuple[str, ...] = (),
    ) -> None:
        self.outcomes = outcomes
        self.endpoints = endpoints
        self.terminal_state_ids = frozenset(terminal_state_ids)
        self.outcome_calls: list[tuple[str, int]] = []
        self.endpoint_calls: list[tuple[str, int]] = []

    def exact_action_outcomes(
        self, state_id: str, action_index: int
    ) -> tuple[TheoryOutcomeV2, ...]:
        self.outcome_calls.append((state_id, action_index))
        key = (state_id, action_index)
        if key not in self.outcomes:
            raise AssertionError(f"unregistered enumeration for {key!r}")
        return self.outcomes[key]

    def endpoint_value(self, state_id: object, depth: int) -> float:
        if state_id is None or state_id in self.terminal_state_ids:
            raise AssertionError(
                f"terminal endpoint bootstrap requested for {state_id!r}"
            )
        key = (state_id, depth)
        if key not in self.endpoints:
            raise AssertionError(f"unregistered endpoint lookup {key!r}")
        self.endpoint_calls.append((str(state_id), depth))
        return self.endpoints[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"forbidden synthetic backend call: {name}")


def _compatibility_backend() -> _RefusingBackend:
    """One enumeration shared by both endpoints; only the endpoints differ.

    Action 0 puts its terminal outcome first and action 2 puts its terminal
    outcome *last*, behind a nonterminal one. Both orders are needed: with only
    terminal-first actions, an implementation that hoisted its successor-value
    initialization out of the outcome loop would leak the previous nonterminal
    successor's endpoint into the terminal outcome and no assertion would
    notice.
    """

    outcomes = {
        ("state-A", 0): (
            TheoryOutcomeV2(
                probability=0.25, reward=4.0, terminal=True, next_state_id=None
            ),
            TheoryOutcomeV2(
                probability=0.75, reward=2.0, terminal=False, next_state_id="succ-a"
            ),
        ),
        ("state-A", 1): (
            TheoryOutcomeV2(
                probability=1.0, reward=-1.0, terminal=False, next_state_id="succ-b"
            ),
        ),
        ("state-A", 2): (
            TheoryOutcomeV2(
                probability=0.5, reward=1.0, terminal=False, next_state_id="succ-a"
            ),
            TheoryOutcomeV2(
                probability=0.5, reward=10.0, terminal=True, next_state_id=None
            ),
        ),
    }
    endpoints = {
        ("succ-a", 2): 8.0,
        ("succ-b", 2): -2.0,
        ("succ-a", 8): 12.0,
        ("succ-b", 8): 6.0,
        ("succ-a", 16): 100.0,
        ("succ-b", 16): -40.0,
    }
    return _RefusingBackend(
        outcomes=outcomes,
        endpoints=endpoints,
        terminal_state_ids=("terminal-absorber",),
    )


def _compatibility_state() -> TheoryStateV2:
    state = TheoryStateV2(
        state_id="state-A",
        record_index=0,
        dataset_record_sha256=_STATE_A,
        registered_state_sha256="",
        action_mask=(True, True, True, False),
        # Current, candidate, and deployed are deliberately all distinct, so a
        # fixture that read the wrong policy would not silently agree.
        current_probabilities=(0.25, 0.5, 0.25, 0.0),
        candidate_probabilities=(0.125, 0.125, 0.75, 0.0),
        deployed_probabilities=(0.5, 0.25, 0.25, 0.0),
    )
    return replace(state, registered_state_sha256=state_identity(state))


class _ExactArithmeticMixin:
    """Fraction-derived expectations with an explicit ULP budget."""

    #: Units in the last place allowed between an implementation result and the
    #: exactly derived rational value. The specified formula needs one division
    #: and one addition, so two roundings; four ULP leaves margin.
    ULP_BUDGET = 4

    def assert_close(self, actual: float, expected: Fraction, label: str) -> None:
        target = float(expected)
        tolerance = self.ULP_BUDGET * math.ulp(abs(target)) if target else 0.0
        self.assertLessEqual(
            abs(actual - target),
            tolerance,
            f"{label}: {actual!r} differs from {target!r} by more than "
            f"{self.ULP_BUDGET} ULP",
        )


class Exp1PairedStateArithmeticTest(unittest.TestCase):
    def test_expectation_is_taken_before_the_absolute_value(self) -> None:
        """Case 1: pi_base-weighted Q straddling U gives residual 0, not 1."""

        diagnostic = paired_state_diagnostic(
            _observation(
                state_id="state-A",
                record_index=0,
                dataset_record_sha256=_STATE_A,
                gamma=0.5,
                endpoint_value_n=0.0,
                endpoint_value_m=0.0,
                action_values_n=(-1.0, 1.0),
                action_values_m=(-1.0, 1.0),
            )
        )
        self.assertEqual(diagnostic.base_operator_n, 0.0)
        self.assertEqual(diagnostic.base_operator_m, 0.0)
        self.assertEqual(diagnostic.signed_residual_n, 0.0)
        self.assertEqual(diagnostic.signed_residual_m, 0.0)
        self.assertEqual(diagnostic.absolute_residual_n, 0.0)
        self.assertEqual(diagnostic.absolute_residual_m, 0.0)
        self.assertEqual(diagnostic.endpoint_discrepancy, 0.0)
        # The mean absolute TD error over the same action values is 1. That is
        # a different quantity and must not be what the module returns.
        mean_absolute_action_deviation = 0.5 * abs(-1.0 - 0.0) + 0.5 * abs(1.0 - 0.0)
        self.assertEqual(mean_absolute_action_deviation, 1.0)
        self.assertNotEqual(diagnostic.absolute_residual_n, 1.0)

    def test_signed_residual_sign_is_preserved_at_both_depths(self) -> None:
        state_a, _ = _separate_maxima_observations(0.5)
        diagnostic = paired_state_diagnostic(state_a)
        self.assertEqual(diagnostic.base_operator_n, -5.0)
        self.assertEqual(diagnostic.base_operator_m, 3.0)
        self.assertEqual(diagnostic.signed_residual_n, 5.0)
        self.assertEqual(diagnostic.signed_residual_m, 0.0)
        self.assertEqual(diagnostic.endpoint_discrepancy, 3.0)

        positive = paired_state_diagnostic(
            _observation(
                state_id="state-B",
                record_index=1,
                dataset_record_sha256=_STATE_B,
                gamma=0.5,
                endpoint_value_n=1.0,
                endpoint_value_m=1.0,
                action_values_n=(0.25, 0.75),
                action_values_m=(-1.5, -0.5),
            )
        )
        self.assertEqual(positive.signed_residual_n, 0.5)
        self.assertEqual(positive.signed_residual_m, 2.0)
        self.assertEqual(positive.absolute_residual_m, 2.0)

        # A state whose operator exceeds its endpoint at both depths: the
        # signed residuals must come back negative and only the absolute
        # residuals must be positive.
        negative = paired_state_diagnostic(
            _observation(
                state_id="state-C",
                record_index=2,
                dataset_record_sha256=_digest("state-C"),
                gamma=0.5,
                endpoint_value_n=-1.0,
                endpoint_value_m=2.0,
                action_values_n=(2.5, 3.5),
                action_values_m=(6.0, 8.0),
            )
        )
        self.assertEqual(negative.base_operator_n, 3.0)
        self.assertEqual(negative.base_operator_m, 7.0)
        self.assertEqual(negative.signed_residual_n, -4.0)
        self.assertEqual(negative.signed_residual_m, -5.0)
        self.assertLess(negative.signed_residual_n, 0.0)
        self.assertLess(negative.signed_residual_m, 0.0)
        self.assertEqual(negative.absolute_residual_n, 4.0)
        self.assertEqual(negative.absolute_residual_m, 5.0)
        self.assertEqual(negative.endpoint_discrepancy, 3.0)

    def test_masked_actions_never_enter_the_expectation(self) -> None:
        """The masked slot carries tolerance-level mass and a huge action value.

        A masked probability of exactly 0.0 would make this vacuous: dropping
        the ``if valid`` filter would add ``0.0 * anything == 0.0`` and change
        nothing. Using the largest mass the contract admits (1e-11) against a
        1e6 action value makes the filter observable: without it the operator
        would be 1.00001, not 1.0.
        """

        diagnostic = paired_state_diagnostic(
            _observation(
                state_id="state-A",
                record_index=0,
                dataset_record_sha256=_STATE_A,
                gamma=0.5,
                endpoint_value_n=0.0,
                endpoint_value_m=0.0,
                action_mask=(True, True, False),
                base_probabilities=(0.25, 0.75, 1e-11),
                action_values_n=(4.0, -4.0, 1e6),
                action_values_m=(4.0, -4.0, -1e6),
            )
        )
        self.assertEqual(diagnostic.base_operator_n, 0.25 * 4.0 + 0.75 * -4.0)
        self.assertEqual(diagnostic.base_operator_n, -2.0)
        self.assertEqual(diagnostic.base_operator_m, -2.0)
        # Unfiltered, the masked term would move each operator by 1e-5.
        self.assertNotEqual(diagnostic.base_operator_n, -2.0 + 1e-11 * 1e6)
        self.assertNotEqual(diagnostic.base_operator_m, -2.0 - 1e-11 * 1e6)

    def test_summation_matches_the_bridge_expression_not_an_accumulator(
        self,
    ) -> None:
        """Pins the documented summation contract on a compensation-sensitive case.

        CPython 3.12+ builtin ``sum`` applies Neumaier compensation, so the
        module is not a plain left-to-right accumulator. This fixture separates
        the two: the compensated answer is 0.25, the accumulator answer is 0.0.
        """

        probabilities = (0.25, 0.25, 0.25, 0.25)
        action_values = (1.0, 1e17, -1e17, 0.0)
        mask = (True, True, True, True)
        diagnostic = paired_state_diagnostic(
            _observation(
                state_id="state-A",
                record_index=0,
                dataset_record_sha256=_STATE_A,
                gamma=0.5,
                endpoint_value_n=0.0,
                endpoint_value_m=0.0,
                action_mask=mask,
                base_probabilities=probabilities,
                action_values_n=action_values,
                action_values_m=action_values,
            )
        )
        accumulator = 0.0
        for probability, action_value in zip(probabilities, action_values):
            accumulator += probability * action_value
        self.assertEqual(accumulator, 0.0)
        self.assertEqual(diagnostic.base_operator_n, 0.25)
        self.assertNotEqual(diagnostic.base_operator_n, accumulator)

        # Bit-for-bit parity with the existing bridge's horizon==1 operator
        # expression, given the same probability vector.
        bridge_operator = sum(
            probability * q_value
            for probability, q_value, valid in zip(
                probabilities, action_values, mask
            )
            if valid
        )
        self.assertEqual(diagnostic.base_operator_n, bridge_operator)

    def test_probability_tolerance_is_pinned_to_the_bridge_constant(self) -> None:
        self.assertEqual(exp1.PROBABILITY_TOLERANCE, 1e-10)
        self.assertEqual(exp1.PROBABILITY_TOLERANCE, _BRIDGE_PROBABILITY_TOLERANCE)
        self.assertEqual(exp1.BELLMAN_HORIZON_K, 1)

    def test_paired_record_carries_both_depths(self) -> None:
        """One record holds U_n, U_m, Q_n and Q_m so depths cannot mispair."""

        observation = _separate_maxima_observations(0.5)[0]
        self.assertEqual(observation.endpoint_value_n, 0.0)
        self.assertEqual(observation.endpoint_value_m, 3.0)
        self.assertEqual(observation.action_values_n, (-6.0, -4.0))
        self.assertEqual(observation.action_values_m, (2.0, 4.0))
        self.assertEqual(
            observation.member(),
            PopulationMember(
                state_id="state-A", record_index=0, dataset_record_sha256=_STATE_A
            ),
        )


class Exp1PopulationAggregationTest(unittest.TestCase, _ExactArithmeticMixin):
    def test_separate_maxima_not_maximum_of_recordwise_sums(self) -> None:
        """Case 2: gamma = 1/2 gives R_hat_n = 10, B_hat = 7, G = 3."""

        summary = _summarize(_separate_maxima_observations(0.5))
        self.assertEqual(summary.maximum_endpoint_discrepancy.value, 3.0)
        self.assertEqual(summary.maximum_absolute_residual_n.value, 5.0)
        self.assertEqual(summary.maximum_absolute_residual_m.value, 2.0)
        self.assertEqual(summary.direct_residual_bound_n, 10.0)
        self.assertEqual(summary.finite_reference_bound, 7.0)
        self.assertEqual(summary.signed_gap, 3.0)

        # The rejected alternative. max_s (D(s) + eps_m(s)/(1-gamma)) is 4,
        # attained at state B, and must never be returned as B_hat.
        recordwise = max(
            diagnostic.endpoint_discrepancy
            + diagnostic.absolute_residual_m / (1.0 - 0.5)
            for diagnostic in summary.state_diagnostics
        )
        self.assertEqual(recordwise, 4.0)
        self.assertNotEqual(summary.finite_reference_bound, recordwise)

        # The two maxima are attained at different states, which is exactly
        # what makes the two aggregates differ.
        self.assertEqual(summary.maximum_endpoint_discrepancy.state_id, "state-A")
        self.assertEqual(summary.maximum_absolute_residual_m.state_id, "state-B")
        self.assertEqual(summary.maximum_absolute_residual_n.state_id, "state-A")

    def test_negative_gap_is_preserved(self) -> None:
        """Case 3: depth-n residuals (1, 1/2) give R_hat_n = 2, B_hat = 7, G = -5."""

        summary = _summarize(
            _separate_maxima_observations(
                0.5, depth_n_action_values_a=(-1.5, -0.5)
            )
        )
        self.assertEqual(summary.maximum_absolute_residual_n.value, 1.0)
        self.assertEqual(summary.maximum_endpoint_discrepancy.value, 3.0)
        self.assertEqual(summary.maximum_absolute_residual_m.value, 2.0)
        self.assertEqual(summary.direct_residual_bound_n, 2.0)
        self.assertEqual(summary.finite_reference_bound, 7.0)
        self.assertEqual(summary.signed_gap, -5.0)
        self.assertLess(summary.signed_gap, 0.0)

    def test_gamma_zero_and_zero_gap(self) -> None:
        """Case 5a: gamma = 0 makes the penalty factor 1 and here G is exactly 0."""

        summary = _summarize(_separate_maxima_observations(0.0))
        self.assertEqual(summary.gamma, 0.0)
        self.assertEqual(summary.direct_residual_bound_n, 5.0)
        self.assertEqual(summary.finite_reference_bound, 5.0)
        self.assertEqual(summary.signed_gap, 0.0)

    def test_study_gamma_against_exact_rational_values(self) -> None:
        """Case 5b: gamma = 0.99, expected values derived with Fraction."""

        summary = _summarize(_separate_maxima_observations(0.99))
        denominator = Fraction(1) - Fraction(0.99)
        expected_direct = Fraction(5) / denominator
        expected_reference = Fraction(3) + Fraction(2) / denominator
        expected_gap = expected_direct - expected_reference
        self.assert_close(
            summary.direct_residual_bound_n, expected_direct, "R_hat_n at gamma=0.99"
        )
        self.assert_close(
            summary.finite_reference_bound,
            expected_reference,
            "B_hat at gamma=0.99",
        )
        self.assert_close(summary.signed_gap, expected_gap, "G at gamma=0.99")
        # 1.0 - 0.99 is exact in binary64 by Sterbenz, so the specified
        # denominator carries no rounding of its own.
        self.assertEqual(Fraction(1.0 - 0.99), denominator)
        self.assertGreater(summary.signed_gap, 0.0)

    def test_invalid_gamma_is_rejected(self) -> None:
        """Case 5c: only finite gamma in [0, 1) is admissible."""

        for gamma in (
            1.0,
            1.5,
            -1e-9,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            "0.5",
            None,
        ):
            with self.subTest(gamma=gamma):
                with self.assertRaises(Exp1DiagnosticsError):
                    paired_state_diagnostic(
                        _observation(
                            state_id="state-A",
                            record_index=0,
                            dataset_record_sha256=_STATE_A,
                            gamma=gamma,
                            endpoint_value_n=0.0,
                            endpoint_value_m=0.0,
                            action_values_n=(1.0, 1.0),
                            action_values_m=(1.0, 1.0),
                        )
                    )

    def test_every_maximum_is_reproducible_from_retained_state_rows(self) -> None:
        summary = _summarize(_separate_maxima_observations(0.5))
        self.assertEqual(summary.state_count, 2)
        self.assertEqual(len(summary.state_diagnostics), 2)
        for maximum, attribute in (
            (summary.maximum_endpoint_discrepancy, "endpoint_discrepancy"),
            (summary.maximum_absolute_residual_n, "absolute_residual_n"),
            (summary.maximum_absolute_residual_m, "absolute_residual_m"),
        ):
            with self.subTest(attribute=attribute):
                rows = {
                    row.state_id: getattr(row, attribute)
                    for row in summary.state_diagnostics
                }
                self.assertEqual(max(rows.values()), maximum.value)
                self.assertEqual(rows[maximum.state_id], maximum.value)
        self.assertEqual(summary.bellman_horizon, 1)
        self.assertEqual(summary.deployed_depth_n, 2)
        self.assertEqual(summary.reference_depth_m, 8)

    def test_population_completeness_is_enforced(self) -> None:
        """Case 6a: missing, extra, duplicated, and misordered identities."""

        state_a, state_b = _separate_maxima_observations(0.5)
        diagnostic_a = paired_state_diagnostic(state_a)
        diagnostic_b = paired_state_diagnostic(state_b)

        with self.assertRaisesRegex(Exp1DiagnosticsError, "size differs"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=[diagnostic_a],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "size differs"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=[diagnostic_a, diagnostic_b, diagnostic_b],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "registered population"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=[diagnostic_a, diagnostic_a],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "registered population"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=[diagnostic_b, diagnostic_a],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "repeats a state"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=(
                    _EXPECTED_POPULATION[0],
                    _EXPECTED_POPULATION[0],
                ),
                state_diagnostics=[diagnostic_a, diagnostic_a],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "must not be empty"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=(),
                state_diagnostics=[],
            )

    def test_nonfinite_state_row_is_rejected_at_every_position(self) -> None:
        """A NaN must never lose a `>` comparison and vanish from a maximum.

        `PairedStateDiagnostic` is public and has no constructor validation, so
        a record rebuilt by a caller (a JSON round-trip of the retained rows is
        the obvious route) can carry a nonfinite field. Position must not
        matter: dropping such a record would manufacture a finite G from a
        nonfinite population.
        """

        state_a, state_b = _separate_maxima_observations(0.5)
        clean = [
            paired_state_diagnostic(state_a),
            paired_state_diagnostic(state_b),
        ]
        for field in (
            "endpoint_discrepancy",
            "absolute_residual_n",
            "absolute_residual_m",
            "signed_residual_n",
            "base_operator_m",
            "endpoint_value_n",
        ):
            for position in (0, 1):
                for bad in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(field=field, position=position, bad=bad):
                        rows = list(clean)
                        rows[position] = replace(rows[position], **{field: bad})
                        with self.assertRaisesRegex(
                            Exp1DiagnosticsError,
                            f"state diagnostic {position} {field} must be finite",
                        ):
                            paired_population_summary(
                                snapshot_id=_SNAPSHOT,
                                population_id="synthetic_paired_population",
                                deployed_depth_n=2,
                                reference_depth_m=8,
                                expected_population=_EXPECTED_POPULATION,
                                state_diagnostics=rows,
                            )
        # A negative magnitude is impossible from paired_state_diagnostic but
        # reachable from a hand-built record, and must not seed a maximum.
        rows = list(clean)
        rows[1] = replace(rows[1], absolute_residual_m=-1.0)
        with self.assertRaisesRegex(Exp1DiagnosticsError, "is negative"):
            paired_population_summary(
                snapshot_id=_SNAPSHOT,
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=rows,
            )

    def test_forged_state_row_cannot_reach_a_maximum(self) -> None:
        """Review finding 1: a row must satisfy its own defining equations.

        A row with `U_n = U_m = operator_n = operator_m = 0` has every residual
        and the discrepancy at zero. Overwriting one stored magnitude used to be
        accepted, and set the primary endpoint to whatever the forger chose.
        """

        clean = paired_state_diagnostic(
            _observation(
                state_id="state-A",
                record_index=0,
                dataset_record_sha256=_STATE_A,
                gamma=0.0,
                endpoint_value_n=0.0,
                endpoint_value_m=0.0,
                action_values_n=(0.0, 0.0),
                action_values_m=(0.0, 0.0),
            )
        )
        self.assertEqual(clean.absolute_residual_n, 0.0)
        self.assertEqual(clean.endpoint_discrepancy, 0.0)

        def summarize(row: Any) -> Any:
            return paired_population_summary(
                snapshot_id=_SNAPSHOT,
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_SINGLETON_POPULATION,
                state_diagnostics=[row],
            )

        # The clean row still aggregates, and its gap is zero, not 123.
        self.assertEqual(summarize(clean).signed_gap, 0.0)

        for field, value in (
            ("absolute_residual_n", 123.0),
            ("absolute_residual_m", 123.0),
            ("endpoint_discrepancy", 123.0),
            ("signed_residual_n", 123.0),
            ("signed_residual_m", -123.0),
            ("base_operator_n", 1.0),
            ("base_operator_m", 1.0),
            ("endpoint_value_n", 1.0),
            ("endpoint_value_m", 1.0),
        ):
            with self.subTest(field=field):
                forged = replace(clean, **{field: value})
                with self.assertRaisesRegex(
                    Exp1DiagnosticsError, "does not equal the value its own"
                ):
                    summarize(forged)

    def test_integer_alias_rows_are_rejected_in_binary64(self) -> None:
        """Review finding 1: identities must be checked in binary64, not int.

        A public row may carry Python ints, and int arithmetic is arbitrary
        precision. Two ints that alias to one binary64 value make the two
        arithmetics disagree: the row's exact integer residual is nonzero while
        the binary64 residual this module computes is 0.0. Deriving from the raw
        attributes accepted such a row and let it set G.

        Both cases build a fresh summary from the row rather than editing an
        already-built one, and both are cross-checked against the canonical
        `PairedEndpointObservation` path, which reports 0.0.
        """

        for label, base, delta in (
            ("2**53 + 1 aliases 2**53", 2**53, 1),
            ("high exponent", 2**1023, 2**969),
        ):
            with self.subTest(label=label):
                # The premise: these two integers are one binary64 value.
                self.assertEqual(float(base + delta), float(base))
                self.assertNotEqual(base + delta, base)

                aliased = PairedStateDiagnostic(
                    state_id="state-A",
                    record_index=0,
                    dataset_record_sha256=_STATE_A,
                    snapshot_id=_SNAPSHOT,
                    deployed_depth_n=2,
                    reference_depth_m=8,
                    gamma=0.0,
                    endpoint_value_n=base + delta,
                    endpoint_value_m=base + delta,
                    base_operator_n=base,
                    base_operator_m=base + delta,
                    signed_residual_n=delta,
                    signed_residual_m=0,
                    absolute_residual_n=delta,
                    absolute_residual_m=0,
                    endpoint_discrepancy=0,
                )
                # Internally consistent under Python integer arithmetic...
                self.assertEqual(
                    aliased.endpoint_value_n - aliased.base_operator_n,
                    aliased.signed_residual_n,
                )
                # ...and inconsistent in binary64, which is what must decide.
                self.assertEqual(
                    float(aliased.endpoint_value_n) - float(aliased.base_operator_n),
                    0.0,
                )

                with self.assertRaisesRegex(
                    Exp1DiagnosticsError, "does not equal the value its own"
                ):
                    paired_population_summary(
                        snapshot_id=_SNAPSHOT,
                        population_id="synthetic_paired_population",
                        deployed_depth_n=2,
                        reference_depth_m=8,
                        expected_population=_SINGLETON_POPULATION,
                        state_diagnostics=[aliased],
                    )

                # The canonical path over the same values reports a zero gap.
                canonical = paired_population_summary(
                    snapshot_id=_SNAPSHOT,
                    population_id="synthetic_paired_population",
                    deployed_depth_n=2,
                    reference_depth_m=8,
                    expected_population=_SINGLETON_POPULATION,
                    state_diagnostics=[
                        paired_state_diagnostic(
                            PairedEndpointObservation(
                                state_id="state-A",
                                record_index=0,
                                dataset_record_sha256=_STATE_A,
                                snapshot_id=_SNAPSHOT,
                                deployed_depth_n=2,
                                reference_depth_m=8,
                                action_mask=(True,),
                                base_probabilities=(1.0,),
                                endpoint_value_n=base + delta,
                                endpoint_value_m=base + delta,
                                action_values_n=(base,),
                                action_values_m=(base + delta,),
                                gamma=0.0,
                            )
                        )
                    ],
                )
                self.assertEqual(canonical.signed_gap, 0.0)

    def test_integer_valued_rows_consistent_in_binary64_are_accepted(self) -> None:
        """Canonicalization rejects aliasing, not integers as such.

        A row carrying small ints whose identities hold in binary64 is
        numerically identical to the float row and stays accepted, so the fix
        is not a blanket ban on integer fields.
        """

        integral = PairedStateDiagnostic(
            state_id="state-A",
            record_index=0,
            dataset_record_sha256=_STATE_A,
            snapshot_id=_SNAPSHOT,
            deployed_depth_n=2,
            reference_depth_m=8,
            gamma=0.0,
            endpoint_value_n=5,
            endpoint_value_m=5,
            base_operator_n=3,
            base_operator_m=5,
            signed_residual_n=2,
            signed_residual_m=0,
            absolute_residual_n=2,
            absolute_residual_m=0,
            endpoint_discrepancy=0,
        )
        summary = paired_population_summary(
            snapshot_id=_SNAPSHOT,
            population_id="synthetic_paired_population",
            deployed_depth_n=2,
            reference_depth_m=8,
            expected_population=_SINGLETON_POPULATION,
            state_diagnostics=[integral],
        )
        self.assertEqual(summary.signed_gap, 2.0)
        # The reported maximum is a canonical binary64 value, not the stored int.
        self.assertIsInstance(summary.maximum_absolute_residual_n.value, float)
        self.assertEqual(summary.maximum_absolute_residual_n.value, 2.0)

    def test_state_row_metadata_lookalikes_are_rejected(self) -> None:
        """Review finding 1: `False == 0.0` and `2.0 == 2` must not pass.

        These were accepted on any row after the first, because the population
        compared raw fields by equality and ran the strict validators on row 0
        only. They do not change the numeric setting, but they contradict the
        module's stated strict input contract.
        """

        rows = [
            paired_state_diagnostic(observation)
            for observation in _separate_maxima_observations(0.0)
        ]

        def summarize(supplied: Any) -> Any:
            return paired_population_summary(
                snapshot_id=_SNAPSHOT,
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=supplied,
            )

        self.assertIsNotNone(summarize(rows))
        for position in (0, 1):
            for field, value, message in (
                ("gamma", False, "gamma must be numeric"),
                ("gamma", True, "gamma must be numeric"),
                ("deployed_depth_n", 2.0, "nonnegative integer depth"),
                ("reference_depth_m", 8.0, "nonnegative integer depth"),
                ("deployed_depth_n", True, "nonnegative integer depth"),
                ("snapshot_id", "", "nonempty string"),
                ("snapshot_id", None, "nonempty string"),
            ):
                with self.subTest(position=position, field=field, value=value):
                    corrupted = list(rows)
                    corrupted[position] = replace(
                        corrupted[position], **{field: value}
                    )
                    with self.assertRaisesRegex(Exp1DiagnosticsError, message):
                        summarize(corrupted)

    def test_population_never_pools_two_snapshots_or_mislabelled_depths(
        self,
    ) -> None:
        """Item 9 is enforced, not merely documented."""

        state_a, _ = _separate_maxima_observations(0.5, snapshot_id="snapshot-ALPHA")
        _, state_b = _separate_maxima_observations(0.5, snapshot_id="snapshot-BETA")
        pooled = [
            paired_state_diagnostic(state_a),
            paired_state_diagnostic(state_b),
        ]
        with self.assertRaisesRegex(
            Exp1DiagnosticsError, "different value-head snapshot"
        ):
            paired_population_summary(
                snapshot_id="snapshot-ALPHA",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=pooled,
            )
        # A summary label that contradicts the depth the data was computed at.
        depth_8_rows = [
            paired_state_diagnostic(observation)
            for observation in _separate_maxima_observations(0.5)
        ]
        with self.assertRaisesRegex(
            Exp1DiagnosticsError, "different endpoint depth pair"
        ):
            paired_population_summary(
                snapshot_id=_SNAPSHOT,
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=16,
                expected_population=_EXPECTED_POPULATION,
                state_diagnostics=depth_8_rows,
            )
        # And the observation itself refuses an inverted depth pair.
        with self.assertRaisesRegex(Exp1DiagnosticsError, "must exceed"):
            paired_state_diagnostic(
                _observation(
                    state_id="state-A",
                    record_index=0,
                    dataset_record_sha256=_STATE_A,
                    gamma=0.5,
                    endpoint_value_n=0.0,
                    endpoint_value_m=0.0,
                    action_values_n=(1.0, 1.0),
                    action_values_m=(1.0, 1.0),
                    deployed_depth_n=8,
                    reference_depth_m=2,
                )
            )

    def test_maximum_tie_goes_to_the_first_state_in_population_order(self) -> None:
        """Item 12's provenance pointer must be deterministic under a tie."""

        tied_a = _observation(
            state_id="state-A",
            record_index=0,
            dataset_record_sha256=_STATE_A,
            gamma=0.5,
            endpoint_value_n=0.0,
            endpoint_value_m=3.0,
            action_values_n=(-6.0, -4.0),
            action_values_m=(2.0, 4.0),
        )
        tied_b = _observation(
            state_id="state-B",
            record_index=1,
            dataset_record_sha256=_STATE_B,
            gamma=0.5,
            endpoint_value_n=0.0,
            endpoint_value_m=3.0,
            action_values_n=(-6.0, -4.0),
            action_values_m=(2.0, 4.0),
        )
        summary = _summarize((tied_a, tied_b))
        self.assertEqual(summary.maximum_endpoint_discrepancy.value, 3.0)
        self.assertEqual(summary.maximum_absolute_residual_n.value, 5.0)
        for maximum in (
            summary.maximum_endpoint_discrepancy,
            summary.maximum_absolute_residual_n,
            summary.maximum_absolute_residual_m,
        ):
            self.assertEqual(maximum.state_id, "state-A")
            self.assertEqual(maximum.record_index, 0)

    def test_population_never_pools_discounts_or_inverted_depths(self) -> None:
        state_a, state_b = _separate_maxima_observations(0.5)
        mixed_b = _observation(
            state_id="state-B",
            record_index=1,
            dataset_record_sha256=_STATE_B,
            gamma=0.99,
            endpoint_value_n=1.0,
            endpoint_value_m=1.0,
            action_values_n=(0.25, 0.75),
            action_values_m=(-1.5, -0.5),
        )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "mixes discount factors"):
            _summarize((state_a, mixed_b))
        for depth_n, depth_m in ((8, 2), (2, 2)):
            with self.subTest(depth_n=depth_n, depth_m=depth_m):
                with self.assertRaisesRegex(Exp1DiagnosticsError, "must exceed"):
                    _summarize(
                        (state_a, state_b),
                        deployed_depth_n=depth_n,
                        reference_depth_m=depth_m,
                    )


class Exp1SeedAggregationTest(unittest.TestCase):
    def _summary_with_gap(self, snapshot_id: str, target_gap: float) -> Any:
        """Build a synthetic snapshot summary whose signed gap is exact.

        At gamma = 0 the penalty factor is 1. State A carries the discrepancy
        (3) and the whole depth-n residual; state B carries the depth-m
        residual (2) and no depth-n residual. So B_hat is fixed at 5 and the
        gap is ``max_residual_n - 5``. ``target_gap`` must be at least -5 for
        the depth-n maximum to remain attainable.
        """

        residual_n = target_gap + 5.0
        self.assertGreaterEqual(residual_n, 0.0)
        state_a = _observation(
            state_id="state-A",
            record_index=0,
            dataset_record_sha256=_STATE_A,
            gamma=0.0,
            endpoint_value_n=0.0,
            endpoint_value_m=3.0,
            action_values_n=(-residual_n - 1.0, -residual_n + 1.0),
            action_values_m=(2.0, 4.0),
            snapshot_id=snapshot_id,
        )
        state_b = _observation(
            state_id="state-B",
            record_index=1,
            dataset_record_sha256=_STATE_B,
            gamma=0.0,
            endpoint_value_n=1.0,
            endpoint_value_m=1.0,
            action_values_n=(1.0, 1.0),
            action_values_m=(-1.5, -0.5),
            snapshot_id=snapshot_id,
        )
        summary = _summarize((state_a, state_b), snapshot_id=snapshot_id)
        self.assertEqual(summary.maximum_endpoint_discrepancy.value, 3.0)
        self.assertEqual(summary.maximum_absolute_residual_m.value, 2.0)
        self.assertEqual(summary.maximum_absolute_residual_n.value, residual_n)
        self.assertEqual(summary.finite_reference_bound, 5.0)
        self.assertEqual(summary.signed_gap, target_gap)
        return summary

    def test_equal_seed_weighting(self) -> None:
        """Case 4: synthetic seed gaps (-5, 3, 8) have mean 2."""

        summaries = [
            SeedSummary(
                seed_id=seed_id, summary=self._summary_with_gap(f"snapshot-{seed_id}", gap)
            )
            for seed_id, gap in (("seed-1", -5.0), ("seed-2", 3.0), ("seed-3", 8.0))
        ]
        aggregate = equal_seed_weight_gap(
            expected_seed_ids=("seed-1", "seed-2", "seed-3"),
            seed_summaries=summaries,
        )
        self.assertEqual(aggregate.seed_count, 3)
        self.assertEqual(aggregate.signed_gaps, (-5.0, 3.0, 8.0))
        self.assertEqual(aggregate.mean_signed_gap, 2.0)
        self.assertEqual(aggregate.seed_ids, ("seed-1", "seed-2", "seed-3"))
        self.assertEqual(
            aggregate.snapshot_ids,
            ("snapshot-seed-1", "snapshot-seed-2", "snapshot-seed-3"),
        )
        # Equal weight per seed, signed. Not the absolute mean, not the
        # positives-only mean, not a maximum across seeds.
        self.assertNotEqual(aggregate.mean_signed_gap, (5.0 + 3.0 + 8.0) / 3.0)
        self.assertNotEqual(aggregate.mean_signed_gap, (3.0 + 8.0) / 2.0)
        self.assertNotEqual(aggregate.mean_signed_gap, 8.0)

    def test_caller_ordering_does_not_change_the_result(self) -> None:
        summaries = [
            SeedSummary(
                seed_id=seed_id, summary=self._summary_with_gap(f"snapshot-{seed_id}", gap)
            )
            for seed_id, gap in (("seed-3", 8.0), ("seed-1", -5.0), ("seed-2", 3.0))
        ]
        aggregate = equal_seed_weight_gap(
            expected_seed_ids=("seed-1", "seed-2", "seed-3"),
            seed_summaries=summaries,
        )
        self.assertEqual(aggregate.signed_gaps, (-5.0, 3.0, 8.0))
        self.assertEqual(aggregate.mean_signed_gap, 2.0)

    def test_missing_duplicate_and_unexpected_seeds_are_rejected(self) -> None:
        gaps = {"seed-1": -5.0, "seed-2": 3.0, "seed-3": 8.0}
        built = {
            seed_id: self._summary_with_gap(f"snapshot-{seed_id}", gap)
            for seed_id, gap in gaps.items()
        }
        inventory = ("seed-1", "seed-2", "seed-3")

        with self.assertRaisesRegex(Exp1DiagnosticsError, "incomplete"):
            equal_seed_weight_gap(
                expected_seed_ids=inventory,
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=built["seed-1"]),
                    SeedSummary(seed_id="seed-2", summary=built["seed-2"]),
                ],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "duplicated"):
            equal_seed_weight_gap(
                expected_seed_ids=inventory,
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=built["seed-1"]),
                    SeedSummary(seed_id="seed-2", summary=built["seed-2"]),
                    SeedSummary(seed_id="seed-2", summary=built["seed-2"]),
                ],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "unexpected"):
            equal_seed_weight_gap(
                expected_seed_ids=inventory,
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=built["seed-1"]),
                    SeedSummary(seed_id="seed-2", summary=built["seed-2"]),
                    SeedSummary(seed_id="seed-9", summary=built["seed-3"]),
                ],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "repeats a seed"):
            equal_seed_weight_gap(
                expected_seed_ids=("seed-1", "seed-1"),
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=built["seed-1"])
                ],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "must not be empty"):
            equal_seed_weight_gap(expected_seed_ids=(), seed_summaries=[])

    def test_inconsistent_conditions_across_seeds_are_rejected(self) -> None:
        baseline = self._summary_with_gap("snapshot-1", 1.0)
        other_gamma = _summarize(
            _separate_maxima_observations(0.5, snapshot_id="snapshot-2"),
            snapshot_id="snapshot-2",
        )
        other_depths = _summarize(
            _separate_maxima_observations(
                0.0, snapshot_id="snapshot-3", reference_depth_m=16
            ),
            snapshot_id="snapshot-3",
            reference_depth_m=16,
        )
        other_population = _summarize(
            _separate_maxima_observations(0.0, snapshot_id="snapshot-4"),
            snapshot_id="snapshot-4",
            population_id="a_different_population",
        )
        for label, summary in (
            ("gamma", other_gamma),
            ("depths", other_depths),
            ("population", other_population),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    Exp1DiagnosticsError, "does not share the reference"
                ):
                    equal_seed_weight_gap(
                        expected_seed_ids=("seed-1", "seed-2"),
                        seed_summaries=[
                            SeedSummary(seed_id="seed-1", summary=baseline),
                            SeedSummary(seed_id="seed-2", summary=summary),
                        ],
                    )

    def test_population_member_identity_mismatch_across_seeds_is_rejected(
        self,
    ) -> None:
        baseline = self._summary_with_gap("snapshot-1", 1.0)
        renamed = (
            PopulationMember(
                state_id="state-A", record_index=0, dataset_record_sha256=_STATE_A
            ),
            PopulationMember(
                state_id="state-B",
                record_index=1,
                dataset_record_sha256=_digest("a-different-record"),
            ),
        )
        state_a, _ = _separate_maxima_observations(0.0, snapshot_id="snapshot-2")
        relabelled = _observation(
            state_id="state-B",
            record_index=1,
            dataset_record_sha256=_digest("a-different-record"),
            gamma=0.0,
            endpoint_value_n=1.0,
            endpoint_value_m=1.0,
            action_values_n=(0.25, 0.75),
            action_values_m=(-1.5, -0.5),
            snapshot_id="snapshot-2",
        )
        other = _summarize(
            (state_a, relabelled),
            snapshot_id="snapshot-2",
            expected_population=renamed,
        )
        with self.assertRaisesRegex(
            Exp1DiagnosticsError, "does not share the reference"
        ):
            equal_seed_weight_gap(
                expected_seed_ids=("seed-1", "seed-2"),
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=baseline),
                    SeedSummary(seed_id="seed-2", summary=other),
                ],
            )

    def test_forged_summary_is_re_derived_and_rejected(self) -> None:
        """The primary endpoint is verified against its own retained rows.

        `PairedPopulationSummary` is public and has no constructor validation,
        so the seed aggregator must not take a supplied summary's aggregates on
        faith. Each field below is corrupted independently while every other
        field stays self-consistent.
        """

        good = self._summary_with_gap("snapshot-1", 3.0)
        equal_seed_weight_gap(
            expected_seed_ids=("seed-1",),
            seed_summaries=[SeedSummary(seed_id="seed-1", summary=good)],
        )
        for field, value, message in (
            ("signed_gap", 999.0, "does not reproduce"),
            ("direct_residual_bound_n", 42.0, "does not reproduce"),
            ("finite_reference_bound", 0.0, "does not reproduce"),
            # These two now fail their own scalar validator before the rebuild,
            # so they report the specific defect rather than a generic mismatch.
            ("state_count", 99, "state count differs from the rows"),
            ("gamma", 1.5, r"gamma must lie in \[0, 1\)"),
            ("deployed_depth_n", 8, "must exceed"),
            ("bellman_horizon", 5, "not produced at K=1"),
            ("diagnostic_kind", "some_other_study_v9", "not an Experiment 1"),
        ):
            with self.subTest(field=field):
                forged = replace(good, **{field: value})
                with self.assertRaisesRegex(Exp1DiagnosticsError, message):
                    equal_seed_weight_gap(
                        expected_seed_ids=("seed-1",),
                        seed_summaries=[
                            SeedSummary(seed_id="seed-1", summary=forged)
                        ],
                    )
        # A maximum reattributed to a state that does not attain it.
        forged_maximum = replace(
            good,
            maximum_absolute_residual_n=replace(
                good.maximum_absolute_residual_n, value=42.0
            ),
        )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "does not reproduce"):
            equal_seed_weight_gap(
                expected_seed_ids=("seed-1",),
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=forged_maximum)
                ],
            )
        # A state row silently edited after the summary was built. This is
        # caught at the row validator now, before the rebuild can even run.
        forged_row = replace(
            good,
            state_diagnostics=(
                replace(good.state_diagnostics[0], absolute_residual_n=1e3),
                good.state_diagnostics[1],
            ),
        )
        with self.assertRaisesRegex(
            Exp1DiagnosticsError, "does not equal the value its own"
        ):
            equal_seed_weight_gap(
                expected_seed_ids=("seed-1",),
                seed_summaries=[SeedSummary(seed_id="seed-1", summary=forged_row)],
            )

    def test_forged_row_inside_a_self_consistent_summary_is_rejected(self) -> None:
        """Review finding 1, seed layer: the forged row must not survive here either.

        The summary below is internally consistent in every aggregate — its
        maxima, bounds, and gap all follow from the forged row it carries — so
        the rebuild-and-compare check alone would pass it. Only revalidating the
        row's own defining equations rejects it.
        """

        clean_row = paired_state_diagnostic(
            _observation(
                state_id="state-A",
                record_index=0,
                dataset_record_sha256=_STATE_A,
                gamma=0.0,
                endpoint_value_n=0.0,
                endpoint_value_m=0.0,
                action_values_n=(0.0, 0.0),
                action_values_m=(0.0, 0.0),
            )
        )
        honest = paired_population_summary(
            snapshot_id=_SNAPSHOT,
            population_id="synthetic_paired_population",
            deployed_depth_n=2,
            reference_depth_m=8,
            expected_population=_SINGLETON_POPULATION,
            state_diagnostics=[clean_row],
        )
        self.assertEqual(honest.signed_gap, 0.0)

        forged_row = replace(clean_row, absolute_residual_n=123.0)
        # Assemble a summary whose every aggregate agrees with the forged row,
        # so nothing but the row check can catch it.
        self_consistent = replace(
            honest,
            state_diagnostics=(forged_row,),
            maximum_absolute_residual_n=replace(
                honest.maximum_absolute_residual_n, value=123.0
            ),
            direct_residual_bound_n=123.0,
            signed_gap=123.0,
        )
        with self.assertRaisesRegex(
            Exp1DiagnosticsError, "does not equal the value its own"
        ):
            equal_seed_weight_gap(
                expected_seed_ids=("seed-1",),
                seed_summaries=[
                    SeedSummary(seed_id="seed-1", summary=self_consistent)
                ],
            )

    def test_summary_scalar_lookalikes_are_rejected(self) -> None:
        """Review finding 2: `False == 0.0` and `True == 1` must not pass.

        Dataclass equality delegates to Python equality, so comparing a supplied
        summary against a rebuilt one accepted bool and float lookalikes in
        every field, including inside a nested `PopulationMaximum`. The
        `gamma=False` case propagated straight into the returned aggregate.

        The fixture is a singleton population at gamma = 0, so every honest
        value is zero and every lookalike below is numerically equal to the
        value it replaces. Only type strictness can separate them.
        """

        good = _extreme_gap_summary("snapshot-1", 0.0)
        self.assertEqual(good.signed_gap, 0.0)
        self.assertEqual(good.gamma, 0.0)
        baseline = equal_seed_weight_gap(
            expected_seed_ids=("seed-1",),
            seed_summaries=[SeedSummary(seed_id="seed-1", summary=good)],
        )
        self.assertIsInstance(baseline.gamma, float)

        # Top-level scalars.
        for field, value in (
            ("gamma", False),
            ("bellman_horizon", True),
            ("bellman_horizon", 1.0),
            ("state_count", True),
            ("deployed_depth_n", 2.0),
            ("reference_depth_m", 8.0),
            ("signed_gap", False),
            ("direct_residual_bound_n", False),
            ("finite_reference_bound", False),
            ("snapshot_id", ""),
            ("population_id", ""),
        ):
            with self.subTest(scope="top-level", field=field, value=value):
                with self.assertRaises(Exp1DiagnosticsError):
                    equal_seed_weight_gap(
                        expected_seed_ids=("seed-1",),
                        seed_summaries=[
                            SeedSummary(
                                seed_id="seed-1",
                                summary=replace(good, **{field: value}),
                            )
                        ],
                    )

        # Nested PopulationMaximum witnesses, each numerically equal to the
        # honest zero-valued witness it replaces.
        honest_witness = good.maximum_absolute_residual_n
        self.assertEqual(honest_witness.value, 0.0)
        self.assertEqual(honest_witness.record_index, 0)
        for name in (
            "maximum_endpoint_discrepancy",
            "maximum_absolute_residual_n",
            "maximum_absolute_residual_m",
        ):
            for field, value in (
                ("value", False),
                ("record_index", False),
                ("state_id", ""),
                ("dataset_record_sha256", "not-a-digest"),
            ):
                with self.subTest(scope=name, field=field, value=value):
                    forged = replace(
                        good,
                        **{
                            name: replace(
                                getattr(good, name), **{field: value}
                            )
                        },
                    )
                    with self.assertRaises(Exp1DiagnosticsError):
                        equal_seed_weight_gap(
                            expected_seed_ids=("seed-1",),
                            seed_summaries=[
                                SeedSummary(seed_id="seed-1", summary=forged)
                            ],
                        )

    def test_int_for_float_lookalikes_are_caught_by_type_aware_comparison(
        self,
    ) -> None:
        """Review finding 2: the half the strict validators cannot see.

        `_finite` rejects bool but accepts int, so an int standing in for a
        float — `gamma=0`, `signed_gap=0`, a witness `value=0` — passes every
        scalar validator. Only comparing against the canonical rebuilt summary
        with matching types rejects it. Without that comparison these summaries
        are accepted, and the module's claim that public summaries are verified
        rather than trusted is false for every float field.

        These are numerically equal to the values they replace, so nothing but
        type strictness can separate them.
        """

        good = _extreme_gap_summary("snapshot-1", 0.0)
        for field in (
            "gamma",
            "signed_gap",
            "direct_residual_bound_n",
            "finite_reference_bound",
        ):
            with self.subTest(scope="top-level", field=field):
                self.assertIsInstance(getattr(good, field), float)
                self.assertEqual(getattr(good, field), 0.0)
                with self.assertRaisesRegex(
                    Exp1DiagnosticsError, "does not reproduce"
                ):
                    equal_seed_weight_gap(
                        expected_seed_ids=("seed-1",),
                        seed_summaries=[
                            SeedSummary(
                                seed_id="seed-1",
                                summary=replace(good, **{field: 0}),
                            )
                        ],
                    )
        for name in (
            "maximum_endpoint_discrepancy",
            "maximum_absolute_residual_n",
            "maximum_absolute_residual_m",
        ):
            with self.subTest(scope=name, field="value"):
                witness = getattr(good, name)
                self.assertIsInstance(witness.value, float)
                with self.assertRaisesRegex(
                    Exp1DiagnosticsError, "does not reproduce"
                ):
                    equal_seed_weight_gap(
                        expected_seed_ids=("seed-1",),
                        seed_summaries=[
                            SeedSummary(
                                seed_id="seed-1",
                                summary=replace(
                                    good,
                                    **{name: replace(witness, value=0)},
                                ),
                            )
                        ],
                    )

    def test_large_magnitude_means_do_not_overflow(self) -> None:
        """Review finding 2: a representable mean must not raise.

        `math.fsum` keeps a running partial and raises
        `intermediate overflow in fsum` when that partial leaves binary64
        range, even when the mean is representable. Every gap below is
        produced by the module from finite inputs.
        """

        self.assertEqual(_mean_of_gaps([1e308, 1e308]), 1e308)
        self.assertEqual(_mean_of_gaps([1e308, 1e308, 1e308]), 1e308)
        self.assertEqual(_mean_of_gaps([-1e308, -1e308]), -1e308)
        self.assertEqual(_mean_of_gaps([1e308, 1e308, 1e308, 1e308]), 1e308)
        # Halving is exact here, so the mean lands mid-range rather than at the
        # boundary, which pins the value rather than just the absence of a raise.
        self.assertEqual(_mean_of_gaps([1e308, 0.0]), 5e307)
        self.assertEqual(_mean_of_gaps([1.5e308, 0.5e308]), 1e308)

    def test_cancelling_means_are_permutation_invariant(self) -> None:
        """Review finding 2: order must not decide between 0.0 and a raise.

        Before the fix `(1e308, 1e308, -1e308, -1e308)` raised while the
        interleaved ordering returned 0.0, contradicting the module's
        order-independence claim.
        """

        cancelling = [1e308, 1e308, -1e308, -1e308]
        means = {
            _mean_of_gaps(list(permutation))
            for permutation in set(itertools.permutations(cancelling))
        }
        self.assertEqual(means, {0.0})
        self.assertEqual(_mean_of_gaps([1e308, -1e308, 1e308, -1e308]), 0.0)
        self.assertEqual(_mean_of_gaps([1e308, -1e308]), 0.0)

        # Ordinary magnitudes stay permutation invariant too.
        ordinary = [-5.0, 3.0, 8.0]
        ordinary_means = {
            _mean_of_gaps(list(permutation))
            for permutation in set(itertools.permutations(ordinary))
        }
        self.assertEqual(ordinary_means, {2.0})

    def test_mean_is_the_exactly_rounded_arithmetic_mean(self) -> None:
        """Summing as exact rationals rounds once, at the division."""

        for gaps in ([0.1, 0.2, 0.3], [1.0, 2.0, 4.0], [-5.0, 3.0, 8.0]):
            with self.subTest(gaps=gaps):
                expected = Fraction(0)
                for gap in gaps:
                    expected += Fraction(gap)
                self.assertEqual(
                    _mean_of_gaps(gaps), float(expected / len(gaps))
                )
        # One rounding, not two: fsum(gaps)/n rounds the total and then the
        # quotient, and differs from the correctly rounded mean here.
        self.assertEqual(_mean_of_gaps([0.1, 0.2, 0.3]), 0.2)
        self.assertNotEqual(math.fsum([0.1, 0.2, 0.3]) / 3, 0.2)


class Exp1MalformedInputTest(unittest.TestCase):
    """Case 6b: malformed, nonfinite, and unrepairable inputs are rejected."""

    def _reject(self, message: str, **overrides: Any) -> None:
        base: dict[str, Any] = {
            "state_id": "state-A",
            "record_index": 0,
            "dataset_record_sha256": _STATE_A,
            "gamma": 0.5,
            "endpoint_value_n": 0.0,
            "endpoint_value_m": 0.0,
            "action_values_n": (1.0, 1.0),
            "action_values_m": (1.0, 1.0),
        }
        base.update(overrides)
        with self.assertRaisesRegex(Exp1DiagnosticsError, message):
            paired_state_diagnostic(_observation(**base))

    def test_identity_fields(self) -> None:
        self._reject("nonempty string", state_id="")
        self._reject("nonempty string", state_id=None)
        self._reject("nonnegative integer", record_index=-1)
        self._reject("nonnegative integer", record_index=True)
        self._reject("nonnegative integer", record_index="0")
        self._reject("SHA-256", dataset_record_sha256="not-a-digest")
        self._reject("SHA-256", dataset_record_sha256=_STATE_A.upper())
        self._reject("SHA-256", dataset_record_sha256=_STATE_A[:-1])

    def test_action_inventory_and_mask(self) -> None:
        self._reject("must not be empty", action_mask=(), base_probabilities=())
        self._reject(
            "at least one valid action",
            action_mask=(False, False),
            base_probabilities=(0.0, 0.0),
        )
        self._reject("only booleans", action_mask=(1, 1))
        self._reject(
            "wrong action inventory",
            action_mask=(True, True),
            base_probabilities=(1.0,),
        )
        self._reject("wrong action inventory", action_values_n=(1.0,))
        self._reject("wrong action inventory", action_values_m=(1.0, 1.0, 1.0))
        self._reject("must be a sequence", action_mask=True)
        self._reject("must be a sequence", base_probabilities="ab")

    def test_probability_contract_is_validated_not_repaired(self) -> None:
        self._reject("is negative", base_probabilities=(-0.1, 1.1))
        self._reject("does not sum to one", base_probabilities=(0.5, 0.6))
        self._reject("does not sum to one", base_probabilities=(0.0, 0.0))
        self._reject(
            "masked action",
            action_mask=(True, True, False),
            base_probabilities=(0.5, 0.5, 1e-6),
            action_values_n=(1.0, 1.0, 0.0),
            action_values_m=(1.0, 1.0, 0.0),
        )
        # Within the stated tolerance the law is accepted unchanged: no
        # renormalization, no masked zeroing.
        leaky = _observation(
            state_id="state-A",
            record_index=0,
            dataset_record_sha256=_STATE_A,
            gamma=0.5,
            endpoint_value_n=0.0,
            endpoint_value_m=0.0,
            action_mask=(True, True, False),
            base_probabilities=(0.5, 0.5, 1e-11),
            action_values_n=(1.0, 1.0, 1e6),
            action_values_m=(1.0, 1.0, 1e6),
        )
        diagnostic = paired_state_diagnostic(leaky)
        self.assertEqual(leaky.base_probabilities, (0.5, 0.5, 1e-11))
        self.assertEqual(diagnostic.base_operator_n, 1.0)

    def test_nonfinite_inputs_and_derived_results(self) -> None:
        self._reject("must be finite", endpoint_value_n=float("nan"))
        self._reject("must be finite", endpoint_value_m=float("inf"))
        self._reject("must be finite", action_values_n=(float("nan"), 1.0))
        self._reject("must be finite", action_values_m=(1.0, float("-inf")))
        self._reject("must be numeric", action_values_n=(None, 1.0))
        # An int wider than binary64 must surface as the module's own error,
        # not as a bare OverflowError from float().
        self._reject("not representable in binary64", endpoint_value_n=10**400)
        self._reject(
            "not representable in binary64", action_values_m=(10**400, 1.0)
        )
        self._reject("must be numeric", endpoint_value_n="0.0")
        # Finite inputs whose signed residual overflows binary64 are rejected
        # rather than reported as an infinite diagnostic.
        self._reject(
            "signed residual",
            endpoint_value_n=1e308,
            action_values_n=(-1e308, -1e308),
        )

    def test_wrong_record_types_are_rejected(self) -> None:
        with self.assertRaisesRegex(Exp1DiagnosticsError, "wrong type"):
            paired_state_diagnostic(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(Exp1DiagnosticsError, "wrong type"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=_EXPECTED_POPULATION[:1],
                state_diagnostics=[object()],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "not a population member"):
            paired_population_summary(
                snapshot_id="snapshot-0",
                population_id="synthetic_paired_population",
                deployed_depth_n=2,
                reference_depth_m=8,
                expected_population=("state-A",),
                state_diagnostics=[
                    paired_state_diagnostic(_separate_maxima_observations(0.5)[0])
                ],
            )
        with self.assertRaisesRegex(Exp1DiagnosticsError, "nonempty string"):
            _summarize(_separate_maxima_observations(0.5), snapshot_id="")
        with self.assertRaisesRegex(Exp1DiagnosticsError, "nonempty string"):
            _summarize(_separate_maxima_observations(0.5), population_id="")


class Exp1PurityTest(unittest.TestCase):
    """Case 6c and case 7: immutability, no randomness, no file or data access."""

    def _run_pipeline(self) -> float:
        summary = _summarize(_separate_maxima_observations(0.5))
        aggregate = equal_seed_weight_gap(
            expected_seed_ids=("seed-1",),
            seed_summaries=[SeedSummary(seed_id="seed-1", summary=summary)],
        )
        return aggregate.mean_signed_gap

    def test_caller_sequences_are_copied_and_never_mutated(self) -> None:
        mask = [True, True]
        probabilities = [0.5, 0.5]
        action_values_n = [-6.0, -4.0]
        action_values_m = [2.0, 4.0]
        observation = _observation(
            state_id="state-A",
            record_index=0,
            dataset_record_sha256=_STATE_A,
            gamma=0.5,
            endpoint_value_n=0.0,
            endpoint_value_m=3.0,
            action_mask=mask,
            base_probabilities=probabilities,
            action_values_n=action_values_n,
            action_values_m=action_values_m,
        )
        # The record copies rather than aliases, so a later caller mutation
        # cannot reach a computed diagnostic.
        self.assertIsInstance(observation.action_mask, tuple)
        self.assertIsInstance(observation.action_values_n, tuple)

        diagnostic = paired_state_diagnostic(observation)
        expected_population = [_EXPECTED_POPULATION[0]]
        supplied = [diagnostic]
        summary = paired_population_summary(
            snapshot_id="snapshot-0",
            population_id="synthetic_paired_population",
            deployed_depth_n=2,
            reference_depth_m=8,
            expected_population=expected_population,
            state_diagnostics=supplied,
        )
        seed_ids = ["seed-1"]
        seed_summaries = [SeedSummary(seed_id="seed-1", summary=summary)]
        equal_seed_weight_gap(
            expected_seed_ids=seed_ids, seed_summaries=seed_summaries
        )

        self.assertEqual(mask, [True, True])
        self.assertEqual(probabilities, [0.5, 0.5])
        self.assertEqual(action_values_n, [-6.0, -4.0])
        self.assertEqual(action_values_m, [2.0, 4.0])
        self.assertEqual(expected_population, [_EXPECTED_POPULATION[0]])
        self.assertEqual(supplied, [diagnostic])
        self.assertEqual(seed_ids, ["seed-1"])
        self.assertEqual(len(seed_summaries), 1)

        action_values_n[0] = 1e9
        self.assertEqual(summary.state_diagnostics[0].base_operator_n, -5.0)

    def test_no_random_number_consumption(self) -> None:
        random.seed(20260909)
        before = random.getstate()
        self._run_pipeline()
        self.assertEqual(random.getstate(), before)

    def test_no_file_or_data_access(self) -> None:
        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the paired-residual core opened a file")

        with mock.patch.object(builtins, "open", _forbidden):
            self.assertEqual(self._run_pipeline(), 3.0)

    def test_module_depends_only_on_deterministic_standard_library(self) -> None:
        """`math`, plus `Fraction` and `Sequence`/`dataclass` as bare names.

        `fractions` backs the exact-rational seed mean. It is standard library
        and pulls in no source of randomness; that is a CPython property
        verified out of band rather than here, because reloading a stdlib
        module inside the suite rebinds `Fraction` and corrupts every later
        test. The behavioral guarantees are covered by
        `test_no_random_number_consumption` and `test_no_file_or_data_access`.
        """

        imported = {
            name
            for name, value in vars(exp1).items()
            if isinstance(value, types.ModuleType)
        }
        self.assertEqual(imported, {"math"})
        self.assertIs(exp1.Fraction, Fraction)
        for forbidden in ("torch", "numpy", "pydantic", "random", "os", "pathlib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, vars(exp1))


class Exp1ActionValueCompatibilityTest(unittest.TestCase):
    """Case E: the existing enumerated action values feed this arithmetic."""

    def test_shared_enumeration_terminal_folding_and_endpoint_lookup(self) -> None:
        backend = _compatibility_backend()
        state = _compatibility_state()
        q_n, q_m = _exact_action_values(
            backend=backend, state=state, n=2, m=8, gamma=0.5
        )

        # One enumeration per valid action, reused by both endpoints. The
        # masked action is never enumerated.
        self.assertEqual(
            backend.outcome_calls,
            [("state-A", 0), ("state-A", 1), ("state-A", 2)],
        )

        # Terminal outcomes fold their reward with no endpoint bootstrap;
        # nonterminal outcomes look up their own successor at each depth. Note
        # action 2 issues its lookups before its terminal outcome, so a leaked
        # bootstrap into that terminal outcome would change q.
        self.assertEqual(
            backend.endpoint_calls,
            [
                ("succ-a", 2),
                ("succ-a", 8),
                ("succ-b", 2),
                ("succ-b", 8),
                ("succ-a", 2),
                ("succ-a", 8),
            ],
        )
        self.assertEqual(q_n, [5.5, -2.0, 7.5, 0.0])
        self.assertEqual(q_m, [7.0, 2.0, 8.5, 0.0])
        # Hand-derived. Action 0: 0.25*4 (terminal, unbootstrapped) plus
        # 0.75*(2 + 0.5*U_q(succ-a)). Action 1: -1 + 0.5*U_q(succ-b).
        # Action 2: 0.5*(1 + 0.5*U_q(succ-a)) plus 0.5*10 (terminal, last).
        self.assertEqual(q_n[0], 0.25 * 4.0 + 0.75 * (2.0 + 0.5 * 8.0))
        self.assertEqual(q_m[0], 0.25 * 4.0 + 0.75 * (2.0 + 0.5 * 12.0))
        self.assertEqual(q_n[1], -1.0 + 0.5 * -2.0)
        self.assertEqual(q_m[1], -1.0 + 0.5 * 6.0)
        self.assertEqual(q_n[2], 0.5 * (1.0 + 0.5 * 8.0) + 0.5 * 10.0)
        self.assertEqual(q_m[2], 0.5 * (1.0 + 0.5 * 12.0) + 0.5 * 10.0)
        # If the terminal outcome had bootstrapped from the preceding
        # nonterminal successor, q_n[2] would be 9.5 rather than 7.5.
        self.assertNotEqual(q_n[2], 0.5 * 5.0 + 0.5 * (10.0 + 0.5 * 8.0))

    def test_comparison_depth_changes_only_the_endpoint_lookup(self) -> None:
        first = _compatibility_backend()
        second = _compatibility_backend()
        state = _compatibility_state()
        q_n_first, q_m_first = _exact_action_values(
            backend=first, state=state, n=2, m=8, gamma=0.5
        )
        q_n_second, q_m_second = _exact_action_values(
            backend=second, state=state, n=2, m=16, gamma=0.5
        )
        self.assertEqual(q_n_first, q_n_second)
        self.assertEqual(first.outcome_calls, second.outcome_calls)
        self.assertEqual(
            [call for call in first.endpoint_calls if call[1] == 2],
            [call for call in second.endpoint_calls if call[1] == 2],
        )
        self.assertEqual(q_m_first, [7.0, 2.0, 8.5, 0.0])
        self.assertEqual(q_m_second, [40.0, -21.0, 30.5, 0.0])
        self.assertNotEqual(q_m_first, q_m_second)

    def test_enumerated_action_values_feed_the_paired_arithmetic(self) -> None:
        backend = _compatibility_backend()
        state = _compatibility_state()
        q_n, q_m = _exact_action_values(
            backend=backend, state=state, n=2, m=8, gamma=0.5
        )
        diagnostic = paired_state_diagnostic(
            PairedEndpointObservation(
                state_id=state.state_id,
                record_index=state.record_index,
                dataset_record_sha256=state.dataset_record_sha256,
                snapshot_id=_SNAPSHOT,
                deployed_depth_n=2,
                reference_depth_m=8,
                action_mask=state.action_mask,
                base_probabilities=state.current_probabilities,
                endpoint_value_n=1.0,
                endpoint_value_m=3.0,
                action_values_n=q_n,
                action_values_m=q_m,
                gamma=0.5,
            )
        )
        self.assertEqual(
            diagnostic.base_operator_n, 0.25 * 5.5 + 0.5 * -2.0 + 0.25 * 7.5
        )
        self.assertEqual(diagnostic.base_operator_n, 2.25)
        self.assertEqual(
            diagnostic.base_operator_m, 0.25 * 7.0 + 0.5 * 2.0 + 0.25 * 8.5
        )
        self.assertEqual(diagnostic.base_operator_m, 4.875)
        self.assertEqual(diagnostic.signed_residual_n, -1.25)
        self.assertEqual(diagnostic.signed_residual_m, -1.875)
        self.assertEqual(diagnostic.absolute_residual_n, 1.25)
        self.assertEqual(diagnostic.absolute_residual_m, 1.875)
        self.assertEqual(diagnostic.endpoint_discrepancy, 2.0)

    def test_forbidden_backend_surfaces_are_hard_failures(self) -> None:
        backend = _compatibility_backend()
        for name in _RefusingBackend._FORBIDDEN:
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    getattr(backend, name)
        with self.assertRaises(AssertionError):
            backend.endpoint_value(None, 2)
        with self.assertRaises(AssertionError):
            backend.endpoint_value("terminal-absorber", 2)


if __name__ == "__main__":
    unittest.main()
